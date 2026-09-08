#!/usr/bin/env python3
"""
Train Split Annotation — Gate3-Style v5 (text-only, no perframe data).

Generates train annotations using the gate3 prompt template (instruction_generation_gate3)
which emphasizes room transitions over explicit turn commands.

Key improvements over train_v24 (text-only with old prompt):
- Uses instruction_generation_gate3 prompt (room transitions, max 2 waypoints)
- init_turn threshold=90° (only mention sharp/around rotations explicitly)
- Path context uses elevation/distance/turn-count
- Expected avg_explicit_turns: ~0.6-0.8 (vs ~1.8 in train_v24)

No gate3 perframe data available for train — landmarks are inferred by Gemma from path geometry.
Room names are generic ("indoor space", "corridor") but instruction style is GT-aligned.

Output: outputs/datasets/train_gate3style_v5.json.gz
Checkpoint: outputs/train_gate3style_v5_checkpoint.json
Deploy: /mnt/nvme0/.../train/train_gate3style_v5.json.gz
"""
import asyncio
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

TRAIN_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/train/train.json.gz"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "train_gate3style_v5.json.gz"
CHECKPOINT_PATH = ROOT / "outputs" / "train_gate3style_v5_checkpoint.json"

INIT_TURN_THRESHOLD = 90.0  # match v5 val_unseen
TRIPLET_TEMPS = [0.3, 0.5, 0.7]  # diversity across same-trajectory triplets

from gate2_path.path_analyzer import analyze_path, primitives_to_text
from gate4_instructions.gemma_vllm_backend import generate_batch_async
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset


def build_triplet_position_map(episodes: list) -> dict:
    """Map episode_id → triplet_position (0/1/2) within its trajectory."""
    from collections import defaultdict
    traj_eps = defaultdict(list)
    for ep in episodes:
        traj_eps[ep["trajectory_id"]].append(ep["episode_id"])
    pos_map = {}
    for traj_id, ep_ids in traj_eps.items():
        for pos, eid in enumerate(sorted(ep_ids)):
            pos_map[eid] = pos % len(TRIPLET_TEMPS)
    return pos_map


def build_train_context(ep: dict, triplet_pos: int = 0) -> dict:
    """Build v5-style context from path geometry alone (no gate3 perframe)."""
    pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
    summary = pa["summary"]
    turn_prims = [p for p in pa["primitives"] if p["type"] in ("left_turn", "right_turn")]
    initial_turn = pa.get("initial_turn")

    total_dist = summary["total_distance_m"]
    elevation = summary["elevation_change_m"]
    n_turns = len(turn_prims)

    context_lines = []

    # Initial facing — only for sharp angles (≥90°)
    if initial_turn and initial_turn["angle_deg"] >= INIT_TURN_THRESHOLD:
        it_dir = initial_turn["direction"]
        it_deg = initial_turn["angle_deg"]
        if initial_turn.get("is_around"):
            context_lines.append("Initial facing: AROUND (~180°)")
        else:
            context_lines.append(f"Initial facing: sharp {it_dir} (~{it_deg:.0f}°)")

    # Path complexity label — gives Gemma a hint for room type
    if abs(elevation) > 1.0:
        path_label = "multi-level path" + (" (stairs up)" if elevation > 0 else " (stairs down)")
    elif n_turns == 0:
        path_label = "straight indoor path"
    elif n_turns == 1:
        path_label = "L-shaped path (1 turn)"
    elif n_turns == 2:
        path_label = "Z-shaped path (2 turns)"
    else:
        path_label = f"complex path ({n_turns} turns)"
    context_lines.append(f"Path: {path_label}")

    # Distance
    dist_str = f"Distance: ~{total_dist:.1f}m"
    if elevation > 0.5:
        dist_str += f" (goes UP {elevation:.1f}m, stairs)"
    elif elevation < -0.5:
        dist_str += f" (goes DOWN {abs(elevation):.1f}m, stairs)"
    context_lines.append(dist_str)

    # Key turns — direction + angle (no landmark, since no gate3 for train)
    # Select up to 2 most significant turns
    key_turns = sorted(turn_prims, key=lambda p: -p["angle_deg"])[:2]
    if key_turns:
        wp_strs = []
        for t in key_turns:
            direction = "left" if t["type"] == "left_turn" else "right"
            angle = t["angle_deg"]
            if angle > 100:
                wp_strs.append(f"sharp {direction} turn (~{angle:.0f}°)")
            else:
                wp_strs.append(f"turn {direction}")
        context_lines.append("Turns: " + " | ".join(wp_strs))
    else:
        context_lines.append("Turns: none, walk straight")

    # No goal hint — the gate3 prompt's "MUST include a stop condition" rule handles this.
    # Without gate3 perframe scene data, any goal hint causes Gemma to copy it verbatim
    # (e.g., "stop near the destination"). No hint → Gemma invents natural stop conditions.

    return {
        "episode_id": ep["episode_id"],
        "motion_sequence": "\n".join(context_lines),
        "prompt_type": "gate3",
        "temperature": TRIPLET_TEMPS[triplet_pos],
    }


def quality_check(text: str) -> tuple:
    words = text.split()
    failures = []
    if len(words) < 8:
        failures.append(f"short({len(words)}w)")
    if len(words) > 70:
        failures.append(f"long({len(words)}w)")
    stop_words = ["stop", "wait", "halt", "stand", "pause"]
    if not any(w in text.lower() for w in stop_words):
        failures.append("no_stop")
    return len(failures) == 0, failures


def clean_output(raw: str) -> str:
    for prefix in ["Instruction:", "Navigation instruction:", "Here is", "Here's", "Answer:", "Sure", "The agent"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].lstrip(":").strip()
    sentences = re.split(r'(?<=[.!?])\s+', raw.strip())
    clean = " ".join(sentences[:4]).strip()
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean


async def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Train Split Annotation — Gate3-Style v5")
    print("Gate3 prompt + init_turn threshold=90° + path geometry context")
    print("=" * 70)

    with gzip.open(TRAIN_PATH, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]
    print(f"  {len(episodes)} train episodes loaded")

    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        print(f"  Checkpoint: {len(checkpoint)} already generated")

    print("\n[Context] Building v5-style contexts (text-only, no gate3 perframe)...")
    triplet_pos_map = build_triplet_position_map(episodes)
    tasks = []
    for ep in episodes:
        eid = str(ep["episode_id"])
        if eid in checkpoint:
            continue
        pos = triplet_pos_map.get(ep["episode_id"], 0)
        ctx = build_train_context(ep, triplet_pos=pos)
        tasks.append(ctx)

    # Sample first 3 contexts
    print("\n[Samples] First 3 contexts:")
    for task in tasks[:3]:
        print(f"\n  EP{task['episode_id']}:")
        for line in task['motion_sequence'].split('\n'):
            print(f"    {line}")

    print(f"\n  {len(tasks)} to generate, {len(checkpoint)} cached")

    if tasks:
        print(f"\n[Gemma] Generating with gate3 prompt (concurrency=16)...")
        t0 = time.time()
        new_results = await generate_batch_async(tasks, concurrency=16, progress_every=500, prompt_type="gate3")
        checkpoint.update({str(k): v for k, v in new_results.items()})
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f)
        elapsed = time.time() - t0
        print(f"  Completed {len(new_results)} in {elapsed:.1f}s ({len(new_results)/elapsed:.1f}/s)")

    all_generated = {int(k): v for k, v in checkpoint.items()}
    ok = sum(1 for v in all_generated.values() if quality_check(clean_output(v))[0])
    print(f"\n[Quality] {ok}/{len(all_generated)} pass ({100*ok/max(1,len(all_generated)):.1f}%)")

    print("\n[Samples] First 5 train instructions:")
    for ep in episodes[:5]:
        eid = ep["episode_id"]
        raw = all_generated.get(eid, "MISSING")
        clean = clean_output(raw) if raw != "MISSING" else "MISSING"
        gt_text = ep.get("instruction", {}).get("instruction_text", "")[:80]
        print(f"\n  EP{eid}:")
        print(f"    GT:  {gt_text}")
        print(f"    v5t: {clean}")

    print("\n[Assemble] Building train dataset...")
    tok = VLNTokenizer(TRAIN_PATH)
    assembled = []
    for ep in episodes:
        eid = ep["episode_id"]
        raw = all_generated.get(eid, "")
        text = clean_output(raw) if raw else ""
        if not text:
            continue
        assembled.append(assemble_episode(ep, text, tok))

    # Quality stats
    texts = [e["instruction"]["instruction_text"] for e in assembled]
    def count_explicit(t): return len(re.findall(r'\bturn (?:left|right|around|slightly)\b', t, re.I))
    explicit_turns = [count_explicit(t) for t in texts]
    avg_explicit = sum(explicit_turns) / len(texts)
    avg_words = sum(len(t.split()) for t in texts) / len(texts)
    pct_zero = sum(1 for t in explicit_turns if t == 0) / len(texts) * 100

    print(f"\n[Stats] Gate3-Style Train v5:")
    print(f"  episodes: {len(assembled)}/{len(episodes)}")
    print(f"  avg_words: {avg_words:.1f}  (GT=26.1)")
    print(f"  avg_explicit_turns: {avg_explicit:.2f}  (GT=0.68, train_v24≈1.8)")
    print(f"  pct_zero_explicit: {pct_zero:.1f}%  (GT=55%)")

    dataset = {
        "episodes": assembled,
        "instruction_vocab": data.get("instruction_vocab", {}),
        "_generation_meta": {
            "mode": "gate3_style_text_only_v5",
            "model": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
            "split": "train",
            "n_episodes": len(assembled),
            "n_skipped": len(episodes) - len(assembled),
            "quality_ok": ok,
            "avg_explicit_turns": avg_explicit,
            "avg_words": avg_words,
            "init_turn_threshold": INIT_TURN_THRESHOLD,
            "note": "text-only context (no gate3 perframe for train), gate3 prompt style",
        },
    }
    save_dataset(dataset, OUTPUT_PATH)

    NVME_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/train/train_gate3style_v5.json.gz")
    NVME_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(NVME_PATH, "wt") as f:
        json.dump(dataset, f)
    print(f"\nSaved:    {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size//1024} KB)")
    print(f"Deployed: {NVME_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
