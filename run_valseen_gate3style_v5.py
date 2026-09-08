#!/usr/bin/env python3
"""
Val_seen Split Annotation — Gate3-Style v5 (text-only, no perframe data).
778 episodes — quick run (~2 min at 20/s).
Mirrors run_train_gate3style_v5.py but for val_seen.
Output: outputs/datasets/val_seen_gate3style_v5.json.gz
"""
import asyncio
import gzip
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

VAL_SEEN_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_seen/val_seen.json.gz"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_seen_gate3style_v5.json.gz"
CHECKPOINT_PATH = ROOT / "outputs" / "val_seen_gate3style_v5_checkpoint.json"

INIT_TURN_THRESHOLD = 90.0
TRIPLET_TEMPS = [0.3, 0.5, 0.7]

from gate2_path.path_analyzer import analyze_path
from gate4_instructions.gemma_vllm_backend import generate_batch_async
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset


def build_triplet_position_map(episodes):
    traj_eps = defaultdict(list)
    for ep in episodes:
        traj_eps[ep["trajectory_id"]].append(ep["episode_id"])
    pos_map = {}
    for traj_id, ep_ids in traj_eps.items():
        for pos, eid in enumerate(sorted(ep_ids)):
            pos_map[eid] = pos % len(TRIPLET_TEMPS)
    return pos_map


def build_context(ep: dict, triplet_pos: int = 0) -> dict:
    pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
    summary = pa["summary"]
    turn_prims = [p for p in pa["primitives"] if p["type"] in ("left_turn", "right_turn")]
    initial_turn = pa.get("initial_turn")
    total_dist = summary["total_distance_m"]
    elevation = summary["elevation_change_m"]
    n_turns = len(turn_prims)
    context_lines = []

    if initial_turn and initial_turn["angle_deg"] >= INIT_TURN_THRESHOLD:
        it_dir = initial_turn["direction"]
        it_deg = initial_turn["angle_deg"]
        if initial_turn.get("is_around"):
            context_lines.append("Initial facing: AROUND (~180°)")
        else:
            context_lines.append(f"Initial facing: sharp {it_dir} (~{it_deg:.0f}°)")

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

    dist_str = f"Distance: ~{total_dist:.1f}m"
    if elevation > 0.5:
        dist_str += f" (goes UP {elevation:.1f}m, stairs)"
    elif elevation < -0.5:
        dist_str += f" (goes DOWN {abs(elevation):.1f}m, stairs)"
    context_lines.append(dist_str)

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
    # No goal hint — gate3 prompt handles stop condition generation naturally

    return {
        "episode_id": ep["episode_id"],
        "motion_sequence": "\n".join(context_lines),
        "prompt_type": "gate3",
        "temperature": TRIPLET_TEMPS[triplet_pos],
    }


def quality_check(text):
    words = text.split()
    if len(words) < 8 or len(words) > 70:
        return False
    return any(w in text.lower() for w in ["stop", "wait", "halt", "stand", "pause"])


def clean_output(raw):
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
    print("Val_seen Annotation — Gate3-Style v5 (778 eps)")
    print("=" * 70)

    with gzip.open(VAL_SEEN_PATH, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]
    print(f"  {len(episodes)} val_seen episodes")

    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        print(f"  Checkpoint: {len(checkpoint)} already done")

    triplet_pos_map = build_triplet_position_map(episodes)
    tasks = []
    for ep in episodes:
        if str(ep["episode_id"]) in checkpoint:
            continue
        pos = triplet_pos_map.get(ep["episode_id"], 0)
        tasks.append(build_context(ep, pos))

    print(f"  {len(tasks)} to generate, {len(checkpoint)} cached")

    if tasks:
        print(f"\n[Gemma] Generating (gate3 prompt, concurrency=16)...")
        t0 = time.time()
        new_results = await generate_batch_async(tasks, concurrency=16, progress_every=200, prompt_type="gate3")
        checkpoint.update({str(k): v for k, v in new_results.items()})
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f)
        elapsed = time.time() - t0
        print(f"  Done {len(new_results)} in {elapsed:.1f}s ({len(new_results)/elapsed:.1f}/s)")

    all_gen = {int(k): v for k, v in checkpoint.items()}
    tok = VLNTokenizer(VAL_SEEN_PATH)
    assembled = []
    for ep in episodes:
        raw = all_gen.get(ep["episode_id"], "")
        text = clean_output(raw) if raw else ""
        if text:
            assembled.append(assemble_episode(ep, text, tok))

    texts = [e["instruction"]["instruction_text"] for e in assembled]
    def count_explicit(t): return len(re.findall(r'\bturn (?:left|right|around|slightly)\b', t, re.I))
    explicit = [count_explicit(t) for t in texts]
    avg_explicit = sum(explicit) / len(texts)
    avg_words = sum(len(t.split()) for t in texts) / len(texts)
    pct_zero = sum(1 for t in explicit if t == 0) / len(texts) * 100
    print(f"\n[Stats] val_seen gate3-style v5:")
    print(f"  episodes: {len(assembled)}/{len(episodes)}")
    print(f"  avg_words: {avg_words:.1f}")
    print(f"  avg_explicit: {avg_explicit:.2f}  (GT=0.68)")
    print(f"  pct_zero: {pct_zero:.1f}%  (GT=55%)")

    dataset = {
        "episodes": assembled,
        "instruction_vocab": data.get("instruction_vocab", {}),
        "_generation_meta": {
            "mode": "gate3_style_text_only_v5",
            "split": "val_seen",
            "n_episodes": len(assembled),
            "avg_explicit": avg_explicit,
            "avg_words": avg_words,
            "init_turn_threshold": INIT_TURN_THRESHOLD,
        },
    }
    save_dataset(dataset, OUTPUT_PATH)

    NVME_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_seen/val_seen_gate3style_v5.json.gz")
    NVME_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(NVME_PATH, "wt") as f:
        json.dump(dataset, f)
    print(f"\nSaved:    {OUTPUT_PATH}")
    print(f"Deployed: {NVME_PATH}")

    print("\n[Samples]")
    for ep in episodes[:5]:
        raw = all_gen.get(ep["episode_id"], "MISSING")
        gt = ep.get("instruction", {}).get("instruction_text", "")[:80]
        print(f"\n  EP{ep['episode_id']}:")
        print(f"    GT:  {gt}")
        print(f"    v5s: {clean_output(raw) if raw != 'MISSING' else 'MISSING'}")


if __name__ == "__main__":
    asyncio.run(main())
