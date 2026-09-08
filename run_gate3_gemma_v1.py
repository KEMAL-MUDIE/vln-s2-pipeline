#!/usr/bin/env python3
"""
Gate3-Informed Gemma Annotation v2 — Natural language from visual context.
v2 fix: include initial_turn (initial orientation alignment) in context.
~83% of episodes need a pre-walk rotation; v1 missed this → severe SR penalty.

Unlike MetadataReproducer (rule-based templates), this feeds gate3 perframe
visual context (room, landmarks, turn cues, stop description) to Gemma and
lets it generate natural language instructions.

Unlike pure Gemma visual (run_gate4_visual_v24.py), this uses TEXT context
from gate3 VLM descriptions rather than actual rendered frames — so it works
without needing rendered frames for train/val_seen splits.

Output: outputs/datasets/val_unseen_gate3_gemma_v2.json.gz
Checkpoint: outputs/gate3_gemma_v2_checkpoint.json
"""
import asyncio
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_gate3_gemma_v2.json.gz"
CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v2_checkpoint.json"
PERFRAME_DIR = ROOT / "outputs" / "gate3_perframe"
LANDMARK_DIR = ROOT / "outputs" / "gate3_landmarks"

from gate2_path.path_analyzer import analyze_path
from gate4_instructions.gemma_vllm_backend import generate_batch_async
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset


def build_context(episode_id: int, ep: dict, perframe: dict, landmark: dict) -> Optional[dict]:
    """Build structured context from gate3 data + path analysis."""
    pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
    summary = pa["summary"]
    turn_prims = [p for p in pa["primitives"] if p["type"] in ("left_turn", "right_turn")]
    initial_turn = pa.get("initial_turn")  # initial orientation alignment, None if already aligned

    start = perframe.get("start", {})
    turns_pf = perframe.get("turns", [])
    goal = perframe.get("goal", {})
    lm_goal = landmark.get("goal_landmark", {})

    # Start context
    start_room = (start.get("room") or "room").lower()
    start_lm = start.get("main_landmark", "")
    start_lms = start.get("landmarks", [])

    # Turn context: merge path direction with perframe room/landmark info
    turn_items = []
    for i, prim in enumerate(turn_prims):
        direction = "left" if prim["type"] == "left_turn" else "right"
        pf_t = turns_pf[i] if i < len(turns_pf) else {}
        turn_room = (pf_t.get("room") or "").lower()
        turn_lm = pf_t.get("main_landmark", "")
        room_trans = pf_t.get("room_transition", "")
        turn_items.append({
            "direction": direction,
            "room": turn_room,
            "landmark": turn_lm,
            "room_transition": room_trans,
            "angle_deg": prim["angle_deg"],
        })

    # Goal context — prefer gate3_perframe goal > gate3_landmarks goal
    goal_room = (goal.get("room") or lm_goal.get("room_type") or "room").lower()
    stop_lm = (goal.get("stop_landmark") or lm_goal.get("stop_landmark") or "").strip()
    stop_desc = (goal.get("stop_description") or lm_goal.get("direction_hint") or "").strip()
    stop_lms = goal.get("landmarks", []) or lm_goal.get("landmarks", [])

    total_dist = summary["total_distance_m"]
    elevation = summary["elevation_change_m"]

    # Format as structured context string for Gemma
    context_lines = []

    # Initial orientation alignment (CRITICAL: ~83% of episodes need this)
    if initial_turn:
        it_dir = initial_turn["direction"]
        it_deg = initial_turn["angle_deg"]
        if initial_turn.get("is_around"):
            context_lines.append(f"Initial orientation: turn AROUND (~180°) before starting")
        elif it_deg > 75:
            context_lines.append(f"Initial orientation: turn {it_dir} sharply (~{it_deg:.0f}°) before starting")
        else:
            context_lines.append(f"Initial orientation: turn {it_dir} (~{it_deg:.0f}°) before starting")

    # Start
    start_line = f"Start: {start_room}"
    if start_lm:
        start_line += f", near the {start_lm}"
    if start_lms:
        start_line += f" (also visible: {', '.join(start_lms[:2])})"
    context_lines.append(start_line)

    # Path geometry
    dist_str = f"Total ~{total_dist:.1f}m"
    if elevation > 0.5:
        dist_str += f", goes UP {elevation:.1f}m (stairs)"
    elif elevation < -0.5:
        dist_str += f", goes DOWN {abs(elevation):.1f}m (stairs)"
    context_lines.append(dist_str)

    # Turns
    if not turn_items:
        context_lines.append("No turns needed — walk straight.")
    else:
        context_lines.append(f"Turns ({len(turn_items)} total):")
        for j, t in enumerate(turn_items):
            angle = t["angle_deg"]
            direction = t["direction"]
            t_room = t["room"]
            t_lm = t["landmark"]
            t_trans = t["room_transition"]
            turn_desc = f"  Turn {j+1}: turn {direction}"
            if t_lm:
                turn_desc += f" at the {t_lm}"
            if t_room:
                turn_desc += f" (in {t_room}"
                if t_trans:
                    turn_desc += f", {t_trans}"
                turn_desc += ")"
            elif t_trans:
                turn_desc += f" ({t_trans})"
            context_lines.append(turn_desc)

    # Goal
    goal_line = f"Goal: {goal_room}"
    if stop_lm:
        goal_line += f", stop near the {stop_lm}"
    if stop_desc:
        goal_line += f". {stop_desc}"
    elif stop_lms:
        goal_line += f" (visible: {', '.join(stop_lms[:2])})"
    context_lines.append(goal_line)

    return {
        "episode_id": episode_id,
        "motion_sequence": "\n".join(context_lines),
        "scene_context": f"Indoor navigation path, {len(turn_items)} turn(s), ~{total_dist:.1f}m",
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
    print("Gate3-Informed Gemma Annotation v2 — val_unseen")
    print("v2 fix: initial_turn orientation alignment included in context")
    print("Uses gate3 visual context (room/landmarks/stop) → Gemma free-text")
    print("=" * 70)

    with gzip.open(GT_PATH, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]
    print(f"  {len(episodes)} episodes loaded")

    # Load checkpoint
    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        print(f"  Checkpoint: {len(checkpoint)} already generated")

    # Build tasks
    print("\n[Context] Building gate3-informed contexts...")
    tasks = []
    missing_gate3 = 0
    for ep in episodes:
        eid = str(ep["episode_id"])
        if eid in checkpoint:
            continue
        pf_path = PERFRAME_DIR / f"episode_{int(eid):06d}.json"
        lm_path = LANDMARK_DIR / f"episode_{int(eid):06d}.json"
        perframe = json.loads(pf_path.read_text()) if pf_path.exists() else {}
        landmark = json.loads(lm_path.read_text()) if lm_path.exists() else {}
        if not perframe:
            missing_gate3 += 1
        ctx = build_context(int(eid), ep, perframe, landmark)
        if ctx:
            tasks.append(ctx)

    print(f"  {len(tasks)} to generate, {len(checkpoint)} cached, {missing_gate3} missing gate3")

    # Generate
    if tasks:
        print(f"\n[Gemma] Generating (concurrency=16)...")
        t0 = time.time()
        new_results = await generate_batch_async(tasks, concurrency=16, progress_every=200)
        checkpoint.update({str(k): v for k, v in new_results.items()})
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f)
        elapsed = time.time() - t0
        print(f"  Completed {len(new_results)} in {elapsed:.1f}s ({len(new_results)/elapsed:.1f}/s)")

    all_generated = {int(k): v for k, v in checkpoint.items()}
    ok = sum(1 for v in all_generated.values() if quality_check(clean_output(v))[0])
    print(f"\n[Quality] {ok}/{len(all_generated)} pass ({100*ok/max(1,len(all_generated)):.1f}%)")

    # Sample comparison
    print("\n[Samples] First 5 episodes:")
    for ep in episodes[:5]:
        eid = ep["episode_id"]
        raw = all_generated.get(eid, "MISSING")
        clean = clean_output(raw) if raw != "MISSING" else "MISSING"
        gt_text = ep.get("instruction", {}).get("instruction_text", "")[:80]
        print(f"\n  EP{eid}:")
        print(f"    GT:  {gt_text}")
        print(f"    G3G: {clean}")

    # Assemble dataset
    print("\n[Assemble] Building dataset...")
    tok = VLNTokenizer(GT_PATH)
    assembled = []
    for ep in episodes:
        eid = ep["episode_id"]
        raw = all_generated.get(eid, "")
        text = clean_output(raw) if raw else ""
        if not text:
            continue
        assembled.append(assemble_episode(ep, text, tok))

    # Analyze quality vs MetadataReproducer
    texts = [e["instruction"]["instruction_text"] for e in assembled]
    def count_explicit(t): return len(re.findall(r'\bturn (?:left|right|around|slightly)\b', t, re.I))
    explicit_turns = [count_explicit(t) for t in texts]
    avg_explicit = sum(explicit_turns) / len(texts)
    avg_words = sum(len(t.split()) for t in texts) / len(texts)
    pct_3plus = sum(1 for t in explicit_turns if t >= 3) / len(texts) * 100

    print(f"\n[Stats] Gate3-Gemma v2:")
    print(f"  episodes: {len(assembled)}/{len(episodes)}")
    print(f"  avg_words: {avg_words:.1f}  (GT=26.8, v218=27.4)")
    print(f"  avg_explicit_turns: {avg_explicit:.2f}  (GT=0.66, v218=1.97)")
    print(f"  pct_3plus_turns: {pct_3plus:.1f}%  (GT=3.8%)")

    dataset = {
        "episodes": assembled,
        "instruction_vocab": data.get("instruction_vocab", {}),
        "_generation_meta": {
            "mode": "gate3_gemma_v2",
            "model": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
            "split": "val_unseen",
            "n_episodes": len(assembled),
            "n_skipped": len(episodes) - len(assembled),
            "quality_ok": ok,
            "avg_explicit_turns": avg_explicit,
            "avg_words": avg_words,
            "v2_fix": "includes initial_turn orientation alignment",
        },
    }
    save_dataset(dataset, OUTPUT_PATH)

    # Deploy to NVMe
    NVME_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_gate3_gemma_v2.json.gz")
    NVME_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(NVME_PATH, "wt") as f:
        json.dump(dataset, f)
    print(f"\nSaved:    {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size//1024} KB)")
    print(f"Deployed: {NVME_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
