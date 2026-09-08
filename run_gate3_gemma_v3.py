#!/usr/bin/env python3
"""
Gate3-Informed Gemma Annotation v3 — GT-aligned style with selective turns.

Key improvements over v2:
- Room sequence context (stairs → hallway → closet) instead of listing all turns
- Selective turn mention: only key turns (doorway transitions, distinctive landmarks), max 2
- Initial facing encoded naturally ("Initial facing: LEFT ~40°")
- New GT-aligned prompt template (instruction_generation_gate3) that prefers room transitions
- Per-triplet temperature variation for same-trajectory instruction diversity

Expected: avg_explicit_turns closer to 1.0–1.4, more natural language, better SR.

Output: outputs/datasets/val_unseen_gate3_gemma_v3.json.gz
Checkpoint: outputs/gate3_gemma_v3_checkpoint.json
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
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_gate3_gemma_v3.json.gz"
CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v3_checkpoint.json"
PERFRAME_DIR = ROOT / "outputs" / "gate3_perframe"
LANDMARK_DIR = ROOT / "outputs" / "gate3_landmarks"

from gate2_path.path_analyzer import analyze_path
from gate4_instructions.gemma_vllm_backend import generate_batch_async
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset


def _is_distinctive_landmark(lm: str) -> bool:
    """True if landmark is specific enough to be a useful navigation cue."""
    if not lm:
        return False
    generic = {"none visible", "none", "wall", "floor", "ceiling", "door", "window", "light"}
    lm_lower = lm.lower().strip()
    if lm_lower in generic:
        return False
    if len(lm_lower) < 4:
        return False
    return True


def _is_key_turn(pf_turn: dict, prim: dict) -> bool:
    """Select a turn as 'key' if it has a room transition OR a distinctive landmark."""
    room_trans = pf_turn.get("room_transition", "").lower()
    if room_trans and "none" not in room_trans and room_trans != "":
        return True
    lm = pf_turn.get("main_landmark", "")
    if _is_distinctive_landmark(lm) and prim.get("angle_deg", 0) >= 30:
        return True
    return False


def build_room_sequence(start_room: str, turns_pf: list, goal_room: str) -> str:
    """Build compact room sequence string: 'hallway → living room → bedroom'."""
    rooms = [start_room]
    for t in turns_pf:
        r = (t.get("room") or "").lower().strip()
        if r and r != rooms[-1] and r != "none":
            rooms.append(r)
    if goal_room and goal_room != rooms[-1] and goal_room != "none":
        rooms.append(goal_room)
    # deduplicate consecutive identical
    deduped = [rooms[0]]
    for r in rooms[1:]:
        if r != deduped[-1]:
            deduped.append(r)
    if len(deduped) == 1:
        return deduped[0]
    return " → ".join(deduped)


def build_context_v3(episode_id: int, ep: dict, perframe: dict, landmark: dict) -> Optional[dict]:
    """Build v3 context: room sequence + selective key turns + natural initial facing."""
    pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
    summary = pa["summary"]
    turn_prims = [p for p in pa["primitives"] if p["type"] in ("left_turn", "right_turn")]
    initial_turn = pa.get("initial_turn")

    start = perframe.get("start", {})
    turns_pf = perframe.get("turns", [])
    goal = perframe.get("goal", {})
    lm_goal = landmark.get("goal_landmark", {})

    start_room = (start.get("room") or "room").lower()
    start_lm = start.get("main_landmark", "")

    goal_room = (goal.get("room") or lm_goal.get("room_type") or "room").lower()
    stop_lm = (goal.get("stop_landmark") or lm_goal.get("stop_landmark") or "").strip()
    stop_desc = (goal.get("stop_description") or lm_goal.get("direction_hint") or "").strip()

    total_dist = summary["total_distance_m"]
    elevation = summary["elevation_change_m"]

    context_lines = []

    # Initial facing — natural phrasing, not "Initial orientation:"
    if initial_turn:
        it_dir = initial_turn["direction"]
        it_deg = initial_turn["angle_deg"]
        if initial_turn.get("is_around"):
            context_lines.append(f"Initial facing: turn AROUND (~180°)")
        elif it_deg > 75:
            context_lines.append(f"Initial facing: sharp {it_dir} (~{it_deg:.0f}°)")
        else:
            context_lines.append(f"Initial facing: {it_dir} (~{it_deg:.0f}°)")

    # Room sequence
    room_seq = build_room_sequence(start_room, turns_pf, goal_room)
    context_lines.append(f"Room sequence: {room_seq}")

    # Start landmark
    start_line = f"Start: {start_room}"
    if _is_distinctive_landmark(start_lm):
        start_line += f", near the {start_lm}"
    context_lines.append(start_line)

    # Path distance + elevation
    dist_str = f"Distance: ~{total_dist:.1f}m"
    if elevation > 0.5:
        dist_str += f" (goes UP {elevation:.1f}m, stairs)"
    elif elevation < -0.5:
        dist_str += f" (goes DOWN {abs(elevation):.1f}m, stairs)"
    context_lines.append(dist_str)

    # Select KEY turns only (max 2)
    key_turns = []
    for i, prim in enumerate(turn_prims):
        pf_t = turns_pf[i] if i < len(turns_pf) else {}
        if _is_key_turn(pf_t, prim):
            direction = "left" if prim["type"] == "left_turn" else "right"
            t_room = (pf_t.get("room") or "").lower()
            t_lm = pf_t.get("main_landmark", "")
            t_trans = pf_t.get("room_transition", "")
            key_turns.append({
                "direction": direction,
                "room": t_room,
                "landmark": t_lm if _is_distinctive_landmark(t_lm) else "",
                "room_transition": t_trans if "none" not in t_trans.lower() else "",
            })

    # Limit to 2 key turns
    key_turns = key_turns[:2]

    if not key_turns:
        n_total = len(turn_prims)
        if n_total > 0:
            context_lines.append(f"Navigation: {n_total} turn(s), walk straight through rooms")
        else:
            context_lines.append("Navigation: walk straight, no turns")
    else:
        wp_strs = []
        for t in key_turns:
            wp = ""
            if t["room_transition"]:
                wp = t["room_transition"]
                if t["landmark"]:
                    wp += f" near the {t['landmark']}"
            elif t["landmark"]:
                wp = f"turn {t['direction']} at the {t['landmark']}"
                if t["room"]:
                    wp += f" (into {t['room']})"
            elif t["room"]:
                wp = f"enter the {t['room']}"
            if wp:
                wp_strs.append(wp)
        if wp_strs:
            context_lines.append("Key waypoints: " + " | ".join(wp_strs))

    # Goal
    goal_line = f"Goal: {goal_room}"
    if _is_distinctive_landmark(stop_lm):
        goal_line += f", stop near the {stop_lm}"
    if stop_desc:
        goal_line += f". {stop_desc}"
    context_lines.append(goal_line)

    return {
        "episode_id": episode_id,
        "motion_sequence": "\n".join(context_lines),
        "prompt_type": "gate3",
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
    print("Gate3-Informed Gemma Annotation v3 — val_unseen")
    print("v3: room-sequence context, selective turns (max 2), GT-aligned prompt")
    print("=" * 70)

    with gzip.open(GT_PATH, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]
    print(f"  {len(episodes)} episodes loaded")

    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        print(f"  Checkpoint: {len(checkpoint)} already generated")

    print("\n[Context] Building v3 gate3-informed contexts...")
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
        ctx = build_context_v3(int(eid), ep, perframe, landmark)
        if ctx:
            tasks.append(ctx)

    print(f"  {len(tasks)} to generate, {len(checkpoint)} cached, {missing_gate3} missing gate3")

    # Show sample contexts
    print("\n[Samples] First 3 contexts:")
    for task in tasks[:3]:
        print(f"\n  EP{task['episode_id']}:")
        for line in task['motion_sequence'].split('\n'):
            print(f"    {line}")

    if tasks:
        print(f"\n[Gemma] Generating with gate3 prompt (concurrency=16)...")
        t0 = time.time()
        new_results = await generate_batch_async(tasks, concurrency=16, progress_every=200, prompt_type="gate3")
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
        print(f"    G3v3: {clean}")

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

    # Quality stats vs MetadataReproducer
    texts = [e["instruction"]["instruction_text"] for e in assembled]
    def count_explicit(t): return len(re.findall(r'\bturn (?:left|right|around|slightly)\b', t, re.I))
    explicit_turns = [count_explicit(t) for t in texts]
    avg_explicit = sum(explicit_turns) / len(texts)
    avg_words = sum(len(t.split()) for t in texts) / len(texts)
    pct_3plus = sum(1 for t in explicit_turns if t >= 3) / len(texts) * 100
    pct_zero = sum(1 for t in explicit_turns if t == 0) / len(texts) * 100

    print(f"\n[Stats] Gate3-Gemma v3:")
    print(f"  episodes: {len(assembled)}/{len(episodes)}")
    print(f"  avg_words: {avg_words:.1f}  (GT=26.8, v218=27.4, v2=29.3)")
    print(f"  avg_explicit_turns: {avg_explicit:.2f}  (GT=0.66, v218=1.97, v2=2.13)")
    print(f"  pct_zero_explicit: {pct_zero:.1f}%  (GT=56%, v218=?)")
    print(f"  pct_3plus_turns: {pct_3plus:.1f}%  (GT=3.8%, v2=37.4%)")

    dataset = {
        "episodes": assembled,
        "instruction_vocab": data.get("instruction_vocab", {}),
        "_generation_meta": {
            "mode": "gate3_gemma_v3",
            "model": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
            "split": "val_unseen",
            "n_episodes": len(assembled),
            "n_skipped": len(episodes) - len(assembled),
            "quality_ok": ok,
            "avg_explicit_turns": avg_explicit,
            "avg_words": avg_words,
            "v3_changes": "room-sequence context, selective turns max-2, GT-aligned prompt",
        },
    }
    save_dataset(dataset, OUTPUT_PATH)

    NVME_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_gate3_gemma_v3.json.gz")
    NVME_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(NVME_PATH, "wt") as f:
        json.dump(dataset, f)
    print(f"\nSaved:    {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size//1024} KB)")
    print(f"Deployed: {NVME_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
