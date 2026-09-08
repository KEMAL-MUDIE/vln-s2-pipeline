#!/usr/bin/env python3
"""
Gate3-Gemma v6 — GT-calibrated init_turn suppression.

v5 → v6 change:
  Suppress init_turn context for 20% of 90°+ episodes (episode_id % 10 < 2).
  This matches GT's ~53% skip rate for 90°+ init_turns:
    - v5 pct_zero=45.6% → v6 pct_zero≈57.2% (GT=56%)
    - v5 avg_explicit=0.54 → v6 avg_explicit≈0.44 (GT=0.66)
  Only 212/1839 episodes need regeneration (rest reuse v5 checkpoint).

Method: for suppressed episodes, use v5 context minus the "Initial facing:" line.
The gate3 perframe data (rooms/landmarks/stops) is unchanged.
"""
import asyncio
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional, Dict

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

VAL_UNSEEN_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz"
GATE3_PERFRAME_DIR = ROOT / "outputs" / "gate3_perframe"
LANDMARK_DIR = ROOT / "outputs" / "gate3_landmarks"

# V5 checkpoint — source for instructions we reuse
V5_CHECKPOINT = ROOT / "outputs" / "gate3_gemma_v5_checkpoint.json"
# V6 checkpoint — new instructions (only ~212 episodes)
CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v6_checkpoint.json"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_gate3_gemma_v6.json.gz"

INIT_TURN_CONTEXT_THRESHOLD = 90.0
# Suppress init_turn for 20% of 90°+ episodes → pct_zero ≈ 57%
SUPPRESS_RATE = 0.20  # 20% → episode_id % 10 < 2

from gate2_path.path_analyzer import analyze_path
from gate4_instructions.gemma_vllm_backend import generate_batch_async
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset

# ── Imports shared with v5 context builder ──────────────────────────────────
from run_gate3_gemma_v5 import (
    _is_distinctive_landmark,
    _is_key_turn_v5,
    _has_new_room,
    build_room_sequence,
    build_context_v5,
    find_triplets,
    TRIPLET_TEMPS,
)


def _should_suppress_init_turn(episode_id: int, angle_deg: float, is_around: bool) -> bool:
    """Deterministic suppression: 20% of 90°+ episodes based on episode_id hash."""
    if angle_deg < INIT_TURN_CONTEXT_THRESHOLD:
        return False
    return (episode_id % 10) < 2  # 20% suppression


def build_context_v6(episode_id: int, ep: dict, perframe: dict, landmark: dict) -> Optional[dict]:
    """
    v6: same as v5 but with stochastic init_turn suppression for 90°+ episodes.
    20% of 90°+ episodes get NO init_turn context → room-transition-only instructions.
    """
    pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
    initial_turn = pa.get("initial_turn")

    # Check if this episode should suppress init_turn
    if initial_turn and initial_turn["angle_deg"] >= INIT_TURN_CONTEXT_THRESHOLD:
        should_suppress = _should_suppress_init_turn(
            episode_id, initial_turn["angle_deg"], initial_turn.get("is_around", False)
        )
    else:
        should_suppress = False

    if not should_suppress:
        # Reuse v5 context exactly (caller should use v5 checkpoint for these)
        return build_context_v5(episode_id, ep, perframe, landmark)

    # Build context WITHOUT init_turn line
    summary = pa["summary"]
    turn_prims = [p for p in pa["primitives"] if p["type"] in ("left_turn", "right_turn")]

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
    # NOTE: NO init_turn line (suppressed for v6)

    room_seq = build_room_sequence(start_room, turns_pf, goal_room)
    context_lines.append(f"Room sequence: {room_seq}")

    start_line = f"Start: {start_room}"
    if _is_distinctive_landmark(start_lm):
        start_line += f", near the {start_lm}"
    context_lines.append(start_line)

    dist_str = f"Distance: ~{total_dist:.1f}m"
    if elevation > 0.5:
        dist_str += f" (goes UP {elevation:.1f}m, stairs)"
    elif elevation < -0.5:
        dist_str += f" (goes DOWN {abs(elevation):.1f}m, stairs)"
    context_lines.append(dist_str)

    turn_candidates = []
    seen_rooms = {start_room}
    for i, prim in enumerate(turn_prims):
        pf_t = turns_pf[i] if i < len(turns_pf) else {}
        is_key, priority = _is_key_turn_v5(pf_t, prim, seen_rooms)
        if is_key:
            direction = "left" if prim["type"] == "left_turn" else "right"
            t_room = (pf_t.get("room") or "").lower()
            t_lm = pf_t.get("main_landmark", "")
            t_trans = pf_t.get("room_transition", "")
            turn_candidates.append({
                "priority": priority,
                "direction": direction,
                "room": t_room,
                "landmark": t_lm if _is_distinctive_landmark(t_lm) else "",
                "room_transition": t_trans if t_trans and "none" not in t_trans.lower() else "",
                "is_new_room": _has_new_room(pf_t, seen_rooms),
            })
            if t_room:
                seen_rooms.add(t_room)

    turn_candidates.sort(key=lambda x: -x["priority"])
    key_turns = turn_candidates[:2]

    if not key_turns:
        n_total = len(turn_prims)
        if n_total > 0:
            context_lines.append(f"Navigation: {n_total} turn(s), follow hallway/rooms")
        else:
            context_lines.append("Navigation: walk straight")
    else:
        wp_strs = []
        for t in key_turns:
            if t["room_transition"]:
                wp = t["room_transition"]
                if t["landmark"]:
                    wp += f" near the {t['landmark']}"
                if t["is_new_room"] and t["room"]:
                    wp += f" (into {t['room']})"
            elif t["landmark"]:
                wp = f"turn {t['direction']} at the {t['landmark']}"
                if t["room"]:
                    wp += f" (into {t['room']})"
            elif t["room"]:
                wp = f"enter the {t['room']}"
            else:
                wp = f"turn {t['direction']}"
            wp_strs.append(wp)
        context_lines.append("Key waypoints: " + " | ".join(wp_strs))

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
    if not any(w in text.lower() for w in ["stop", "wait", "halt", "stand", "pause"]):
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
    print("Gate3-Gemma v6 — GT-calibrated init_turn suppression")
    print("Suppress 20% of 90°+ init_turns → pct_zero≈57% (GT=56%)")
    print("=" * 70)

    with gzip.open(VAL_UNSEEN_PATH, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]
    print(f"  {len(episodes)} val_unseen episodes")

    # Load v5 checkpoint
    with open(V5_CHECKPOINT) as f:
        v5_ck = json.load(f)
    print(f"  v5 checkpoint: {len(v5_ck)} instructions loaded")

    # Load v6 checkpoint (for the 212 regenerated episodes)
    v6_ck = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            v6_ck = json.load(f)
        print(f"  v6 checkpoint: {len(v6_ck)} already done")

    gate3_dir = GATE3_PERFRAME_DIR

    # Build triplet position map
    triplet_pos = find_triplets(episodes)

    # Identify which episodes need regeneration
    regen_tasks = []
    reuse_count = 0

    for ep in episodes:
        eid = ep["episode_id"]
        pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        it = pa.get("initial_turn")

        should_suppress = (
            it is not None
            and it["angle_deg"] >= INIT_TURN_CONTEXT_THRESHOLD
            and (eid % 10) < 2
        )

        if should_suppress:
            if str(eid) in v6_ck:
                reuse_count += 1
                continue
            pf_path = gate3_dir / f"episode_{eid:06d}.json"
            lm_path = LANDMARK_DIR / f"episode_{eid:06d}.json"
            pf = json.loads(pf_path.read_text()) if pf_path.exists() else {}
            lm = json.loads(lm_path.read_text()) if lm_path.exists() else {}
            ctx = build_context_v6(eid, ep, pf, lm)
            if ctx:
                # Set triplet temperature
                pos = triplet_pos.get(eid, 0)
                ctx["temperature"] = TRIPLET_TEMPS[pos % len(TRIPLET_TEMPS)]
                regen_tasks.append(ctx)
        else:
            reuse_count += 1

    print(f"\n  Episodes to regenerate (suppressed init_turn): {len(regen_tasks)}")
    print(f"  Episodes reusing v5 instruction: {reuse_count}")
    print(f"  v6 checkpoint already done: {len(v6_ck)}")

    if regen_tasks:
        print(f"\n[Gemma] Regenerating {len(regen_tasks)} episodes (no init_turn context)...")
        t0 = time.time()
        new_results = await generate_batch_async(regen_tasks, concurrency=16, progress_every=50, prompt_type="gate3")
        v6_ck.update({str(k): v for k, v in new_results.items()})
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(v6_ck, f)
        elapsed = time.time() - t0
        print(f"  Done {len(new_results)} in {elapsed:.1f}s ({len(new_results)/elapsed:.1f}/s)")

    # Merge: v6_ck overrides v5_ck for suppressed episodes
    merged = {**v5_ck, **v6_ck}
    all_gen = {int(k): v for k, v in merged.items()}

    # Quality stats
    ok = sum(1 for v in all_gen.values() if quality_check(clean_output(v))[0])
    texts_all = [clean_output(v) for v in all_gen.values() if v]

    def count_ex(t): return len(re.findall(r'\bturn (?:left|right|around|slightly)\b', t, re.I))
    explicit = [count_ex(t) for t in texts_all]
    avg_explicit = sum(explicit) / len(texts_all)
    avg_words = sum(len(t.split()) for t in texts_all) / len(texts_all)
    pct_zero = sum(1 for t in explicit if t == 0) / len(texts_all) * 100
    pct_3plus = sum(1 for t in explicit if t >= 3) / len(texts_all) * 100

    print(f"\n[Stats] Gate3-Gemma v6 vs v5 vs GT:")
    print(f"  {'Metric':<25} {'v6':>8} {'v5':>8} {'GT':>8}")
    print(f"  {'avg_explicit_turns':<25} {avg_explicit:>8.2f} {'0.54':>8} {'0.66':>8}")
    print(f"  {'pct_zero_explicit':<25} {pct_zero:>8.1f}% {'45.6%':>8} {'56%':>8}")
    print(f"  {'pct_3plus_turns':<25} {pct_3plus:>8.1f}% {'0.0%':>8} {'3.8%':>8}")
    print(f"  {'avg_words':<25} {avg_words:>8.1f} {'24.9':>8} {'26.8':>8}")
    print(f"  quality_ok: {ok}/{len(all_gen)}")

    # Sample regenerated episodes
    print(f"\n[Samples] 5 suppressed episodes (v5 → v6 change):")
    suppressed_eids = [int(k) for k in v6_ck.keys()][:5]
    for eid in suppressed_eids:
        v5_text = clean_output(v5_ck.get(str(eid), ""))
        v6_text = clean_output(v6_ck.get(str(eid), ""))
        print(f"\n  EP{eid}:")
        print(f"    v5: {v5_text}")
        print(f"    v6: {v6_text}")

    # Assemble dataset
    print("\n[Assemble] Building v6 val_unseen dataset...")
    tok = VLNTokenizer(VAL_UNSEEN_PATH)
    assembled = []
    for ep in episodes:
        eid = ep["episode_id"]
        raw = all_gen.get(eid, "")
        text = clean_output(raw) if raw else ""
        if text:
            assembled.append(assemble_episode(ep, text, tok))

    dataset = {
        "episodes": assembled,
        "instruction_vocab": data.get("instruction_vocab", {}),
        "_generation_meta": {
            "mode": "gate3_gemma_v6",
            "model": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
            "split": "val_unseen",
            "n_episodes": len(assembled),
            "n_regenerated": len(v6_ck),
            "avg_explicit_turns": avg_explicit,
            "pct_zero_explicit": pct_zero,
            "avg_words": avg_words,
            "init_turn_threshold": INIT_TURN_CONTEXT_THRESHOLD,
            "suppress_rate": SUPPRESS_RATE,
            "note": "v5 + 20% suppression of init_turn for 90°+ episodes → pct_zero≈57% (GT=56%)",
        },
    }
    save_dataset(dataset, OUTPUT_PATH)

    NVME_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_gate3_gemma_v6.json.gz")
    NVME_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(NVME_PATH, "wt") as f:
        json.dump(dataset, f)
    print(f"\nSaved:    {OUTPUT_PATH}")
    print(f"Deployed: {NVME_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
