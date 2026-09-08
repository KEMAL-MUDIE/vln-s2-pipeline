#!/usr/bin/env python3
"""
Gate3-Gemma v7 — GT-distribution-aligned sentence count + 2-turn coverage.

Key improvements over v6 (all 1839 episodes regenerated):

1. Dynamic Target_sentences (injected into context):
   - 0 gate3 turns → 1 sentence
   - 1 gate3 turn  → 2 sentences
   - 2 gate3 turns → 3 sentences
   - 3 gate3 turns → 4 sentences
   - 4+ gate3 turns → 4 sentences
   GT: 1-sent=18.9%, 2-sent=33.8%, 3-sent=31.0%, 4-sent=11.7%
   v6: 2-sent=92.7% (massive mismatch, corrected here)

2. 2-turn path support — numbered Waypoint 1/Waypoint 2 format:
   - Previously: "Key waypoints: X | Y" (model ignored second turn)
   - Now: "Waypoint 1: turn LEFT at X" + "Waypoint 2: enter the Y"
   - Compels model to write 3-sentence instructions covering both turns
   GT: 10.3% 2-explicit-turn episodes; v6: 0% (fixed here)

3. Vocabulary normalization in prompt:
   - Do NOT use 'continue' more than once (v6: 3x GT frequency)
   - Use 'exit' when leaving start room (v6: 75% below GT)
   - Use 'walk down/up the stairs' for elevation changes
   - Allow 'make a right/left' as variation

4. Stop condition enriched:
   - New: "Stop: wait near the X" / "Stop: stop in front of the X"
   - Varied patterns instead of always "stop at"

5. Keep v6 init_turn suppression (20% of 90°+, episode_id % 10 < 2)
   → pct_zero stays calibrated at ~56% (matches GT=56%)

Predicted SR: 60-68% (fixes structural mismatch that limits v6)
"""
import asyncio
import gzip
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

VAL_UNSEEN_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz"
GATE3_PERFRAME_DIR = ROOT / "outputs" / "gate3_perframe"
LANDMARK_DIR = ROOT / "outputs" / "gate3_landmarks"
CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v7_checkpoint.json"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_gate3_gemma_v7.json.gz"

INIT_TURN_CONTEXT_THRESHOLD = 90.0
SUPPRESS_RATE_THRESHOLD = 2  # episode_id % 10 < 2  → 20% suppression

TRIPLET_TEMPS = [0.3, 0.5, 0.7]

from gate2_path.path_analyzer import analyze_path
from gate4_instructions.gemma_vllm_backend import generate_batch_async
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset

from run_gate3_gemma_v5 import (
    _is_distinctive_landmark,
    _is_key_turn_v5,
    _has_new_room,
    build_room_sequence,
    find_triplets,
)


def _n_target_sentences(n_gate3_turns: int) -> int:
    """
    Map gate3 turn count → target sentence count.
    Calibrated to match GT distribution:
      GT: 1s=18.9%, 2s=33.8%, 3s=31.0%, 4s+11.7%
      gate3: 0t=14.7%, 1t=27.4%, 2t=34.9%, 3t=16%, 4+t=7%
    """
    if n_gate3_turns == 0:
        return 1
    elif n_gate3_turns == 1:
        return 2
    elif n_gate3_turns == 2:
        return 3
    elif n_gate3_turns == 3:
        return 4
    else:
        return 4


def _format_waypoint(t: dict, idx: int) -> str:
    """
    Format a single waypoint for the context.
    Priority: room transition > enter new room > landmark > direction.
    Explicit 'turn left/right' used ONLY as last resort (GT-aligned pct_zero).
    """
    direction = t.get("direction", "")
    room = t.get("room", "")
    landmark = t.get("landmark", "")
    room_trans = t.get("room_transition", "")
    is_new_room = t.get("is_new_room", False)

    # Priority 1: Room transition (GT uses these for majority of waypoints)
    if room_trans and "none" not in room_trans.lower():
        wp = room_trans
        if landmark:
            wp += f" near the {landmark}"
        if is_new_room and room:
            wp += f" into the {room}"
        return wp
    # Priority 2: Enter a new room (room transition implied)
    if is_new_room and room:
        if landmark:
            return f"go into the {room} past the {landmark}"
        return f"enter the {room}"
    # Priority 3: Landmark without explicit direction (GT style)
    if landmark:
        return f"walk past the {landmark}"
    # Priority 4: Explicit direction (only when no room/landmark info)
    if direction:
        return f"turn {direction}"
    return f"head through the {room}" if room else "continue forward"


def _stop_phrase(stop_lm: str, stop_desc: str, goal_room: str) -> str:
    """Choose varied stop condition phrase matching GT patterns."""
    # GT top patterns: "wait at", "stop in the doorway", "wait there", "stop at", "wait near"
    if stop_lm and _is_distinctive_landmark(stop_lm):
        # Vary between GT-common stop patterns
        options = [
            f"wait near the {stop_lm}",
            f"stop near the {stop_lm}",
            f"stop in front of the {stop_lm}",
            f"wait at the {stop_lm}",
        ]
        return random.choice(options[:2])  # keep consistent (first 2 are most common in GT)
    if stop_desc:
        # strip leading "at/near" to avoid double
        sd = stop_desc.lstrip()
        if sd.lower().startswith(("at ", "near ", "in front of ", "by ")):
            return f"stop {sd}"
        return f"stop {sd}"
    if goal_room:
        return f"wait in the {goal_room}"
    return "stop there"


def build_context_v7(episode_id: int, ep: dict, perframe: dict, landmark: dict) -> Optional[dict]:
    """
    v7 context builder with dynamic Target_sentences and numbered waypoints.
    Keeps v6's 20% init_turn suppression for pct_zero calibration.
    """
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

    # --- Init turn (v6 suppression: 20% of 90°+ episodes) ---
    should_suppress_init = False
    if initial_turn and initial_turn["angle_deg"] >= INIT_TURN_CONTEXT_THRESHOLD:
        should_suppress_init = (episode_id % 10) < SUPPRESS_RATE_THRESHOLD

    if initial_turn and initial_turn["angle_deg"] >= INIT_TURN_CONTEXT_THRESHOLD and not should_suppress_init:
        if initial_turn.get("is_around"):
            context_lines.append("Initial facing: AROUND (~180°)")
        else:
            context_lines.append(f"Initial facing: sharp {initial_turn['direction']} (~{initial_turn['angle_deg']:.0f}°)")

    # --- Room sequence ---
    room_seq = build_room_sequence(start_room, turns_pf, goal_room)
    context_lines.append(f"Room sequence: {room_seq}")

    # --- Start ---
    start_line = f"Start: {start_room}"
    if _is_distinctive_landmark(start_lm):
        start_line += f", near the {start_lm}"
    context_lines.append(start_line)

    # --- Distance / stairs ---
    dist_str = f"Distance: ~{total_dist:.1f}m"
    if elevation > 0.5:
        dist_str += f" (walk UP {elevation:.1f}m stairs)"
    elif elevation < -0.5:
        dist_str += f" (walk DOWN {abs(elevation):.1f}m stairs)"
    context_lines.append(dist_str)

    # --- Select key waypoints ---
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

    # --- Target sentence count based on key_turns ---
    n_target = _n_target_sentences(len(key_turns))
    context_lines.append(f"Target_sentences: {n_target}")

    # --- Numbered waypoints ---
    if not key_turns:
        n_total = len(turn_prims)
        if n_total > 0:
            context_lines.append(f"Navigation: {n_total} directional change(s), follow room transitions")
        else:
            context_lines.append("Navigation: walk straight through rooms")
    else:
        for idx, t in enumerate(key_turns, 1):
            wp = _format_waypoint(t, idx)
            context_lines.append(f"Waypoint {idx}: {wp}")

    # --- Stop condition (varied phrasing) ---
    stop_phrase = _stop_phrase(stop_lm, stop_desc, goal_room)
    context_lines.append(f"Stop: {stop_phrase}")

    return {
        "episode_id": episode_id,
        "motion_sequence": "\n".join(context_lines),
        "prompt_type": "gate3_v7",
        "n_target_sentences": n_target,
    }


def quality_check(text: str) -> tuple:
    words = text.split()
    failures = []
    if len(words) < 6:
        failures.append(f"short({len(words)}w)")
    if len(words) > 80:
        failures.append(f"long({len(words)}w)")
    if not any(w in text.lower() for w in ["stop", "wait", "halt", "stand", "pause"]):
        failures.append("no_stop")
    return len(failures) == 0, failures


def postprocess(raw: str, n_target: int) -> str:
    """Clean output and truncate to target sentence count."""
    for prefix in ["Instruction:", "Navigation instruction:", "Here is", "Here's",
                   "Answer:", "Sure", "The agent", "The instruction:"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].lstrip(":").strip()

    # Replace overused 'continue' — pick a substitute from GT-common alternatives
    subs = ["walk forward", "head", "walk through", "proceed", "go"]
    raw_lower = raw.lower()
    count = raw_lower.count("continue")
    if count > 1:
        # Replace all but the first occurrence
        first = raw.find("continue")
        rest = raw[first + len("continue"):]
        for _ in range(count - 1):
            rest = re.sub(r'\bcontinue\b', random.choice(subs), rest, count=1)
        raw = raw[:first + len("continue")] + rest

    # Split to sentences and truncate to target + 1 (allow stop sentence)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw.strip()) if s.strip()]
    keep = min(len(sentences), max(n_target, 2))  # always keep at least 2 sentences
    clean = " ".join(sentences[:keep]).strip()
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean


async def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Gate3-Gemma v7 — Dynamic sentence count + 2-turn coverage")
    print("Full regeneration of all 1839 val_unseen episodes")
    print("=" * 70)

    with gzip.open(VAL_UNSEEN_PATH, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]
    print(f"  {len(episodes)} val_unseen episodes")

    # Load checkpoint
    ck = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            ck = json.load(f)
        print(f"  Checkpoint: {len(ck)} already done")

    triplet_pos = find_triplets(episodes)

    # Build all task contexts
    tasks = []
    for ep in episodes:
        eid = ep["episode_id"]
        if str(eid) in ck:
            continue

        pf_path = GATE3_PERFRAME_DIR / f"episode_{eid:06d}.json"
        lm_path = LANDMARK_DIR / f"episode_{eid:06d}.json"
        pf = json.loads(pf_path.read_text()) if pf_path.exists() else {}
        lm = json.loads(lm_path.read_text()) if lm_path.exists() else {}

        ctx = build_context_v7(eid, ep, pf, lm)
        if ctx:
            pos = triplet_pos.get(eid, 0)
            ctx["temperature"] = TRIPLET_TEMPS[pos % len(TRIPLET_TEMPS)]
            tasks.append(ctx)

    print(f"  Tasks to generate: {len(tasks)}")
    if not tasks:
        print("  All done from checkpoint.")

    if tasks:
        t0 = time.time()
        print(f"\n[Gemma] Generating {len(tasks)} instructions (concurrency=16)...")
        results = await generate_batch_async(tasks, concurrency=16, progress_every=100, prompt_type="gate3_v7")
        ck.update({str(k): v for k, v in results.items()})
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(ck, f)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s ({len(tasks)/elapsed:.1f} eps/s)")

    # Assemble dataset
    print("\n[Assemble] Building output dataset...")
    tok = VLNTokenizer()
    out_episodes = []
    stats = {
        "total": 0, "quality_ok": 0, "quality_fail": 0,
        "total_words": 0, "total_explicit": 0, "zero_explicit": 0,
        "sent_counts": {},
    }

    turn_words = ["turn left", "turn right", "turn around"]

    for ep in episodes:
        eid = ep["episode_id"]
        key = str(eid)
        if key not in ck:
            continue

        raw = ck[key]

        # Get n_target from context to apply postprocessing correctly
        pf_path = GATE3_PERFRAME_DIR / f"episode_{eid:06d}.json"
        lm_path = LANDMARK_DIR / f"episode_{eid:06d}.json"
        pf = json.loads(pf_path.read_text()) if pf_path.exists() else {}
        lm = json.loads(lm_path.read_text()) if lm_path.exists() else {}
        ctx = build_context_v7(eid, ep, pf, lm)
        n_target = ctx["n_target_sentences"] if ctx else 2

        text = postprocess(raw, n_target)
        ok, failures = quality_check(text)

        stats["total"] += 1
        if ok:
            stats["quality_ok"] += 1
        else:
            stats["quality_fail"] += 1

        words = text.split()
        stats["total_words"] += len(words)
        n_explicit = sum(1 for t in turn_words if t in text.lower())
        stats["total_explicit"] += n_explicit
        if n_explicit == 0:
            stats["zero_explicit"] += 1

        # Sentence count
        n_sents = len([s for s in re.split(r'[.!?]+', text.strip()) if s.strip()])
        stats["sent_counts"][n_sents] = stats["sent_counts"].get(n_sents, 0) + 1

        out_ep = assemble_episode(ep, text, tok)
        out_episodes.append(out_ep)

    n = stats["total"]
    print(f"\n=== v7 Statistics ({n} episodes) ===")
    print(f"  quality_ok:   {stats['quality_ok']}/{n} ({stats['quality_ok']/n*100:.1f}%)")
    print(f"  avg_words:    {stats['total_words']/n:.1f}")
    print(f"  avg_explicit: {stats['total_explicit']/n:.3f}  (GT=0.562)")
    print(f"  pct_zero:     {stats['zero_explicit']/n*100:.1f}%  (GT=55.0%)")
    print(f"  Sentence distribution:")
    for s in sorted(stats["sent_counts"]):
        c = stats["sent_counts"][s]
        print(f"    {s} sent: {c} ({c/n*100:.1f}%)  [GT target: 1s=18.9%, 2s=33.8%, 3s=31.0%, 4s=11.7%]")

    if out_episodes:
        out_data = dict(data)
        out_data["episodes"] = out_episodes
        save_dataset(out_data, OUTPUT_PATH)
        print(f"\n  Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
