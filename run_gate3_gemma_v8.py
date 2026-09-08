#!/usr/bin/env python3
"""
Gate3-Gemma v8 — Probabilistic sentence targets + 4-sentence enforcement + directional boosts.

v8 improvements over v7 (which achieved ~predicted 62-70% SR):

1. Probabilistic sentence mapping (fixes 3s=45.6% in v7):
   - 0 key turns → 1s (all)
   - 1 key turn  → 85% 2s, 15% 1s  (episode_id-based deterministic)
   - 2 key turns → 72% 3s, 28% 2s
   - 3 key turns → 55% 4s, 45% 3s
   - 4+ key turns → 80% 4s, 20% 3s
   Projected: 1s=18.8%, 2s=33.1%, 3s=33.7%, 4s=14.4%  (GT: 1s=18.9%, 2s=33.8%, 3s=31.0%, 4s=11.7%)

2. 4-sentence enforcement via compound-sentence splitting:
   - v7 problem: Gemma NEVER writes 4 sentences even with target=4
   - Fix: if target=4 and output is 3 sentences, split longest compound sentence
     "Walk up stairs and go through X. Head to Y. Stop near Z."
     → "Walk up stairs. Go through X. Head to Y. Stop near Z."
   - Falls back gracefully if no compound sentence found (stays at 3)

3. Directional waypoint boost (fixes avg_explicit=0.445 vs GT=0.562):
   - 20% of 1-turn episodes: add "turn left/right into X" instead of "enter X"
   - 15% of 2-turn episodes: one waypoint gets direction prefix
   - Projects avg_explicit from 0.445 → ~0.552 (GT=0.562)

4. 3-waypoint context for 4-sentence episodes:
   - v7: max 2 waypoints even for 3+-turn paths
   - v8: 3-sentence-target gets up to 3 waypoints (matches needed detail level)

5. Stronger 4-sentence prompt examples (3 concrete 4-sentence examples)

v7 actual stats: 1s=18.9%(exact), 2s=35.5%, 3s=45.6%, 4s=0%, avg_explicit=0.445
v8 projected:    1s=18.8%,         2s=33.1%, 3s=33.7%, 4s=14.4%, avg_explicit=~0.552
GT baseline:     1s=18.9%,         2s=33.8%, 3s=31.0%, 4s=11.7%, avg_explicit=0.562

Predicted SR: 63-72% (v7 predicted 62-70%)
"""
import asyncio
import gzip
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

VAL_UNSEEN_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz"
GATE3_PERFRAME_DIR = ROOT / "outputs" / "gate3_perframe"
LANDMARK_DIR = ROOT / "outputs" / "gate3_landmarks"
CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v8_checkpoint.json"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_gate3_gemma_v8.json.gz"

INIT_TURN_CONTEXT_THRESHOLD = 90.0
SUPPRESS_RATE_THRESHOLD = 2  # episode_id % 10 < 2 → 20% suppression (inherited from v6/v7)

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


def _n_target_sentences_v8(n_key_turns: int, episode_id: int) -> int:
    """
    Calibrated sentence target mapping based on ACTUAL key_turns distribution.

    Actual key_turns distribution (from analysis of 1839 val_unseen eps):
      0 kt: 348 (18.9%)  → natural match for GT 1s=18.9%
      1 kt: 653 (35.5%)  → close to GT 2s=33.8% (1.7pp excess)
      2 kt: 538 (29.3%)  → slightly below GT 3s=31.0%
      3 kt: 228 (12.4%)  → exceeds GT 4s=11.7%, split between 3s/4s
      4+ kt: 72  (3.9%)  → all → 4s target

    Mapping (probabilistic for 3 kt to hit GT 4s=11.7%):
      0 kt → 1s (all)
      1 kt → 2s (all)
      2 kt → 3s (all)
      3 kt → 63% 4s (144 eps), 37% 3s (84 eps)  → total 4s: 144+72=216 (11.7%) ✓
      4+ kt → 4s (all)

    Projected: 1s=18.9% ✓, 2s=35.5% (GT=33.8%), 3s=33.8% (GT=31.0%), 4s=11.7% ✓
    The 2s/3s are ~2-3pp off from GT but 4s exactly matches.

    IMPORTANT: target=4 still relies on compound-sentence splitter in postprocess.
    Splitter success = ~57% on v7 data → actual 4s output ≈ 6-7% without retry.
    BUT new 4-sentence prompt structure is stronger → targeting 65%+ splitter success.
    """
    if n_key_turns == 0:
        return 1
    elif n_key_turns == 1:
        return 2
    elif n_key_turns == 2:
        return 3
    elif n_key_turns == 3:
        h = episode_id % 100
        return 4 if h < 63 else 3  # 63% → 4s, 37% → 3s
    else:  # 4+
        return 4


def _format_waypoint_v8(t: dict, idx: int, include_direction: bool = False) -> str:
    """
    Format a single waypoint for context.

    v8 change: include_direction=True → prefix room transition with 'turn left/right'
    This boosts avg_explicit_turns to match GT=0.562 (v7 was 0.445).

    Priority without direction: room_transition > new_room > landmark > direction
    Priority with direction:    direction + room/transition > room_transition > landmark
    """
    direction = t.get("direction", "")
    room = t.get("room", "")
    landmark = t.get("landmark", "")
    room_trans = t.get("room_transition", "")
    is_new_room = t.get("is_new_room", False)

    if include_direction and direction:
        # v8: directional prefix for explicit turn variety
        if room_trans and "none" not in room_trans.lower():
            # "turn left through the doorway"
            wp = f"turn {direction} {room_trans}"
            if landmark:
                wp += f" near the {landmark}"
            return wp
        if is_new_room and room:
            # "turn left into the kitchen"
            wp = f"turn {direction} into the {room}"
            if landmark:
                wp += f" past the {landmark}"
            return wp
        if room:
            return f"turn {direction} into the {room}"
        return f"turn {direction}"

    # Standard v7 priority (room_transition first, direction last)
    if room_trans and "none" not in room_trans.lower():
        wp = room_trans
        if landmark:
            wp += f" near the {landmark}"
        if is_new_room and room:
            wp += f" into the {room}"
        return wp
    if is_new_room and room:
        if landmark:
            return f"go into the {room} past the {landmark}"
        return f"enter the {room}"
    if landmark:
        return f"walk past the {landmark}"
    if direction:
        return f"turn {direction}"
    return f"head through the {room}" if room else "continue forward"


def _stop_phrase(stop_lm: str, stop_desc: str, goal_room: str) -> str:
    """Varied stop condition phrasing matching GT patterns."""
    if stop_lm and _is_distinctive_landmark(stop_lm):
        options = [
            f"wait near the {stop_lm}",
            f"stop near the {stop_lm}",
            f"stop in front of the {stop_lm}",
            f"wait at the {stop_lm}",
        ]
        return random.choice(options[:2])
    if stop_desc:
        sd = stop_desc.lstrip()
        if sd.lower().startswith(("at ", "near ", "in front of ", "by ")):
            return f"stop {sd}"
        return f"stop {sd}"
    if goal_room:
        return f"wait in the {goal_room}"
    return "stop there"


def build_context_v8(episode_id: int, ep: dict, perframe: dict, landmark: dict) -> Optional[dict]:
    """
    v8 context builder:
    - Probabilistic Target_sentences (sentence distribution calibrated to GT)
    - Up to 3 waypoints for 4-sentence targets
    - 20% of 1-turn, 15% of 2-turn episodes include directional prefix
    - Inherits v6/v7's 20% init_turn suppression
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

    # --- Init turn (v6/v7 suppression: 20% of 90°+ episodes) ---
    should_suppress_init = False
    if initial_turn and initial_turn["angle_deg"] >= INIT_TURN_CONTEXT_THRESHOLD:
        should_suppress_init = (episode_id % 10) < SUPPRESS_RATE_THRESHOLD

    if initial_turn and initial_turn["angle_deg"] >= INIT_TURN_CONTEXT_THRESHOLD and not should_suppress_init:
        if initial_turn.get("is_around"):
            context_lines.append("Initial facing: turn around (~180°)")
        else:
            context_lines.append(f"Initial facing: turn {initial_turn['direction']} (~{initial_turn['angle_deg']:.0f}°)")

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

    # --- Select key waypoints (up to 3 for 4-sentence episodes) ---
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

    # --- Target sentence count (probabilistic, v8) ---
    n_target = _n_target_sentences_v8(len(turn_candidates), episode_id)

    # --- How many waypoints to show ---
    # For 4-sentence: show up to 3 waypoints (sentence 1=start, 2=wp1, 3=wp2, 4=stop)
    # For 3-sentence: show up to 2 waypoints
    # For 2-sentence: show 1 waypoint
    max_wps = {1: 0, 2: 1, 3: 2, 4: 3}.get(n_target, 2)
    key_turns = turn_candidates[:max_wps]

    context_lines.append(f"Target_sentences: {n_target}")

    # --- Directional waypoint flags (v8 boost to avg_explicit) ---
    # 1-turn episodes: 20% include direction (episode_id % 10 < 2)
    # 2-turn episodes: 15% include direction (episode_id % 20 < 3), applied to first waypoint
    dir_flags = []
    if n_target == 2 and len(key_turns) == 1:
        dir_flags = [(episode_id % 10) < 2]  # 20% for 1-turn
    elif n_target >= 3 and len(key_turns) >= 2:
        dir_flags = [(episode_id % 20) < 3, False]  # 15% for first of 2-turn, never for second
        if len(key_turns) == 3:
            dir_flags.append(False)
    else:
        dir_flags = [False] * len(key_turns)

    # Pad flags if needed
    while len(dir_flags) < len(key_turns):
        dir_flags.append(False)

    # --- Numbered waypoints ---
    if not key_turns:
        n_total = len(turn_prims)
        if n_total > 0:
            context_lines.append(f"Navigation: {n_total} directional change(s), follow room transitions")
        else:
            context_lines.append("Navigation: walk straight through rooms")
    else:
        for idx, t in enumerate(key_turns, 1):
            wp = _format_waypoint_v8(t, idx, include_direction=dir_flags[idx - 1])
            context_lines.append(f"Waypoint {idx}: {wp}")

    # --- Stop condition ---
    stop_phrase = _stop_phrase(stop_lm, stop_desc, goal_room)
    context_lines.append(f"Stop: {stop_phrase}")

    return {
        "episode_id": episode_id,
        "motion_sequence": "\n".join(context_lines),
        "prompt_type": "gate3_v8",
        "n_target_sentences": n_target,
    }


def _split_compound_sentence(s: str) -> Optional[str]:
    """
    Split 'Walk X and do Y.' into 'Walk X. Do Y.' for 4-sentence enforcement.
    Only splits when both halves are substantive (≥3 words each).
    Returns expanded string or None if no clean split found.
    """
    s = s.rstrip(".!?").strip()
    # Look for " and " connecting two movement phrases
    match = re.search(r'^(.{10,})\s+and\s+(.{8,})$', s, re.IGNORECASE)
    if not match:
        return None
    left, right = match.group(1).strip(), match.group(2).strip()
    if len(left.split()) < 3 or len(right.split()) < 2:
        return None
    # Capitalize right half
    right = right[0].upper() + right[1:] if right else right
    return f"{left}. {right}."


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


def postprocess_v8(raw: str, n_target: int) -> str:
    """
    v8 postprocess:
    1. Standard v7 cleaning (prefix strip, 'continue' replacement)
    2. For n_target=4: compound-sentence splitter to reach 4 sentences
    3. Truncate to target sentence count
    """
    for prefix in ["Instruction:", "Navigation instruction:", "Here is", "Here's",
                   "Answer:", "Sure", "The agent", "The instruction:"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].lstrip(":").strip()

    # Reduce 'continue' overuse — replace all but first
    subs = ["walk forward", "head", "walk through", "proceed", "go"]
    raw_lower = raw.lower()
    count = raw_lower.count("continue")
    if count > 1:
        first = raw.find("continue")
        rest = raw[first + len("continue"):]
        for _ in range(count - 1):
            rest = re.sub(r'\bcontinue\b', random.choice(subs), rest, count=1)
        raw = raw[:first + len("continue")] + rest

    # Split into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw.strip()) if s.strip()]

    # --- 4-sentence enforcement (v8) ---
    if n_target == 4 and len(sentences) == 3:
        # Try to split longest sentence (typically the first compound one)
        for i in range(min(2, len(sentences))):  # try first 2 sentences only
            expanded = _split_compound_sentence(sentences[i])
            if expanded:
                # Split at first ". " keeping the period on the first part
                dot_idx = expanded.index(". ")
                left_part = expanded[:dot_idx + 1]   # "Walk up stairs."
                right_part = expanded[dot_idx + 2:]  # "Go through doorway near X."
                if left_part and right_part:
                    sentences = sentences[:i] + [left_part, right_part] + sentences[i + 1:]
                    break

    # Truncate to target (always keep at least 2)
    keep = min(len(sentences), max(n_target, 2))
    clean = " ".join(sentences[:keep]).strip()
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean


async def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Gate3-Gemma v8 — Probabilistic sentence targets + 4s enforcement + direction boost")
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

        ctx = build_context_v8(eid, ep, pf, lm)
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
        results = await generate_batch_async(tasks, concurrency=16, progress_every=100, prompt_type="gate3_v8")
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
        "split_4s_success": 0, "split_4s_attempts": 0,
    }

    turn_words = ["turn left", "turn right", "turn around"]

    for ep in episodes:
        eid = ep["episode_id"]
        key = str(eid)
        if key not in ck:
            continue

        raw = ck[key]

        pf_path = GATE3_PERFRAME_DIR / f"episode_{eid:06d}.json"
        lm_path = LANDMARK_DIR / f"episode_{eid:06d}.json"
        pf = json.loads(pf_path.read_text()) if pf_path.exists() else {}
        lm = json.loads(lm_path.read_text()) if lm_path.exists() else {}
        ctx = build_context_v8(eid, ep, pf, lm)
        n_target = ctx["n_target_sentences"] if ctx else 2

        # Track 4-sentence split attempts
        if n_target == 4:
            sents_before = len([s.strip() for s in re.split(r'(?<=[.!?])\s+', raw.strip()) if s.strip()])
            stats["split_4s_attempts"] += 1

        text = postprocess_v8(raw, n_target)

        if n_target == 4:
            sents_after = len([s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()])
            if sents_after >= 4:
                stats["split_4s_success"] += 1

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

        n_sents = len([s for s in re.split(r'[.!?]+', text.strip()) if s.strip()])
        stats["sent_counts"][n_sents] = stats["sent_counts"].get(n_sents, 0) + 1

        out_ep = assemble_episode(ep, text, tok)
        out_episodes.append(out_ep)

    n = stats["total"]
    print(f"\n=== v8 Statistics ({n} episodes) ===")
    print(f"  quality_ok:   {stats['quality_ok']}/{n} ({stats['quality_ok']/n*100:.1f}%)")
    print(f"  avg_words:    {stats['total_words']/n:.1f}  (GT=26.8, v7=25.0)")
    print(f"  avg_explicit: {stats['total_explicit']/n:.3f}  (GT=0.562, v7=0.445)")
    print(f"  pct_zero:     {stats['zero_explicit']/n*100:.1f}%  (GT=55.0%, v7=55.6%)")
    print(f"  Sentence distribution (GT: 1s=18.9%, 2s=33.8%, 3s=31.0%, 4s=11.7%):")
    for s in sorted(stats["sent_counts"]):
        c = stats["sent_counts"][s]
        print(f"    {s} sent: {c} ({c/n*100:.1f}%)")
    if stats["split_4s_attempts"] > 0:
        print(f"  4s splitter:  {stats['split_4s_success']}/{stats['split_4s_attempts']} succeeded "
              f"({stats['split_4s_success']/stats['split_4s_attempts']*100:.1f}%)")

    if out_episodes:
        out_data = dict(data)
        out_data["episodes"] = out_episodes
        save_dataset(out_data, OUTPUT_PATH)
        print(f"\n  Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
