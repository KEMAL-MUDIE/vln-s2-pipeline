#!/usr/bin/env python3
"""
Generate v89: Room-transition-focused instructions at GT abstraction level.

KEY INSIGHT from v213 analysis:
  v213: SR=36.87% with avg_words=26.8 PERFECT and landmark avg 1.96 words.
  GT: SR=63.57% with avg_words=26.8 and landmark avg 1.73 words.
  Gap is NOT vocabulary statistics — gap is SEMANTIC STRUCTURE.

  GT instructions operate at ROOM-TRANSITION level:
    "Exit the bedroom and turn left. Walk through the kitchen. Stop near the couch."
  v213 operates at OBJECT-DETECTION level:
    "Pass the brown wooden double doors. Turn left at the white dining table. Stop by grey chaise lounge."

  The model was trained on GT instructions → responds to ROOM TRANSITIONS, not object lists.

v89 DESIGN:
  Use gate3_perframe ROOM NAMES (not object landmarks) as primary navigation anchors:
    - start_room from perframe (where agent starts)
    - transition rooms at each turn (where agent goes)
    - goal_room at stop
  Use path_analyzer for turn DIRECTIONS (always correct)
  Use initial turn from start_rotation
  Use simple stop landmark (prefer room name over specific object)

  Template:
    "[Initial turn and] [exit/leave/walk out of] [start_room]. [Turn dir] [into/toward/through] [next_room].
     [Continue to/walk into] [goal_room]. [Stop prep] [stop_lm/goal_room]."

  This matches GT's dominant pattern: room exits → turns → room entries → stop.

EXPECTED:
  - Hallway%: ~32% (perframe has hallway in ~32% of goal rooms)
  - Turn%: ~90% (initial turn + path turns)
  - avg_words: ~20-25 (shorter but GT-like)
  - SR: potentially higher than v213 due to room-transition focus
"""
import gzip
import json
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gate2_path.path_analyzer import (
    analyze_path, quaternion_yaw_deg, heading_between_xz, signed_angle_diff
)

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
GT_PATH = BASE / "val_unseen" / "val_unseen_patched.json.gz"
V71_PATH = BASE / "val_unseen" / "val_unseen_v71.json.gz"
PERFRAME_DIR = Path("/home/kemal/VLNav/s2_pipeline_new/outputs/gate3_perframe")
OUT_PATH = BASE / "val_unseen" / "val_unseen_v89.json.gz"

INITIAL_TURN_THRESHOLD = 30.0

INVALID_LM = {"none", "none visible", "n/a", "na", "unknown", "unclear",
              "not visible", "not applicable", "no landmark", ""}

COLOR_STRIP = re.compile(
    r'\b(white|grey|gray|brown|black|dark|light|beige|cream|tan|'
    r'rectangular|circular|oval|decorative|large|small|tall|short|'
    r'wooden|metal|glass|fabric|stone|marble|polished|ornate|modern)\b\s*',
    re.IGNORECASE
)
HALLWAY_LIKE = {"hallway", "corridor", "hall", "passage"}
CLOSED_ROOMS = {"bedroom", "bathroom", "office", "closet", "kitchen", "dining room",
                "living room", "garage", "study", "den", "library", "laundry room"}


def clean_lm(lm):
    if not lm or str(lm).lower().strip() in INVALID_LM:
        return None
    lm = COLOR_STRIP.sub('', str(lm)).strip()
    lm = re.sub(r'\s+', ' ', lm).strip().lower()
    return lm if len(lm) > 2 else None


def get_initial_turn(qxyzw, path, threshold=INITIAL_TURN_THRESHOLD):
    if not qxyzw or len(path) < 2:
        return None
    agent_yaw = quaternion_yaw_deg(qxyzw)
    first_heading = heading_between_xz(path[0], path[1])
    angle = signed_angle_diff(agent_yaw, first_heading)
    if abs(angle) < threshold:
        return None
    return {
        'dir': 'left' if angle < 0 else 'right',
        'angle': abs(angle),
        'is_around': abs(angle) >= 150,
        'is_slight': abs(angle) < 45,
    }


def open_verb(start_room, rng):
    r = rng.random()
    if start_room in CLOSED_ROOMS:
        if r < 0.35:
            return f"Exit the {start_room} and"
        elif r < 0.60:
            return f"Leave the {start_room} and"
        elif r < 0.80:
            return f"Walk out of the {start_room} and"
        else:
            return f"Walk through the {start_room} and"
    elif start_room in HALLWAY_LIKE:
        # Don't say "hallway" explicitly — just walk forward
        if r < 0.50:
            return "Walk forward and"
        elif r < 0.75:
            return "Continue straight and"
        else:
            return "Walk ahead and"
    else:
        if r < 0.40:
            return f"Walk through the {start_room} and"
        elif r < 0.65:
            return f"Walk forward through the {start_room} and"
        else:
            return f"Continue through the {start_room} and"


def turn_phrase(direction, next_room, rng):
    """Simple turn phrase with optional next-room context."""
    r = rng.random()
    room_part = f" into the {next_room}" if next_room and rng.random() < 0.45 else ""
    if r < 0.60:
        return f"turn {direction}{room_part}."
    elif r < 0.80:
        return f"make a {direction} turn{room_part}."
    else:
        return f"turn to the {direction}{room_part}."


def stop_phrase(stop_lm, goal_room, rng):
    """Simple stop phrase — prefer stop_lm, fallback to goal_room (avoid bare 'hallway')."""
    if stop_lm:
        landmark = stop_lm
    elif goal_room and goal_room not in HALLWAY_LIKE:
        landmark = goal_room
    else:
        # hallway or missing: use generic spatial anchors
        landmark = rng.choice(["the doorway", "the end of the hallway", "the corner"])
    r_verb = rng.random()
    verb = "Stop" if r_verb < 0.55 else "Wait"
    r_prep = rng.random()
    if r_prep < 0.25:
        return f"{verb} near the {landmark}."
    elif r_prep < 0.45:
        return f"{verb} in front of the {landmark}."
    elif r_prep < 0.65:
        return f"{verb} at the {landmark}."
    elif r_prep < 0.83:
        return f"{verb} by the {landmark}."
    else:
        return f"{verb} next to the {landmark}."


def get_stop_sentence(inst):
    sents = re.split(r'(?<=[.!?])\s+', inst.strip())
    return sents[-1].strip() if len(sents) > 1 else inst.strip()


def build_instruction(ep, perframe, v71_stop, rng):
    eid = ep["episode_id"]
    path = ep["reference_path"]
    qxyzw = ep.get("start_rotation")

    # Path analysis
    pa = analyze_path(path, qxyzw, turn_threshold_deg=75.0)
    turn_prims = [p for p in pa["primitives"] if p["type"] in ("left_turn", "right_turn")]
    init_turn = pa.get("initial_turn")
    total_dist = pa["summary"]["total_distance_m"]

    # Perframe data
    pf_start = perframe.get("start", {})
    pf_turns = perframe.get("turns", [])
    pf_goal = perframe.get("goal", {})

    start_room = (pf_start.get("room") or "hallway").lower().strip()
    goal_room = (pf_goal.get("room") or "room").lower().strip()
    raw_stop = (pf_goal.get("stop_landmark") or "").strip()
    stop_lm = clean_lm(raw_stop)

    # If same room start/goal on long path → generic goal
    if start_room == goal_room and total_dist > 4.0 and start_room in CLOSED_ROOMS:
        goal_room = "room"
        stop_lm = None

    parts = []

    # Initial turn prefix
    init_prefix = ""
    if init_turn:
        if init_turn["is_around"]:
            init_prefix = "Turn around and"
        elif init_turn["angle_deg"] < 45:
            init_prefix = f"Turn slightly {init_turn['direction']} and"
        else:
            init_prefix = f"Turn {init_turn['direction']} and"

    if len(turn_prims) == 0:
        # 0-turn: straight path
        if init_prefix:
            if total_dist < 6.0:
                parts.append(f"{init_prefix} walk to the {goal_room}.")
            else:
                if start_room in HALLWAY_LIKE:
                    parts.append(f"{init_prefix} walk forward.")
                else:
                    parts.append(f"{init_prefix} walk through the {start_room}.")
                parts.append(f"Continue toward the {goal_room}.")
        else:
            if total_dist < 5.0:
                parts.append(f"Walk forward to the {goal_room}.")
            elif total_dist < 10.0:
                if start_room in HALLWAY_LIKE:
                    parts.append(f"Walk forward.")
                else:
                    parts.append(f"Walk through the {start_room}.")
                parts.append(f"Continue straight to the {goal_room}.")
            else:
                if start_room in HALLWAY_LIKE:
                    parts.append(f"Walk forward.")
                else:
                    parts.append(f"Walk through the {start_room}.")
                parts.append(f"Continue walking straight ahead.")
        # Always end with stop phrase (removed duplicate from long-path case)
        parts.append(stop_phrase(stop_lm, goal_room, rng))

    else:
        # Turn-based path: use room transitions
        first_turn = turn_prims[0]
        first_dir = "left" if first_turn["type"] == "left_turn" else "right"

        # Get next room after first turn
        pf_t0 = pf_turns[0] if pf_turns else {}
        next_room_raw = (pf_t0.get("room_transition") or "").strip()
        next_room = None
        if next_room_raw and "entering" in next_room_raw.lower():
            m = re.search(r"entering\s+(.+?)(?:\s+area)?$", next_room_raw, re.I)
            if m:
                next_room = m.group(1).strip().lower()
        if not next_room:
            next_room = (pf_t0.get("room") or "").lower().strip() or None

        # Opening + first turn
        if init_prefix:
            opener = f"{init_prefix} {open_verb(start_room, rng).lower()}"
        else:
            opener = open_verb(start_room, rng)

        tp = turn_phrase(first_dir, next_room, rng)
        parts.append(f"{opener} {tp}")

        # Subsequent turns
        for i, turn in enumerate(turn_prims[1:], 1):
            dir_i = "left" if turn["type"] == "left_turn" else "right"
            pf_ti = pf_turns[i] if i < len(pf_turns) else {}
            nr_raw = (pf_ti.get("room_transition") or "").strip()
            nr = None
            if nr_raw and "entering" in nr_raw.lower():
                m2 = re.search(r"entering\s+(.+?)(?:\s+area)?$", nr_raw, re.I)
                if m2:
                    nr = m2.group(1).strip().lower()
            if not nr:
                nr = (pf_ti.get("room") or "").lower().strip() or None

            if rng.random() < 0.60:
                parts.append(f"Continue and turn {dir_i}{f' into the {nr}' if nr else ''}.")
            else:
                parts.append(f"Turn {dir_i}{f' into the {nr}' if nr else ''}.")

        # Final approach
        if goal_room and goal_room not in (next_room or ""):
            approach_r = rng.random()
            if approach_r < 0.40:
                parts.append(f"Walk into the {goal_room}.")
            elif approach_r < 0.70:
                parts.append(f"Continue toward the {goal_room}.")
            else:
                parts.append(f"Head into the {goal_room}.")

        parts.append(stop_phrase(stop_lm, goal_room, rng))

    # Append v71 stop phrase (scene-accurate stop)
    # Replace our stop phrase with v71 stop if v71_stop is different
    # Actually use our generated stop + v71 stop as suffix if they don't conflict
    instruction = " ".join(parts)
    return instruction


def main():
    print("=== v89: Room-transition-focused, GT abstraction level ===")
    print("  Uses gate3_perframe ROOM NAMES (not object names) as primary anchors")
    print("  Matches GT's dominant structure: exit_room → turn → enter_room → stop")
    print()

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V71_PATH) as f:
        v71_data = json.load(f)

    v71_by_eid = {ep["episode_id"]: ep for ep in v71_data["episodes"]}

    new_episodes = []
    stats = {'no_perframe': 0, 'has_perframe': 0, 'zero_turn': 0, 'one_turn': 0,
             'multi_turn': 0, 'has_init_turn': 0}

    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        pf_path = PERFRAME_DIR / f"episode_{eid:06d}.json"

        if pf_path.exists():
            perframe = json.loads(pf_path.read_text())
            stats['has_perframe'] += 1
        else:
            perframe = {}
            stats['no_perframe'] += 1

        v71_inst = v71_by_eid[eid]["instruction"]["instruction_text"]
        v71_stop = get_stop_sentence(v71_inst)

        rng = random.Random(eid ^ 0xFEED_BEEF)

        pa = analyze_path(ep["reference_path"], ep.get("start_rotation"), turn_threshold_deg=75.0)
        turn_prims = [p for p in pa["primitives"] if p["type"] in ("left_turn", "right_turn")]

        if len(turn_prims) == 0:
            stats['zero_turn'] += 1
        elif len(turn_prims) == 1:
            stats['one_turn'] += 1
        else:
            stats['multi_turn'] += 1
        if pa.get("initial_turn"):
            stats['has_init_turn'] += 1

        instruction = build_instruction(ep, perframe, v71_stop, rng)

        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        new_ep["instruction"]["instruction_text"] = instruction
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    print(f"Stats: {stats}")

    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]
    HALLWAY_RE = re.compile(r'\b(hallway|hall|corridor)\b', re.I)
    TURN_RE = re.compile(r'\bturn\b', re.I)

    print(f"\nQuality metrics (n={n}):")
    print(f"  avg_words: {sum(words)/n:.1f}  [GT=26.8]")
    print(f"  hallway:   {sum(1 for i in all_insts if HALLWAY_RE.search(i))/n*100:.1f}%  [GT=20.4%]")
    print(f"  through:   {sum(1 for i in all_insts if 'through' in i.lower())/n*100:.1f}%  [GT=27.2%]")
    print(f"  turn%:     {sum(1 for i in all_insts if TURN_RE.search(i))/n*100:.1f}%  [GT=84.7%]")
    print(f"  unique:    {len(set(all_insts))/n*100:.1f}%")

    print(f"\nSample instructions:")
    shown = 0
    for ep in gt_data["episodes"]:
        if shown >= 8:
            break
        eid = ep["episode_id"]
        inst = next(e["instruction"]["instruction_text"] for e in new_episodes if e["episode_id"] == eid)
        gt_inst = ep["instruction"]["instruction_text"]
        print(f"  GT:  {gt_inst[:150]}")
        print(f"  v89: {inst[:150]}")
        print()
        shown += 1

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
