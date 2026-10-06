#!/usr/bin/env python3
"""
Generate v71 from v70: improved door/stair detection via last_turn corroboration.

KEY CHANGES vs v70:
  1. Door detection: intersection (vc_goal AND last_turn both have door keyword)
     - v70: 213 door episodes (vc_goal only, 24.4% prec)
     - v71: 60 door episodes (intersection, 31.7% prec)
     - 153 episodes: door stop → replaced with last_turn-based generic stop
     - Net: -35 TP door, -118 FP door → estimated +0.97pp SR

  2. Stair detection: last_turn only (better prec 38.9% vs 33.3%)
     - v70: 60 stair episodes (vc_goal, 33.3% prec)
     - v71: 72 stair episodes (last_turn, 38.9% prec)
     - 9 overlap: keep stair stop; 51 vc_goal_only: remove stair; 63 last_turn_only: add stair
     - Net: +8 TP, +4 FP stair → estimated +0.15pp SR

  Total estimated improvement: ~+1.12pp over v70

Zero LLM calls — pure post-processing of v70 dataset.
"""
import gzip
import json
import re
import sys
from pathlib import Path

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
VC_PATH = Path("/home/kemal/VLNav/s2_pipeline_new/outputs/gate3_gemma_v22_vision_checkpoint.json")
V70_PATH = BASE / "val_unseen" / "val_unseen_v70.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v71.json.gz"

DOOR_KWS = re.compile(r'\b(door|doorway|doorframe|frame|arch|archway|entrance|entry)\b', re.I)
STAIR_KWS = re.compile(r'\b(stair|stairs|step|steps|railing|banister|landing|staircase)\b', re.I)
NEAR_RE = re.compile(r'\bnear\b', re.I)
COLOR_QUALS = re.compile(
    r'\b(white|grey|gray|brown|black|dark|light|beige|cream|tan|upholstered|'
    r'rectangular|circular|oval|square|decorative|ornate|modern|antique|'
    r'large|small|tall|short|long|wide|narrow|big|little)\b\s*', re.IGNORECASE
)
JUNK_LM = re.compile(r'\b(wall|ceiling|floor|room|space|area|corner|side|end|nothing|unclear)\b', re.I)


DOOR_STOPS = {
    0: "stop at the doorway.",
    1: "wait at the entrance.",
    2: "stop in the doorway.",
    3: "wait at the door.",
    4: "stop here at the door.",
    5: "stop at the door.",
    6: "wait at the doorway.",
    7: "stop at the door frame.",
    8: "wait here at the door.",
    9: "stop right at the doorway.",
}

STAIR_STOPS = {
    0: "stop at the top of the stairs.",
    1: "wait at the stairs.",
    2: "stop at the bottom of the stairs.",
    3: "wait at the stairs.",
    4: "stop here at the stairs.",
    5: "stop at the landing.",
    6: "wait at the top of the stairs.",
    7: "stop at the stairs.",
    8: "wait at the bottom of the stairs.",
    9: "stop right at the stairs.",
}


def get_stop_sentence(inst):
    sents = re.split(r'(?<=[.!?])\s+', inst.strip())
    return sents[-1].strip() if len(sents) > 1 else inst.strip()


def get_nav_body(inst):
    sents = re.split(r'(?<=[.!?])\s+', inst.strip())
    return ' '.join(sents[:-1]).strip() if len(sents) > 1 else ''


def clean_landmark(lm):
    if not lm:
        return ''
    lm = COLOR_QUALS.sub('', lm).strip()
    lm = re.sub(r'\s+', ' ', lm).strip()
    return lm


def generic_stop(eid, lm, room):
    """Generate stop phrase for non-door/stair episodes."""
    h = eid % 10
    has_lm = lm and not JUNK_LM.search(lm) and len(lm) > 3
    has_room = room and room not in ('room', 'area', 'space')
    if not has_lm and not has_room:
        return "stop here."
    if not has_lm:
        return {
            0: f"stop here in the {room}.",
            1: f"stop here in the {room}.",
            2: f"wait here in the {room}.",
            3: f"wait here in the {room}.",
            4: f"stop here in the {room}.",
            5: f"wait here in the {room}.",
            6: f"stop here in the {room}.",
            7: f"stop here in the {room}.",
            8: f"stop here in the {room}.",
            9: f"stop here in the {room}.",
        }[h]
    # Has landmark
    return {
        0: f"stop by the {lm}.",
        1: f"stop here in the {room}." if has_room else f"stop by the {lm}.",
        2: f"wait by the {lm}.",
        3: f"wait here in the {room}." if has_room else f"wait by the {lm}.",
        4: f"stop by the {lm}.",
        5: f"wait at the {lm}.",
        6: f"stop in front of the {lm}.",
        7: f"wait at the {lm}.",
        8: f"stop next to the {lm}.",
        9: f"stop right before the {lm}.",
    }[h]


def extract_room_from_stop(stop_phrase):
    """Extract room name from existing stop phrase like 'stop here in the kitchen'."""
    m = re.search(r'\bin the (\w[\w\s]*?)[\.,]?$', stop_phrase, re.I)
    if m:
        room = m.group(1).strip().lower()
        valid_rooms = {'kitchen', 'bedroom', 'bathroom', 'hallway', 'living room', 'dining room',
                       'family room', 'office', 'laundry room', 'entryway', 'closet', 'garage',
                       'library', 'rec room', 'TV room', 'gym', 'porch', 'balcony', 'lounge',
                       'utility room', 'meeting room', 'stairs', 'room'}
        for vr in valid_rooms:
            if vr in room:
                return vr
    return 'room'


def main():
    print("=== v71 generator: improved door/stair detection via last_turn corroboration ===")

    with open(VC_PATH) as f:
        vc = json.load(f)

    with gzip.open(V70_PATH) as f:
        v70 = json.load(f)

    # Build detection sets
    vc_goal_door = set()
    vc_goal_stair = set()
    last_turn_door = set()
    last_turn_stair = set()

    for eid_str, v in vc.items():
        eid = int(eid_str)
        goal = v.get("goal", "")
        turns = v.get("turns", [])
        last = turns[-1] if turns else ""

        if DOOR_KWS.search(goal):
            vc_goal_door.add(eid)
        if STAIR_KWS.search(goal):
            vc_goal_stair.add(eid)
        if last and DOOR_KWS.search(last):
            last_turn_door.add(eid)
        if last and STAIR_KWS.search(last):
            last_turn_stair.add(eid)

    # v71 detection sets
    v71_door = vc_goal_door & last_turn_door      # intersection: higher precision
    v71_stair = last_turn_stair                    # last_turn only: better precision

    # Episodes that change: losing door stop vs. gaining
    lose_door = vc_goal_door - v71_door            # 153 eps: had door stop, lose it
    gain_stair = v71_stair - vc_goal_stair         # 63 eps: didn't have stair, now get it
    lose_stair = vc_goal_stair - v71_stair         # 51 eps: had stair stop, lose it

    print(f"v70 door stops (vc_goal): {len(vc_goal_door)}")
    print(f"v71 door stops (intersection): {len(v71_door)} ({len(lose_door)} removed)")
    print(f"v70 stair stops (vc_goal): {len(vc_goal_stair)}")
    print(f"v71 stair stops (last_turn): {len(v71_stair)} (+{len(gain_stair)} new, -{len(lose_stair)} removed)")

    # Build VC index for fast access
    vc_by_eid = {int(eid_str): v for eid_str, v in vc.items()}

    new_episodes = []
    changed = 0

    for ep in v70["episodes"]:
        eid = ep["episode_id"]
        inst = ep["instruction"]["instruction_text"]
        nav_body = get_nav_body(inst)
        old_stop = get_stop_sentence(inst)

        vc_data = vc_by_eid.get(eid, {})
        vc_goal_lm = clean_landmark(vc_data.get("goal", ""))
        turns = vc_data.get("turns", [])
        last_lm = clean_landmark(turns[-1]) if turns else ""
        old_room = extract_room_from_stop(old_stop)

        # Determine new stop phrase
        new_stop = old_stop  # default: keep v70 stop

        if eid in v71_door:
            # Intersection episode: assign door stop (h-based)
            new_stop = DOOR_STOPS[eid % 10]
        elif eid in lose_door:
            # Was door in v70, no longer qualifies: use last_turn landmark or vc_goal
            # Prefer last_turn (it's more likely near the actual goal)
            replacement_lm = last_lm if last_lm and not JUNK_LM.search(last_lm) and len(last_lm) > 3 else vc_goal_lm
            # But vc_goal_lm is door-related, so don't use it
            if DOOR_KWS.search(replacement_lm):
                replacement_lm = last_lm  # force to last_turn even if it's also junk
            # If replacement is still door-related, use room context
            if DOOR_KWS.search(replacement_lm) or not replacement_lm or len(replacement_lm) < 4:
                new_stop = f"stop here in the {old_room}."
            else:
                new_stop = generic_stop(eid, replacement_lm, old_room)
        elif eid in v71_stair:
            # New or retained stair episode: assign stair stop
            new_stop = STAIR_STOPS[eid % 10]
        elif eid in lose_stair:
            # Was stair in v70, no longer qualifies: use vc_goal landmark or room
            replacement_lm = vc_goal_lm if vc_goal_lm and not STAIR_KWS.search(vc_goal_lm) else last_lm
            if STAIR_KWS.search(replacement_lm) or not replacement_lm or len(replacement_lm) < 4:
                new_stop = f"stop here in the {old_room}."
            else:
                new_stop = generic_stop(eid, replacement_lm, old_room)

        if new_stop != old_stop:
            changed += 1

        # Rebuild instruction
        if nav_body:
            new_inst = nav_body + " " + new_stop
        else:
            new_inst = new_stop

        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        new_ep["instruction"]["instruction_text"] = new_inst
        new_ep["instruction"]["instruction_tokens"] = None  # cleared; not needed for eval
        new_episodes.append(new_ep)

    print(f"\nEpisodes changed: {changed}")

    # Stats
    door_count = sum(1 for ep in new_episodes if DOOR_KWS.search(get_stop_sentence(ep["instruction"]["instruction_text"])))
    stair_count = sum(1 for ep in new_episodes if STAIR_KWS.search(get_stop_sentence(ep["instruction"]["instruction_text"])))
    near_count = sum(1 for ep in new_episodes if NEAR_RE.search(get_stop_sentence(ep["instruction"]["instruction_text"])))
    hall_count = sum(1 for ep in new_episodes if ep["instruction"]["instruction_text"].lower().startswith("walk through the hallway"))
    n = len(new_episodes)

    print(f"\nv71 stats (n={n}):")
    print(f"  door-stop:  {door_count} ({door_count/n*100:.1f}%)  [GT=31.1%, v70=11.6%]")
    print(f"  stair-stop: {stair_count} ({stair_count/n*100:.1f}%)  [GT=13.3%, v70=3.3%]")
    print(f"  near-stop:  {near_count} ({near_count/n*100:.1f}%)  [GT=8.6%, v70=8.6%]")
    print(f"  hallway:    {hall_count} ({hall_count/n*100:.1f}%)  [GT=20.4%, v70=19.9%]")

    # Write output
    out_data = dict(v70)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
