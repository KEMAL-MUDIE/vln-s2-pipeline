#!/usr/bin/env python3
"""
Generate v83: vc_turn intermediate landmark enrichment for 0-turn episodes.

KEY INSIGHT: 86.2% of 0-turn episodes (900/1047) have vc_turn landmarks from rendered
Habitat images at intermediate path waypoints. These are SCENE-ACCURATE intermediate
landmarks the robot walks PAST on the way to the goal.

We've been IGNORING these for 0-turn episodes (only using vc_goal). v83 uses them:
  - 0 vc_turns: keep v80 length-calibrated template
  - 1 vc_turn:  "Walk forward past the [lm1]. Continue toward the [vc_goal]. [stop]"
  - 2 vc_turns: "Walk forward past the [lm1] and the [lm2]. Continue toward [vc_goal]. [stop]"
  - 3 vc_turns: "Walk forward past the [lm1] and the [lm2]. Continue past the [lm3] toward [vc_goal]. [stop]"

Turn episodes: UNCHANGED from v80 (Gemma body + vc_goal anchor — turn episodes are already good).

Expected improvement:
  - 0-turn eps: much richer, scene-accurate instructions → potentially SR+5-15%
  - Turn eps: same as v80
  - Through: slightly increases (vc_turns may have archways, doorways that are "through" contexts)

This is a pure Python post-processing of v80. No Gemma needed — fast, deterministic.

GT distribution reference: through=27.2%, hallway=20.4%, avg_words=26.8
v80:                        through=3.5%,  hallway=4.9%,  avg_words=24.6
v83 expected:               through=5-8%,  hallway=4.9%,  avg_words=26-28 (0-turn eps get longer)
"""
import gzip
import json
import math
import re
from pathlib import Path

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
VC_PATH = Path("/home/kemal/VLNav/s2_pipeline_new/outputs/gate3_gemma_v22_vision_checkpoint.json")
GT_PATH = BASE / "val_unseen" / "val_unseen.json.gz"
V80_PATH = BASE / "val_unseen" / "val_unseen_v80.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v83.json.gz"

TURN_THRESHOLD = 80

COLOR_STRIP = re.compile(
    r'\b(white|grey|gray|brown|black|dark|light|beige|cream|tan|upholstered|'
    r'rectangular|circular|oval|square|decorative|ornate|modern|antique|'
    r'large|small|tall|short|long|wide|narrow|big|little|wooden|metal|glass|'
    r'fabric|cloth|leather|stone|marble|polished)\b\s*',
    re.IGNORECASE
)
HALLWAY_RE = re.compile(r'\b(hallway|hall)\b', re.IGNORECASE)


def clean_lm(lm):
    if not lm:
        return ""
    lm = COLOR_STRIP.sub('', lm).strip()
    lm = re.sub(r'\s+', ' ', lm).strip()
    return lm.lower() if len(lm) > 3 else ""


def is_same_lm(a, b):
    """Check if two landmarks are essentially the same (avoid redundant mention)."""
    a_clean = clean_lm(a)
    b_clean = clean_lm(b)
    if not a_clean or not b_clean:
        return False
    # Consider same if one is contained in the other
    return a_clean in b_clean or b_clean in a_clean


def count_turns(path, threshold=TURN_THRESHOLD):
    for i in range(1, len(path)-1):
        A, B, C = path[i-1], path[i], path[i+1]
        ax, az = B[0]-A[0], B[2]-A[2]
        bx, bz = C[0]-B[0], C[2]-B[2]
        la = math.sqrt(ax*ax + az*az)
        lb = math.sqrt(bx*bx + bz*bz)
        if la < 0.05 or lb < 0.05:
            continue
        ax, az = ax/la, az/la
        bx, bz = bx/lb, bz/lb
        cross_z = ax*bz - az*bx
        dot = ax*bx + az*bz
        angle = math.degrees(math.atan2(cross_z, dot))
        if abs(angle) > threshold:
            return True  # has at least one turn
    return False


def path_length(path):
    total = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dz = path[i][2] - path[i-1][2]
        total += math.sqrt(dx*dx + dz*dz)
    return total


def build_0turn_vc_instruction(vc_turns_raw, vc_goal, v71_stop, plen):
    """Build 0-turn instruction enriched with scene-accurate vc_turn landmarks."""
    goal_lm = clean_lm(vc_goal)
    if not goal_lm:
        goal_lm = "the destination"

    # Clean turn landmarks, filter duplicates of goal and empty ones
    turn_lms = []
    for t in vc_turns_raw:
        lm = clean_lm(t)
        if not lm:
            continue
        if is_same_lm(lm, goal_lm):
            continue  # skip if same as goal (redundant)
        # Skip hallway references
        if HALLWAY_RE.search(lm):
            continue
        # Deduplicate similar turns (e.g., "white door" + "white interior door")
        if any(is_same_lm(lm, existing) for existing in turn_lms):
            continue
        turn_lms.append(lm)

    n = len(turn_lms)

    if n == 0:
        # No valid turn landmarks — use length-calibrated template (same as v80)
        if plen < 6.0:
            return f"Walk forward toward the {goal_lm}. {v71_stop}", 'template'
        elif plen < 10.0:
            return f"Walk straight ahead toward the {goal_lm}. Continue forward until you reach it. {v71_stop}", 'template'
        elif plen < 15.0:
            return (f"Walk straight ahead and continue forward past the area, making your way "
                    f"toward the {goal_lm}. Keep going straight until you get there. {v71_stop}"), 'template'
        else:
            return (f"Walk straight ahead and continue forward through the area. Keep going, "
                    f"heading toward the {goal_lm} at the far end. {v71_stop}"), 'template'
    elif n == 1:
        return f"Walk forward past the {turn_lms[0]}. Continue toward the {goal_lm}. {v71_stop}", 'vc_lm'
    elif n == 2:
        return (f"Walk forward past the {turn_lms[0]} and the {turn_lms[1]}. "
                f"Continue toward the {goal_lm}. {v71_stop}"), 'vc_lm'
    else:  # n >= 3 (cap at 3)
        return (f"Walk forward past the {turn_lms[0]} and the {turn_lms[1]}. "
                f"Continue past the {turn_lms[2]} toward the {goal_lm}. {v71_stop}"), 'vc_lm'


def main():
    print("=== v83 generator: vc_turn intermediate landmark enrichment for 0-turn episodes ===")
    print("  86.2% of 0-turn eps have scene-accurate vc_turn intermediate landmarks")
    print("  Turn eps: unchanged from v80 (Gemma body + vc_goal anchor)")
    print()

    with open(VC_PATH) as f:
        vc = json.load(f)
    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V80_PATH) as f:
        v80_data = json.load(f)

    v80_by_eid = {ep["episode_id"]: ep for ep in v80_data["episodes"]}
    vc_by_eid = {int(k): v for k, v in vc.items()}

    new_episodes = []
    stats = {
        'zero_turn_template': 0,
        'zero_turn_vc_lm': 0,
        'zero_turn_no_valid_lm': 0,
        'turn_eps_kept': 0,
    }

    # Need v71 stop phrases — extract from v80 (they're the same)
    # v80 turn instructions end with v71 stop
    # For 0-turn, we need the v71 stop too — v80 templates already use v71 stop

    # Load v71 for stop phrases
    V71_PATH = BASE / "val_unseen" / "val_unseen_v71.json.gz"
    with gzip.open(V71_PATH) as f:
        v71_data = json.load(f)
    v71_by_eid = {ep["episode_id"]: ep for ep in v71_data["episodes"]}

    def get_stop_sentence(inst):
        sents = re.split(r'(?<=[.!?])\s+', inst.strip())
        return sents[-1].strip() if len(sents) > 1 else inst.strip()

    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        path = ep["reference_path"]
        plen = path_length(path)
        is_turn = count_turns(path)

        v80_ep = v80_by_eid[eid]
        v80_inst = v80_ep["instruction"]["instruction_text"]

        if is_turn:
            # Keep v80 instruction for turn episodes
            new_inst = v80_inst
            stats['turn_eps_kept'] += 1
        else:
            # 0-turn episode: enrich with vc_turn landmarks
            vdata = vc_by_eid.get(eid, {})
            vc_turns_raw = vdata.get("turns", [])
            vc_goal = vdata.get("goal", "")
            v71_stop = get_stop_sentence(v71_by_eid[eid]["instruction"]["instruction_text"])

            new_inst, mode = build_0turn_vc_instruction(vc_turns_raw, vc_goal, v71_stop, plen)
            if mode == 'vc_lm':
                stats['zero_turn_vc_lm'] += 1
            else:
                stats['zero_turn_template'] += 1

        new_ep = dict(v80_ep)
        new_ep["instruction"] = dict(v80_ep["instruction"])
        new_ep["instruction"]["instruction_text"] = new_inst
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    print(f"Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]

    def get_stop_sent(inst):
        sents = re.split(r'(?<=[.!?])\s+', inst.strip())
        return sents[-1].strip() if len(sents) > 1 else inst.strip()

    avg_turns_kw = sum(len(re.findall(r'\bturn\b', i, re.I)) for i in all_insts) / n
    hall = sum(1 for i in all_insts if HALLWAY_RE.search(i))
    thru = sum(1 for i in all_insts if 'through the' in i.lower())
    toward = sum(1 for i in all_insts if 'toward' in i.lower())
    past_kw = sum(1 for i in all_insts if re.search(r'\bpast\b', i, re.I))
    near = sum(1 for i in all_insts if re.search(r'\bnear\b', get_stop_sent(i), re.I))

    print(f"\nQuality metrics (n={n}):")
    print(f"  avg turns: {avg_turns_kw:.3f}  [GT=0.587, v80=0.569]")
    print(f"  avg words: {sum(words)/n:.1f}  [GT=26.8, v80=24.6]")
    print(f"  hallway:   {hall/n*100:.1f}%  [GT=20.4%, v80=4.9%]")
    print(f"  through:   {thru/n*100:.1f}%  [GT=27.2%, v80=3.5%]")
    print(f"  toward:    {toward/n*100:.1f}%  [v80=99.5%]")
    print(f"  past:      {past_kw/n*100:.1f}%  [v80 had minimal 'past']")
    print(f"  near-stop: {near/n*100:.1f}%  [GT=8.6%]")
    print(f"  unique:    {len(set(all_insts))/n*100:.1f}%")

    # Sample changed 0-turn instructions
    print(f"\nSample 0-turn instructions enriched with vc_turns:")
    shown = 0
    for ep in gt_data["episodes"]:
        if shown >= 8:
            break
        eid = ep["episode_id"]
        path = ep["reference_path"]
        if count_turns(path):
            continue
        v80_inst = v80_by_eid[eid]["instruction"]["instruction_text"]
        new_inst = next(e["instruction"]["instruction_text"] for e in new_episodes if e["episode_id"] == eid)
        if new_inst != v80_inst:
            print(f"  v80: {v80_inst[:120]}")
            print(f"  v83: {new_inst[:120]}")
            print()
            shown += 1

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
