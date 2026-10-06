#!/usr/bin/env python3
"""
Generate v85: pure template for ALL episodes (no Gemma), using vc_turns everywhere.

MOTIVATION:
  v80: Gemma for turn episodes → hallucination risk, variable quality
  v83: vc_turns for 0-turn → excellent scene-accurate instructions
  v85: extend v83's vc_turn approach to TURN episodes too

DESIGN:
  0-turn (same as v83): "Walk forward past [lm1] and [lm2]. Continue toward [vc_goal]. [stop]"
  1-turn: "Walk forward and turn [dir] near the [vc_lm]. Continue toward the [vc_goal]. [stop]"
  2-turn: "Walk forward and turn [dir1] near [lm1]. Continue and turn [dir2] near [lm2]. Continue toward [vc_goal]. [stop]"
  3-turn: "Walk forward and turn [dir1] near [lm1]. Continue and turn [dir2] near [lm2]. Turn [dir3] near [lm3]. Continue toward [vc_goal]. [stop]"

  With through injection for long preturn (>7.44m):
  "Walk through the area and turn [dir] near [vc_lm]. Continue toward [vc_goal]. [stop]"

ADVANTAGES:
  - No Gemma hallucination
  - Deterministic, reproducible
  - Uses ALL available vc_turn information
  - Common GT phrase patterns ("walk forward and turn", "continue toward")
  - Scene-accurate landmarks from Habitat renders

EXPECTED:
  through: ~7% (geometry-calibrated)
  past: ~52% (0-turn vc_turns)
  avg_words: ~22-25
  toward: ~99.5%
  unique: ~95%+ (diverse vc_turns and vc_goals)
"""
import gzip
import json
import math
import re
from pathlib import Path

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
VC_PATH = Path("/home/kemal/VLNav/s2_pipeline_new/outputs/gate3_gemma_v22_vision_checkpoint.json")
GT_PATH = BASE / "val_unseen" / "val_unseen.json.gz"
V71_PATH = BASE / "val_unseen" / "val_unseen_v71.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v85.json.gz"

TURN_THRESHOLD = 80
INJECT_THRESHOLD = 7.44  # geometry calibrated for ~10% through

COLOR_STRIP = re.compile(
    r'\b(white|grey|gray|brown|black|dark|light|beige|cream|tan|upholstered|'
    r'rectangular|circular|oval|square|decorative|ornate|modern|antique|'
    r'large|small|tall|short|long|wide|narrow|big|little|wooden|metal|glass|'
    r'fabric|cloth|leather|stone|marble|polished)\b\s*',
    re.IGNORECASE
)
HALLWAY_RE = re.compile(r'\b(hallway|hall)\b', re.IGNORECASE)
STOP_VERBS = re.compile(r'\b(stop|wait|halt|pause|stand)\b', re.IGNORECASE)


def clean_lm(lm):
    if not lm:
        return ""
    lm = COLOR_STRIP.sub('', lm).strip()
    lm = re.sub(r'\s+', ' ', lm).strip()
    return lm.lower() if len(lm) > 3 else ""


def is_same_lm(a, b):
    a_c, b_c = clean_lm(a), clean_lm(b)
    if not a_c or not b_c:
        return False
    return a_c in b_c or b_c in a_c


def get_stop_sentence(inst):
    sents = re.split(r'(?<=[.!?])\s+', inst.strip())
    return sents[-1].strip() if len(sents) > 1 else inst.strip()


def path_length(path):
    total = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dz = path[i][2] - path[i-1][2]
        total += math.sqrt(dx*dx + dz*dz)
    return total


def extract_turns_and_dists(path, threshold=TURN_THRESHOLD):
    cumul = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dz = path[i][2] - path[i-1][2]
        cumul.append(cumul[-1] + math.sqrt(dx*dx + dz*dz))
    total = cumul[-1]
    turns = []
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
            turns.append({
                'dir': 'right' if angle > 0 else 'left',
                'dist': cumul[i],
                'frac': cumul[i] / total if total > 0 else 0,
            })
    return turns, total


def assign_vc(turns, vc_turns_raw):
    if not vc_turns_raw or not turns:
        return ['' for _ in turns]
    n_vc = len(vc_turns_raw)
    vc_fracs = [(i+1)/(n_vc+1) for i in range(n_vc)]
    used = set()
    results = []
    for t in turns:
        dists = sorted((abs(vc_fracs[i] - t['frac']), i) for i in range(n_vc))
        assigned = ''
        for _, i in dists:
            if i not in used:
                used.add(i)
                assigned = vc_turns_raw[i]
                break
        results.append(assigned)
    return results


def build_0turn_instruction(vc_turns_raw, vc_goal, v71_stop, plen):
    """Same as v83: vc_turn-enriched 0-turn instruction."""
    goal_lm = clean_lm(vc_goal)
    if not goal_lm:
        goal_lm = "the destination"

    turn_lms = []
    for t in vc_turns_raw:
        lm = clean_lm(t)
        if not lm:
            continue
        if HALLWAY_RE.search(lm):
            continue
        if is_same_lm(lm, goal_lm):
            continue
        if any(is_same_lm(lm, existing) for existing in turn_lms):
            continue
        turn_lms.append(lm)

    n = len(turn_lms)
    if n == 0:
        if plen < 6.0:
            return f"Walk forward toward the {goal_lm}. {v71_stop}"
        elif plen < 10.0:
            return f"Walk straight ahead toward the {goal_lm}. Continue forward until you reach it. {v71_stop}"
        elif plen < 15.0:
            return (f"Walk straight ahead and continue forward past the area, making your way "
                    f"toward the {goal_lm}. Keep going straight until you get there. {v71_stop}")
        else:
            return (f"Walk straight ahead and continue forward through the area. Keep going, "
                    f"heading toward the {goal_lm} at the far end. {v71_stop}")
    elif n == 1:
        return f"Walk forward past the {turn_lms[0]}. Continue toward the {goal_lm}. {v71_stop}"
    elif n == 2:
        return (f"Walk forward past the {turn_lms[0]} and the {turn_lms[1]}. "
                f"Continue toward the {goal_lm}. {v71_stop}")
    else:
        return (f"Walk forward past the {turn_lms[0]} and the {turn_lms[1]}. "
                f"Continue past the {turn_lms[2]} toward the {goal_lm}. {v71_stop}")


def build_turn_instruction(turns, vc_lms_raw, vc_goal, v71_stop, preturn_dist):
    """Pure template for turn episodes — no Gemma."""
    goal_lm = clean_lm(vc_goal) or "the destination"

    # Clean and validate turn landmarks
    cleaned_lms = [clean_lm(lm) for lm in vc_lms_raw]

    # First turn: optionally "through the area" if preturn is long
    use_through = preturn_dist >= INJECT_THRESHOLD

    parts = []
    for i, (t, lm) in enumerate(zip(turns, cleaned_lms)):
        if i == 0:
            # First segment
            lm_part = f" near the {lm}" if lm else ""
            if use_through:
                parts.append(f"Walk through the area and turn {t['dir']}{lm_part}.")
            else:
                parts.append(f"Walk forward and turn {t['dir']}{lm_part}.")
        else:
            # Subsequent turns
            lm_part = f" near the {lm}" if lm else ""
            parts.append(f"Continue and turn {t['dir']}{lm_part}.")

    # Final segment toward goal
    parts.append(f"Continue toward the {goal_lm}.")
    parts.append(v71_stop)

    return " ".join(parts)


def main():
    print("=== v85 generator: pure template for all episodes (no Gemma) ===")
    print("  0-turn: same as v83 (vc_turn landmarks, scene-accurate)")
    print("  turn: pure template 'Walk forward and turn [dir] near [vc_lm]'")
    print("  Through injection for turn eps with pre-turn >= 7.44m")
    print()

    with open(VC_PATH) as f:
        vc = json.load(f)
    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V71_PATH) as f:
        v71_data = json.load(f)

    v71_by_eid = {ep["episode_id"]: ep for ep in v71_data["episodes"]}
    vc_by_eid = {int(k): v for k, v in vc.items()}

    new_episodes = []
    stats = {
        'zero_turn': 0,
        'turn_1': 0,
        'turn_2': 0,
        'turn_3plus': 0,
        'through_injected': 0,
    }

    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        path = ep["reference_path"]
        plen = path_length(path)

        vdata = vc_by_eid.get(eid, {})
        vc_turns_raw = vdata.get("turns", [])
        vc_goal = vdata.get("goal", "")
        v71_stop = get_stop_sentence(v71_by_eid[eid]["instruction"]["instruction_text"])

        turns, total = extract_turns_and_dists(path)

        if not turns:
            instruction = build_0turn_instruction(vc_turns_raw, vc_goal, v71_stop, plen)
            stats['zero_turn'] += 1
        else:
            vc_lms = assign_vc(turns, vc_turns_raw)
            preturn_dist = turns[0]['dist']
            if preturn_dist >= INJECT_THRESHOLD:
                stats['through_injected'] += 1
            if len(turns) == 1:
                stats['turn_1'] += 1
            elif len(turns) == 2:
                stats['turn_2'] += 1
            else:
                stats['turn_3plus'] += 1
            instruction = build_turn_instruction(turns, vc_lms, vc_goal, v71_stop, preturn_dist)

        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        new_ep["instruction"]["instruction_text"] = instruction
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
    print(f"  avg turns: {avg_turns_kw:.3f}  [GT=0.587]")
    print(f"  avg words: {sum(words)/n:.1f}  [GT=26.8, v83=23.0]")
    print(f"  hallway:   {hall/n*100:.1f}%  [GT=20.4%]")
    print(f"  through:   {thru/n*100:.1f}%  [GT=27.2%, v84=7.4%]")
    print(f"  toward:    {toward/n*100:.1f}%  [v83=99.5%]")
    print(f"  past:      {past_kw/n*100:.1f}%  [v83=52.7%]")
    print(f"  near-stop: {near/n*100:.1f}%  [GT=8.6%]")
    print(f"  unique:    {len(set(all_insts))/n*100:.1f}%")

    # Sample turn instructions
    print(f"\nSample turn instructions (pure template):")
    shown = 0
    for ep in gt_data["episodes"]:
        if shown >= 6:
            break
        path = ep["reference_path"]
        turns, _ = extract_turns_and_dists(path)
        if not turns:
            continue
        inst = next(e["instruction"]["instruction_text"] for e in new_episodes if e["episode_id"] == ep["episode_id"])
        print(f"  [{len(turns)} turn] {inst[:150]}")
        shown += 1

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
