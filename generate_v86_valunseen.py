#!/usr/bin/env python3
"""
Generate v86: most comprehensive vc_turn utilization.

MOTIVATION:
  v83/v84/v85 use vc_turns for 0-turn (as "walk past" landmarks) — excellent
  v85 uses vc_turns for turn episodes only at TURN POSITIONS (one vc_lm per turn)

  UNUSED: vc_turns at INTER-TURN positions (between detected turns) — ignored in v85!

  v86: use ALL vc_turns, assigning each to the correct path segment:
    - Before first turn: "Walk past [lm]" prefix
    - At each turn position: "turn [dir] near [lm]"
    - Between turns: "Continue past [lm]"
    - After last turn: "Continue toward [vc_goal]"

DESIGN:
  All vc_turns are spatially sorted and assigned to path segments by position fraction.
  The segment assignment:
    - vc_turn[i] frac < first_turn_frac → PRE segment (walk past)
    - vc_turn[i] frac ≈ turn[j] frac → AT turn j (near landmark)
    - Between turn[j] and turn[j+1] → BETWEEN segment (continue past)
    - After last turn → POST segment (absorbed into "Continue toward [vc_goal]")

ADVANTAGE:
  - Uses maximum information from Habitat renders
  - Scene-accurate landmarks at every stage of navigation
  - Richer instructions (more words from "past [lm1] and [lm2]" phrases)
  - Expected avg_words: ~22-26 (up from v85=20.9)

SAME:
  - Geometry turns (94.2% accurate)
  - v71 stop phrases (calibrated)
  - vc_goal anchor
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
OUT_PATH = BASE / "val_unseen" / "val_unseen_v86.json.gz"

TURN_THRESHOLD = 80
INJECT_THRESHOLD = 7.44  # geometry calibrated

COLOR_STRIP = re.compile(
    r'\b(white|grey|gray|brown|black|dark|light|beige|cream|tan|upholstered|'
    r'rectangular|circular|oval|square|decorative|ornate|modern|antique|'
    r'large|small|tall|short|long|wide|narrow|big|little|wooden|metal|glass|'
    r'fabric|cloth|leather|stone|marble|polished)\b\s*',
    re.IGNORECASE
)
HALLWAY_RE = re.compile(r'\b(hallway|hall)\b', re.IGNORECASE)
DOORWAY_RE = re.compile(r'\b(doorway|arched|arch|doorframe|door frame)\b', re.IGNORECASE)
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


def assign_vc_to_segments(turns, vc_turns_raw, goal_lm):
    """
    Assign vc_turns to path segments:
      PRE: before first turn
      AT[j]: at turn j (the near landmark)
      BETWEEN[j,j+1]: between turns j and j+1
      POST: after last turn (→ subsumed into vc_goal)

    Returns:
      pre_lms: list of lms before first turn
      at_lms: list of one lm per turn (the near landmark)
      between_lms: list of lists of lms between consecutive turns
    """
    if not vc_turns_raw or not turns:
        return [], ['' for _ in turns], [[] for _ in range(len(turns)-1)]

    n_vc = len(vc_turns_raw)
    # VC turns are at evenly spaced positions along the path
    vc_fracs = [(i+1)/(n_vc+1) for i in range(n_vc)]

    turn_fracs = [t['frac'] for t in turns]

    # Assign each vc_turn to a segment
    pre_lms = []
    at_lms_by_turn = [[] for _ in turns]
    between_lms = [[] for _ in range(len(turns)-1)]
    post_lms = []

    for vc_i, vc_frac in enumerate(vc_fracs):
        raw_lm = vc_turns_raw[vc_i]
        lm = clean_lm(raw_lm)
        if not lm or HALLWAY_RE.search(lm):
            continue
        if is_same_lm(lm, goal_lm):
            continue

        # Find which segment this vc_turn belongs to
        if vc_frac < turn_fracs[0] - 0.05:
            # Before first turn
            pre_lms.append(lm)
        elif vc_frac > turn_fracs[-1] + 0.05:
            # After last turn → post (absorbed into vc_goal)
            post_lms.append(lm)
        else:
            # Find closest turn
            min_dist = float('inf')
            closest_turn = 0
            for j, tf in enumerate(turn_fracs):
                d = abs(vc_frac - tf)
                if d < min_dist:
                    min_dist = d
                    closest_turn = j

            if min_dist < 0.12:  # vc_turn is AT this turn
                at_lms_by_turn[closest_turn].append((min_dist, lm))
            else:
                # vc_turn is BETWEEN turns — find which gap
                for j in range(len(turns)-1):
                    if turn_fracs[j] < vc_frac < turn_fracs[j+1]:
                        between_lms[j].append(lm)
                        break

    # For each turn, pick the closest vc_turn as the "near" landmark
    at_lms = []
    for j, candidates in enumerate(at_lms_by_turn):
        if candidates:
            candidates.sort()  # sorted by distance
            at_lms.append(candidates[0][1])  # pick closest
        else:
            at_lms.append('')

    # Deduplicate pre and between lms
    def dedup_lms(lm_list):
        seen = []
        for lm in lm_list:
            if not any(is_same_lm(lm, s) for s in seen):
                seen.append(lm)
        return seen[:3]  # cap at 3

    pre_lms = dedup_lms(pre_lms)
    between_lms = [dedup_lms(b) for b in between_lms]

    return pre_lms, at_lms, between_lms


def spatial_verb(lm):
    """Use 'through' for doorways/arches, 'past' for others."""
    if DOORWAY_RE.search(lm):
        return "through the"
    return "past the"


def build_0turn_instruction(vc_turns_raw, vc_goal, v71_stop, plen):
    """Same as v83/v84/v85: vc_turn-enriched 0-turn instruction."""
    goal_lm = clean_lm(vc_goal)
    if not goal_lm:
        goal_lm = "the destination"

    turn_lms = []
    for t in vc_turns_raw:
        lm = clean_lm(t)
        if not lm or HALLWAY_RE.search(lm):
            continue
        if is_same_lm(lm, goal_lm):
            continue
        if any(is_same_lm(lm, ex) for ex in turn_lms):
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
        v = spatial_verb(turn_lms[0])
        return f"Walk {v} {turn_lms[0]}. Continue toward the {goal_lm}. {v71_stop}"
    elif n == 2:
        v0, v1 = spatial_verb(turn_lms[0]), spatial_verb(turn_lms[1])
        if v0 == v1:
            return (f"Walk {v0} {turn_lms[0]} and the {turn_lms[1]}. "
                    f"Continue toward the {goal_lm}. {v71_stop}")
        else:
            return (f"Walk {v0} {turn_lms[0]}. Continue {v1} {turn_lms[1]} "
                    f"toward the {goal_lm}. {v71_stop}")
    else:
        v0, v2 = spatial_verb(turn_lms[0]), spatial_verb(turn_lms[2])
        return (f"Walk {v0} {turn_lms[0]} and the {turn_lms[1]}. "
                f"Continue {v2} {turn_lms[2]} toward the {goal_lm}. {v71_stop}")


def build_turn_instruction(turns, vc_turns_raw, vc_goal, v71_stop, total_len):
    """Comprehensive turn instruction using ALL vc_turns across all segments."""
    goal_lm = clean_lm(vc_goal) or "the destination"
    pre_lms, at_lms, between_lms = assign_vc_to_segments(turns, vc_turns_raw, goal_lm)

    parts = []
    preturn_dist = turns[0]['dist']
    use_through = preturn_dist >= INJECT_THRESHOLD

    # PRE segment: walk past landmarks before first turn
    if pre_lms:
        verbs = [spatial_verb(lm) for lm in pre_lms]
        if len(pre_lms) == 1:
            parts.append(f"Walk {verbs[0]} {pre_lms[0]}")
        elif len(pre_lms) == 2:
            parts.append(f"Walk {verbs[0]} {pre_lms[0]} and the {pre_lms[1]}")
        else:
            parts.append(f"Walk {verbs[0]} {pre_lms[0]} and the {pre_lms[1]}")
    elif use_through:
        parts.append("Walk through the area")
    else:
        parts.append("Walk forward")

    # First turn
    lm0 = at_lms[0] if at_lms else ''
    lm0_part = f" near the {lm0}" if lm0 else ""
    if parts:
        parts[-1] += f" and turn {turns[0]['dir']}{lm0_part}."
    else:
        parts.append(f"Walk forward and turn {turns[0]['dir']}{lm0_part}.")

    # Subsequent turns with between segments
    for j in range(1, len(turns)):
        between = between_lms[j-1] if j-1 < len(between_lms) else []
        lm_j = at_lms[j] if j < len(at_lms) else ''
        lm_j_part = f" near the {lm_j}" if lm_j else ""

        if between:
            verbs = [spatial_verb(lm) for lm in between]
            if len(between) == 1:
                parts.append(f"Continue {verbs[0]} {between[0]} and turn {turns[j]['dir']}{lm_j_part}.")
            else:
                parts.append(f"Continue {verbs[0]} {between[0]} and the {between[1]} and turn {turns[j]['dir']}{lm_j_part}.")
        else:
            parts.append(f"Continue and turn {turns[j]['dir']}{lm_j_part}.")

    # Final: toward goal
    parts.append(f"Continue toward the {goal_lm}.")
    parts.append(v71_stop)

    return " ".join(parts)


def main():
    print("=== v86 generator: comprehensive vc_turn utilization (all segments) ===")
    print("  0-turn: same as v83 (walk past/through scene-accurate landmarks)")
    print("  Turn: uses ALL vc_turns (pre, at-turn, between, goal) — richest instructions")
    print("  through/past choice based on vc_lm: doorways→'through', furniture→'past'")
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
    stats = {'zero_turn': 0, 'turn_1': 0, 'turn_2': 0, 'turn_3plus': 0,
             'pre_lm_used': 0, 'between_lm_used': 0}

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
            goal_lm = clean_lm(vc_goal)
            pre_lms, at_lms, between_lms = assign_vc_to_segments(turns, vc_turns_raw, goal_lm)
            if pre_lms:
                stats['pre_lm_used'] += 1
            if any(between_lms):
                stats['between_lm_used'] += 1
            if len(turns) == 1:
                stats['turn_1'] += 1
            elif len(turns) == 2:
                stats['turn_2'] += 1
            else:
                stats['turn_3plus'] += 1
            instruction = build_turn_instruction(turns, vc_turns_raw, vc_goal, v71_stop, total)

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
    print(f"  avg words: {sum(words)/n:.1f}  [GT=26.8, v85=20.9]")
    print(f"  hallway:   {hall/n*100:.1f}%  [GT=20.4%]")
    print(f"  through:   {thru/n*100:.1f}%  [GT=27.2%, v85=6.9%]")
    print(f"  toward:    {toward/n*100:.1f}%  [v85=100%]")
    print(f"  past:      {past_kw/n*100:.1f}%  [v85=49.8%]")
    print(f"  near-stop: {near/n*100:.1f}%  [GT=8.6%]")
    print(f"  unique:    {len(set(all_insts))/n*100:.1f}%")

    print(f"\nSample turn instructions:")
    shown = 0
    for ep in gt_data["episodes"]:
        if shown >= 8:
            break
        path = ep["reference_path"]
        turns, _ = extract_turns_and_dists(path)
        if not turns:
            continue
        inst = next(e["instruction"]["instruction_text"] for e in new_episodes if e["episode_id"] == ep["episode_id"])
        print(f"  [{len(turns)} turn] {inst[:160]}")
        shown += 1

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
