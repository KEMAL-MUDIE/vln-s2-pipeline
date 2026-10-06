#!/usr/bin/env python3
"""
Generate v72: semantics-first instruction generation.

User direction: "focus well on the sequences of words in the instructions,
semantic meaning the similar, statistics then has to be high"

Approach:
  1. Use reference_path 3D geometry for CORRECT turn directions (86.1% match vs GT)
  2. Use VC visual descriptions assigned to path positions for spatial landmarks
  3. Generate natural instructions where semantics drive statistics, not vice versa
  4. Stop phrase reused from v71 (already optimized)

Key improvement over v61hpp:
  - v61hpp: Gemma-4 generates nav body (may hallucinate turn directions)
  - v72: path geometry gives correct turn directions for each episode
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
OUT_PATH = BASE / "val_unseen" / "val_unseen_v72.json.gz"

TURN_THRESHOLD = 80  # degrees — calibrated to match GT avg turns (0.579 vs GT 0.587)

# Junk landmarks to ignore
JUNK_LM = re.compile(r'^(wall|ceiling|floor|room|space|area|corner|side|end|nothing|unclear|a |an |the )$', re.I)

# Color qualifiers to strip from landmarks
COLOR_STRIP = re.compile(
    r'\b(white|grey|gray|brown|black|dark|light|beige|cream|tan|upholstered|'
    r'rectangular|circular|oval|square|decorative|ornate|modern|antique|'
    r'large|small|tall|short|long|wide|narrow|big|little|wooden|metal|glass|'
    r'fabric|cloth|leather|stone|marble|tile|wooden|polished|glazed)\b\s*',
    re.IGNORECASE
)


def cumulative_path(path):
    dists = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dz = path[i][2] - path[i-1][2]
        dists.append(dists[-1] + math.sqrt(dx*dx + dz*dz))
    return dists


def extract_significant_turns(path, threshold=TURN_THRESHOLD):
    """Return list of {idx, dir, angle, frac} for significant turns."""
    cumul = cumulative_path(path)
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
        # CORRECTED: positive cross_z = RIGHT turn in Habitat
        cross_z = ax*bz - az*bx
        dot = ax*bx + az*bz
        angle = math.degrees(math.atan2(cross_z, dot))
        if abs(angle) > threshold:
            turns.append({
                'idx': i,
                'dir': 'right' if angle > 0 else 'left',
                'angle': abs(angle),
                'frac': cumul[i]/total if total > 0 else 0,
            })
    return turns


def assign_vc_to_turns(turns, vc_turn_descs):
    """Map VC visual descriptions to path turns by fractional position.
    Returns dict: turn_idx -> vc_description (or "" if none).
    """
    if not vc_turn_descs or not turns:
        return {t['idx']: "" for t in turns}

    n_vc = len(vc_turn_descs)
    # VC descriptions are sampled at roughly equal intervals along the path
    vc_fracs = [(i + 1) / (n_vc + 1) for i in range(n_vc)]

    # Track which VC descs are used (avoid repeating same desc for multiple turns)
    used = set()
    assignments = {}
    for t in turns:
        dists = [(abs(vc_fracs[i] - t['frac']), i) for i in range(n_vc)]
        dists.sort()
        for _, best_i in dists:
            if best_i not in used:
                used.add(best_i)
                assignments[t['idx']] = vc_turn_descs[best_i]
                break
        else:
            # All used — don't repeat, leave empty
            assignments[t['idx']] = ""

    return assignments


def clean_landmark(lm):
    if not lm:
        return ""
    lm = COLOR_STRIP.sub('', lm).strip()
    lm = re.sub(r'\s+', ' ', lm).strip()
    if JUNK_LM.match(lm) or len(lm) < 4:
        return ""
    return lm.lower()


def get_stop_sentence(inst):
    sents = re.split(r'(?<=[.!?])\s+', inst.strip())
    return sents[-1].strip() if len(sents) > 1 else inst.strip()


def get_nav_body(inst):
    sents = re.split(r'(?<=[.!?])\s+', inst.strip())
    return ' '.join(sents[:-1]).strip() if len(sents) > 1 else ''


# ─── Phrase templates (indexed by eid % N for variety) ──────────────────────

OPENING_STRAIGHT = [
    "Walk forward",
    "Go straight",
    "Walk straight",
    "Walk forward",
    "Go forward",
    "Walk ahead",
    "Head forward",
    "Walk straight",
]

OPENING_WITH_LM = [
    "Walk forward past the",
    "Go straight past the",
    "Walk past the",
    "Continue past the",
    "Walk forward toward the",
    "Walk past the",
    "Head past the",
    "Walk forward past the",
]

WALK_CONTINUE = [
    "Walk forward",
    "Continue forward",
    "Walk straight",
    "Go forward",
    "Continue walking",
    "Walk forward",
    "Continue straight",
    "Go straight",
]

WALK_WITH_LM = [
    "Walk forward past the",
    "Continue past the",
    "Walk straight past the",
    "Go past the",
    "Walk past the",
    "Continue forward past the",
    "Walk straight past the",
    "Go forward past the",
]

TURN_PREPS = [
    "past",
    "near",
    "at",
    "through",
    "by",
    "past",
    "near",
    "at",
]

TURN_PREPS_EXACT = [  # when landmark is a clear visual marker
    "at the",
    "past the",
    "near the",
    "through the",
    "by the",
    "at the",
    "past the",
    "near the",
]


def make_opening(eid, turns, vc_turns, assignments):
    """Generate the opening phrase."""
    h = eid % 8
    # If first turn is very early (first 20% of path) and sharp, lead with it
    if turns and turns[0]['frac'] < 0.20 and turns[0]['angle'] > 50:
        t = turns[0]
        lm = clean_landmark(assignments.get(t['idx'], ""))
        if lm:
            prep = TURN_PREPS[h]
            return f"Turn {t['dir']} {prep} the {lm}", 1
        else:
            return f"Turn {t['dir']}", 1
    # Start with straight movement
    # Use first VC description as an opening landmark if no early turn
    first_vc = clean_landmark(vc_turns[0]) if vc_turns and not turns else ""
    if first_vc and len(turns) == 0:
        return f"{OPENING_WITH_LM[h]} {first_vc}", 0
    else:
        return OPENING_STRAIGHT[h], 0


def make_turn_phrase(eid, direction, lm, turn_idx):
    h = (eid + turn_idx * 3) % 8
    if lm:
        prep = TURN_PREPS[h]
        return f"Turn {direction} {prep} the {lm}"
    return f"Turn {direction}"


def make_walk_phrase(eid, lm, walk_idx):
    h = (eid + walk_idx * 5) % 8
    if lm:
        return f"{WALK_WITH_LM[h]} {lm}"
    return WALK_CONTINUE[h]


def generate_nav_body(eid, path, vc_turns, vc_goal):
    """Generate nav body from path geometry + VC descriptions."""
    turns = extract_significant_turns(path)
    assignments = assign_vc_to_turns(turns, vc_turns)

    segments = []
    opening, turn_start = make_opening(eid, turns, vc_turns, assignments)
    segments.append(opening)

    prev_frac = 0.0
    walk_idx = 0

    for i, t in enumerate(turns[turn_start:], start=turn_start):
        lm = clean_landmark(assignments.get(t['idx'], ""))
        turn_phrase = make_turn_phrase(eid, t['dir'], lm, i)

        # Add intermediate walk phrase if gap from previous turn is large
        if i > turn_start or (i == turn_start and turns[turn_start]['frac'] > 0.30):
            gap = t['frac'] - prev_frac
            if gap > 0.30 and i > 0:
                # Find an unused VC desc for this segment
                mid_vc = ""
                # Look for VC descriptions not yet assigned
                for vi, vd in enumerate(vc_turns):
                    vc_frac = (vi + 1) / (len(vc_turns) + 1) if vc_turns else 0.5
                    if prev_frac < vc_frac < t['frac']:
                        mid_lm = clean_landmark(vd)
                        if mid_lm:
                            mid_vc = mid_lm
                            break
                walk_phrase = make_walk_phrase(eid, mid_vc, walk_idx)
                segments.append(walk_phrase)
                walk_idx += 1

        segments.append(turn_phrase)
        prev_frac = t['frac']

    # If no turns at all, add one more walk phrase toward goal
    if not turns:
        # Use VC goal as directional target
        goal_lm = clean_landmark(vc_goal)
        if goal_lm:
            h = eid % 8
            segments.append(f"{WALK_WITH_LM[h]} {goal_lm}")

    return ". ".join(segments)


def main():
    print("=== v72 generator: semantics-first from path geometry + VC ===")
    print("  User direction: 'focus on sequences of words, semantic meaning the similar'")
    print("  Approach: path geometry → correct turns; VC → spatial landmarks; v71 stop phrases")
    print()

    with open(VC_PATH) as f:
        vc = json.load(f)
    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V71_PATH) as f:
        v71_data = json.load(f)

    gt_by_eid = {ep["episode_id"]: ep for ep in gt_data["episodes"]}
    v71_by_eid = {ep["episode_id"]: ep for ep in v71_data["episodes"]}
    vc_by_eid = {int(k): v for k, v in vc.items()}

    new_episodes = []
    stats = {
        'total': 0, 'no_turns': 0, 'one_turn': 0, 'multi_turn': 0,
        'has_vc': 0, 'no_vc': 0,
    }

    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        path = ep["reference_path"]

        vdata = vc_by_eid.get(eid, {})
        vc_turns = vdata.get("turns", [])
        vc_goal = vdata.get("goal", "")

        # Get stop phrase from v71 (already optimized for door/stair/landmark)
        v71_ep = v71_by_eid.get(eid)
        if v71_ep:
            stop_phrase = get_stop_sentence(v71_ep["instruction"]["instruction_text"])
        else:
            stop_phrase = "stop here."

        # Generate nav body from geometry + VC
        nav_body = generate_nav_body(eid, path, vc_turns, vc_goal)

        # Build full instruction
        instruction = nav_body + ". " + stop_phrase

        # Track stats
        stats['total'] += 1
        turns = extract_significant_turns(path)
        if len(turns) == 0:
            stats['no_turns'] += 1
        elif len(turns) == 1:
            stats['one_turn'] += 1
        else:
            stats['multi_turn'] += 1
        if vc_turns:
            stats['has_vc'] += 1
        else:
            stats['no_vc'] += 1

        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        new_ep["instruction"]["instruction_text"] = instruction
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    # Compute quality metrics
    import re as re2
    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]

    turns_counts = [len(re2.findall(r'\bturn\b', i, re2.I)) for i in all_insts]
    avg_turns = sum(turns_counts) / n

    hall_count = sum(1 for i in all_insts if re2.search(r'\bhallway\b|\bhall\b', i, re2.I))
    through_count = sum(1 for i in all_insts if 'through the' in i.lower())
    near_count = sum(1 for i in all_insts if re2.search(r'\bnear\b', get_stop_sentence(i), re2.I))
    door_count = sum(1 for i in all_insts if re2.search(r'\bdoor\b|\bdoorway\b|\bentrance\b', get_stop_sentence(i), re2.I))
    stair_count = sum(1 for i in all_insts if re2.search(r'\bstair\b|\bstep\b|\blanding\b', get_stop_sentence(i), re2.I))

    print(f"Generated {n} episodes")
    print(f"  Turn distribution: 0={stats['no_turns']} 1={stats['one_turn']} multi={stats['multi_turn']}")
    print(f"  VC available: {stats['has_vc']} ({stats['has_vc']/n*100:.1f}%), missing: {stats['no_vc']}")
    print()
    print(f"Quality metrics (GT targets):")
    print(f"  avg turns: {avg_turns:.3f}  [GT=0.587]")
    print(f"  hallway:   {hall_count/n*100:.1f}%  [GT=20.4%]")
    print(f"  through:   {through_count/n*100:.1f}%  [GT=27.2%]")
    print(f"  near-stop: {near_count/n*100:.1f}%  [GT=8.6%]")
    print(f"  door-stop: {door_count/n*100:.1f}%  [GT=31.1%]")
    print(f"  stair-stop:{stair_count/n*100:.1f}%  [GT=15.9%]")
    print()

    # Sample comparison
    print("=== Sample instructions ===")
    for ep in gt_data["episodes"][:5]:
        eid = ep["episode_id"]
        gen = next(e["instruction"]["instruction_text"] for e in new_episodes if e["episode_id"] == eid)
        gt = ep["instruction"]["instruction_text"]
        print(f"eid={eid}:")
        print(f"  GT:  {gt[:120]}")
        print(f"  v72: {gen[:120]}")
        print()

    # Save
    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
