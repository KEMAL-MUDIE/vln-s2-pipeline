#!/usr/bin/env python3
"""
Generate v84: best-of-both combination.
  v83: scene-accurate vc_turn landmarks for 0-turn episodes (past=52.7%, unique=93.8%)
  v82: geometry-calibrated 'through the area' injection for long-preturn turn episodes

v84 = v83 (0-turn) + v82 through-injection (turn episodes with pre-turn >= 7.44m)

Why:
  - v83 0-turn instructions are excellent (scene-accurate vc_turns)
  - v82's through injection adds calibrated traversal context for long paths before a turn
  - These are INDEPENDENT improvements — they don't interfere

Expected quality:
  through:   ~10% (v82 effect on turn eps, GT non-hallway through ~10-12%)
  hallway:   ~4.9% (maintained from v80)
  past:      ~52.7% (v83 effect on 0-turn eps)
  avg_words: ~24-26 (between v83=23.0 and v80=24.6)
  toward:    ~99.5%
  unique:    ~94% (v83 brought high uniqueness)
"""
import gzip
import json
import math
import re
from pathlib import Path

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
V83_PATH = BASE / "val_unseen" / "val_unseen_v83.json.gz"
GT_PATH = BASE / "val_unseen" / "val_unseen.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v84.json.gz"

TURN_THRESHOLD = 80
INJECT_THRESHOLD = 7.44  # same threshold as v82

WALK_FORWARD_RE = re.compile(r'^Walk forward\b', re.IGNORECASE)
HALLWAY_RE = re.compile(r'\b(hallway|hall)\b', re.IGNORECASE)
ROOM_NAMES = re.compile(
    r'\b(kitchen|bedroom|bathroom|living room|dining room|office|study|library|'
    r'garage|basement|attic|foyer|entryway|lobby|corridor|passage)\b',
    re.IGNORECASE
)


def extract_first_turn_dist(path, threshold=TURN_THRESHOLD):
    cumul = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dz = path[i][2] - path[i-1][2]
        cumul.append(cumul[-1] + math.sqrt(dx*dx + dz*dz))
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
            return cumul[i]
    return None


def count_turns(path, threshold=TURN_THRESHOLD):
    return extract_first_turn_dist(path, threshold) is not None


def inject_through(inst):
    return WALK_FORWARD_RE.sub('Walk through the area', inst, count=1)


def main():
    print("=== v84 generator: v83 (0-turn vc_lms) + v82 through injection (turn eps) ===")
    print("  Best-of-both: scene-accurate 0-turn + calibrated through for turn eps")
    print()

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V83_PATH) as f:
        v83_data = json.load(f)

    v83_by_eid = {ep["episode_id"]: ep for ep in v83_data["episodes"]}

    new_episodes = []
    stats = {
        'zero_turn_kept': 0,
        'turn_injected': 0,
        'turn_skipped_has_through': 0,
        'turn_skipped_short': 0,
    }

    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        path = ep["reference_path"]

        v83_ep = v83_by_eid[eid]
        v83_inst = v83_ep["instruction"]["instruction_text"]

        pt_dist = extract_first_turn_dist(path)

        if pt_dist is None:
            # 0-turn: keep v83's vc_turn-enriched instruction
            new_inst = v83_inst
            stats['zero_turn_kept'] += 1
        elif "through the" in v83_inst.lower():
            new_inst = v83_inst
            stats['turn_skipped_has_through'] += 1
        elif pt_dist >= INJECT_THRESHOLD and WALK_FORWARD_RE.match(v83_inst):
            new_inst = inject_through(v83_inst)
            stats['turn_injected'] += 1
        else:
            new_inst = v83_inst
            stats['turn_skipped_short'] += 1

        new_ep = dict(v83_ep)
        new_ep["instruction"] = dict(v83_ep["instruction"])
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
    rooms = sum(1 for i in all_insts if ROOM_NAMES.search(i))

    print(f"\nQuality metrics (n={n}):")
    print(f"  avg turns: {avg_turns_kw:.3f}  [GT=0.587, v80=0.569]")
    print(f"  avg words: {sum(words)/n:.1f}  [GT=26.8, v83=23.0, v82=24.7]")
    print(f"  hallway:   {hall/n*100:.1f}%  [GT=20.4%, v80=4.9%]")
    print(f"  room names:{rooms/n*100:.1f}%")
    print(f"  through:   {thru/n*100:.1f}%  [GT=27.2%, v82=9.9%, v83=1.0%]")
    print(f"  toward:    {toward/n*100:.1f}%  [v80=99.5%]")
    print(f"  past:      {past_kw/n*100:.1f}%  [v83=52.7%]")
    print(f"  near-stop: {near/n*100:.1f}%  [GT=8.6%]")
    print(f"  unique:    {len(set(all_insts))/n*100:.1f}%")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
