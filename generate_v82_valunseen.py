#!/usr/bin/env python3
"""
Generate v82: geometry-calibrated 'through the area' injection into v80.

Analysis:
  GT through=27.2% includes through-hallway (~20%) → non-hallway through target ≈ 10-12%
  v80 through=3.5% (64/1839 eps) — too low, no spatial traversal language
  v81 target 15-25% — may overshoot non-hallway target

v82 design: POST-PROCESS v80 (no Gemma needed)
  For each turn episode where:
    - pre-turn distance >= 7.44m (top 15% of pre-turn distances)
    - instruction does NOT already contain "through the"
  → Replace "Walk forward" at instruction start with "Walk through the area"

  Threshold 7.44m calibrated to inject into exactly 119 turn episodes:
    64 (current through in v80) + 119 = 183 total = 10.0% of 1839 episodes

Result:
  through: ~10.1% (calibrated, vs GT=27.2% but we ban hallway so target ~10-12%)
  hallway: ~4.9% (same as v80, unchanged)
  avg_words: ~25.0 (adds "the area" = 2 words to 119 instructions)
  toward: ~99.5% (same as v80)
  unique: ~89% (119 instructions modified)

Why 7.44m threshold is correct:
  - Long pre-turn distance (>7.44m) means the robot walks >7.4m before turning
  - This is genuinely a traversal through space, not just a short pivot
  - "Walk through the area" accurately describes this behavior
  - Shorter paths (<7.44m) don't warrant "through" — it would be misleading
"""
import gzip
import json
import math
import re
from pathlib import Path

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
GT_PATH = BASE / "val_unseen" / "val_unseen.json.gz"
V80_PATH = BASE / "val_unseen" / "val_unseen_v80.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v82.json.gz"

TURN_THRESHOLD = 80
INJECT_THRESHOLD = 7.44  # meters pre-turn distance


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


WALK_FORWARD_RE = re.compile(r'^Walk forward\b', re.IGNORECASE)
HALLWAY_RE = re.compile(r'\b(hallway|hall)\b', re.IGNORECASE)
ROOM_NAMES = re.compile(
    r'\b(kitchen|bedroom|bathroom|living room|dining room|office|study|library|'
    r'garage|basement|attic|foyer|entryway|lobby|corridor|passage)\b',
    re.IGNORECASE
)


def inject_through(inst):
    """Replace 'Walk forward' at instruction start with 'Walk through the area'."""
    return WALK_FORWARD_RE.sub('Walk through the area', inst, count=1)


def main():
    print("=== v82 generator: geometry-calibrated 'through the area' injection ===")
    print(f"  Base: v80 (through=3.5%, avg_words=24.6)")
    print(f"  Inject 'through the area' for turn eps with pre-turn >= {INJECT_THRESHOLD}m")
    print(f"  Target: through~10% (calibrated to GT non-hallway through estimate)")
    print()

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V80_PATH) as f:
        v80_data = json.load(f)

    v80_by_eid = {ep["episode_id"]: ep for ep in v80_data["episodes"]}

    new_episodes = []
    stats = {
        'injected': 0,
        'skipped_has_through': 0,
        'skipped_short_preturn': 0,
        'zero_turn': 0,
    }

    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        path = ep["reference_path"]

        v80_ep = v80_by_eid[eid]
        v80_inst = v80_ep["instruction"]["instruction_text"]

        pt_dist = extract_first_turn_dist(path)

        new_inst = v80_inst  # default: keep v80 instruction

        if pt_dist is None:
            stats['zero_turn'] += 1
        elif "through the" in v80_inst.lower():
            stats['skipped_has_through'] += 1
        elif pt_dist >= INJECT_THRESHOLD and WALK_FORWARD_RE.match(v80_inst):
            new_inst = inject_through(v80_inst)
            stats['injected'] += 1
        else:
            stats['skipped_short_preturn'] += 1

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
    near = sum(1 for i in all_insts if re.search(r'\bnear\b', get_stop_sent(i), re.I))
    rooms = sum(1 for i in all_insts if ROOM_NAMES.search(i))

    print(f"\nQuality metrics (n={n}):")
    print(f"  avg turns: {avg_turns_kw:.3f}  [GT=0.587, v80=0.569]")
    print(f"  avg words: {sum(words)/n:.1f}  [GT=26.8, v80=24.6]")
    print(f"  hallway:   {hall/n*100:.1f}%  [GT=20.4%, v80=4.9%]")
    print(f"  room names:{rooms/n*100:.1f}%  [target ~0%]")
    print(f"  through:   {thru/n*100:.1f}%  [GT=27.2%, v80=3.5%, target=10-12%]")
    print(f"  toward:    {toward/n*100:.1f}%  [v80=99.5%]")
    print(f"  near-stop: {near/n*100:.1f}%  [GT=8.6%]")
    print(f"  unique:    {len(set(all_insts))/n*100:.1f}%")

    # Sample injected instructions
    print(f"\nSample injected instructions:")
    for ep in gt_data["episodes"][:3]:
        eid = ep["episode_id"]
        orig = v80_by_eid[eid]["instruction"]["instruction_text"]
        new = next(e["instruction"]["instruction_text"] for e in new_episodes if e["episode_id"] == eid)
        if orig != new:
            print(f"  ORIG: {orig[:100]}")
            print(f"  NEW:  {new[:100]}")
            print()

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
