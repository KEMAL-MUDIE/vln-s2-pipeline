#!/usr/bin/env python3
"""
Generate v93: v90 with initial turn prefixes REMOVED.

CRITICAL HYPOTHESIS:
  GT achieves 63.57% with initial turns stated for only 19.5% of needed episodes.
  v213/v90 states initial turns for 82.5% of episodes → achieves only 36.87%.

  CONCLUSION: Adding initial turn prefixes HURTS SR for this model.
  Reason: InternVLA-N1 dual-system uses S1 (visual nav) to handle orientation.
  When the instruction says "Turn left and walk forward", S2 tries to find a
  left turn before the path even starts — but S1 has already corrected orientation.
  The redundant "Turn left" from start_rotation confuses S2's waypoint planning.

v93 = v90 (vocabulary normalized v213) WITH initial turn prefixes removed.
  - "Turn slightly left and [instruction]" → "[Instruction]" (capitalized)
  - "Turn left and [instruction]" → "[Instruction]"
  - "Turn right and [instruction]" → "[Instruction]"
  - "Turn around and [instruction]" → "[Instruction]"
  - "Turn slightly right and [instruction]" → "[Instruction]"
  These are ONLY removed when they represent initial_turn (start_rotation correction)
  NOT when they represent actual first path turn.

DETECTION: An initial turn prefix is identifiable as "Turn [dir] and [action]" at
  the very beginning of the instruction, where [action] is NOT another turn but
  a movement (walk, go, exit, leave, pass, etc.).
"""
import gzip
import json
import re
from pathlib import Path

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
GT_PATH = BASE / "val_unseen" / "val_unseen_patched.json.gz"
V90_PATH = BASE / "val_unseen" / "val_unseen_v90.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v93.json.gz"

# Pattern: initial turn prefix — starts instruction, followed by movement (not another turn)
# "Turn [slightly] [left/right/around] and [walk/go/exit/leave/pass/continue/head...]"
INIT_TURN_PREFIX_RE = re.compile(
    r'^Turn\s+(?:slightly\s+)?(?:left|right|around)\s+and\s+'
    r'(?=walk|go|exit|leave|pass|continue|head|move|proceed|step|travel)',
    re.IGNORECASE
)


def remove_initial_turn(instruction):
    """Remove initial turn prefix if it matches the pattern."""
    m = INIT_TURN_PREFIX_RE.match(instruction)
    if not m:
        return instruction

    # Remove the prefix
    rest = instruction[m.end():]
    if not rest:
        return instruction

    # Capitalize first letter
    return rest[0].upper() + rest[1:]


def main():
    print("=== v93: v90 with initial turn prefixes REMOVED ===")
    print("  Hypothesis: initial turns HURT this model (S1 visual system handles orientation)")
    print("  GT: 63.57% with initial turns stated for only 19.5% of episodes")
    print("  v213/v90: 36.87% with initial turns stated for 82.5% of episodes")
    print("  Removing initial turns from v90 may restore GT-like performance")
    print()

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V90_PATH) as f:
        v90_data = json.load(f)

    v90_by_eid = {ep["episode_id"]: ep for ep in v90_data["episodes"]}

    removed = 0
    total = 0
    new_episodes = []

    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        v90_ep = v90_by_eid[eid]
        orig = v90_ep["instruction"]["instruction_text"]
        cleaned = remove_initial_turn(orig)

        if cleaned != orig:
            removed += 1
        total += 1

        new_ep = dict(v90_ep)
        new_ep["instruction"] = dict(v90_ep["instruction"])
        new_ep["instruction"]["instruction_text"] = cleaned
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    print(f"Removed initial turn prefix from {removed}/{total} ({removed/total*100:.1f}%)")

    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]
    HALLWAY_RE = re.compile(r'\b(hallway|hall|corridor)\b', re.I)
    TURN_RE = re.compile(r'\bturn\b', re.I)
    INIT_TURN_RE = re.compile(r'^Turn (?:slightly )?(?:left|right|around)', re.I)

    print(f"\nQuality metrics (n={n}):")
    print(f"  avg_words: {sum(words)/n:.1f}  [v90=25.6, GT=26.8]")
    print(f"  hallway:   {sum(1 for i in all_insts if HALLWAY_RE.search(i))/n*100:.1f}%")
    print(f"  through:   {sum(1 for i in all_insts if 'through' in i.lower())/n*100:.1f}%")
    print(f"  turn%:     {sum(1 for i in all_insts if TURN_RE.search(i))/n*100:.1f}%")
    print(f"  starts_with_turn: {sum(1 for i in all_insts if INIT_TURN_RE.match(i))/n*100:.1f}%")
    print(f"  unique:    {len(set(all_insts))/n*100:.1f}%")

    print(f"\nSample comparisons:")
    shown = 0
    for ep, v90_ep in zip(new_episodes, v90_data["episodes"]):
        orig = v90_ep["instruction"]["instruction_text"]
        cleaned = ep["instruction"]["instruction_text"]
        if orig != cleaned and shown < 8:
            print(f"  v90: {orig[:160]}")
            print(f"  v93: {cleaned[:160]}")
            print()
            shown += 1

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
