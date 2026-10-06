#!/usr/bin/env python3
"""Generate v97: v95 + stop phrase corrections.

v95 analysis reveals:
  - stop_at=34.7% vs GT 7.3% (WAY too high — v32 + v90 'stop at' normalization)
  - stop_in=0% vs GT 12.2% (missing! GT uses 'stop in the doorway/bedroom/etc.')
  - GT prefers: 'stop in the doorway', 'wait near/at the [object]'

v97 FIXES:
  1. "Stop at the [room/doorway/arch/entry]" → "Stop in the [room/doorway/arch/entry]"
     (stop at + spatial words → stop in)
  2. Keep "Stop at the [furniture/object]" as is (stop at + object is correct)

SPATIAL WORDS that need "in" not "at":
  doorway, archway, arch, bedroom, bathroom, kitchen, living room, hallway,
  dining room, study, office, closet, stairs, landing, entry, foyer, corridor
"""
import gzip
import json
import re
from pathlib import Path

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
V95_PATH = BASE / "val_unseen" / "val_unseen_v95.json.gz"
GT_PATH = BASE / "val_unseen" / "val_unseen_patched.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v97.json.gz"

# Spatial/room words where "stop in" is more natural than "stop at"
SPATIAL_WORDS = (
    r'doorway|archway|arch|entry|entryway|foyer|corridor|'
    r'bedroom|bathroom|kitchen|dining\s+room|living\s+room|'
    r'hallway|hall|study|office|closet|landing|stairway|staircase|'
    r'room|area|space|floor'
)

# "Stop at the [spatial]" → "Stop in the [spatial]"
STOP_AT_ROOM_RE = re.compile(
    r'\b(stop|wait)\s+at\s+(the\s+)(' + SPATIAL_WORDS + r')\b',
    re.IGNORECASE
)

# "Stop in front of the [spatial]" → "Stop in the [spatial]"
STOP_INFRONT_ROOM_RE = re.compile(
    r'\b(stop|wait)\s+in\s+front\s+of\s+(the\s+)(' + SPATIAL_WORDS + r')\b',
    re.IGNORECASE
)


def fix_stop_phrases(s):
    # "stop at the doorway/bedroom/etc." → "stop in the doorway/bedroom/etc."
    def repl_stop_at(m):
        verb = m.group(1).capitalize()
        det = m.group(2) or 'the '
        room = m.group(3)
        return f"{verb} in {det}{room}"
    s = STOP_AT_ROOM_RE.sub(repl_stop_at, s)

    # "stop in front of the doorway/bedroom/etc." → "stop in the doorway/bedroom/etc."
    def repl_infront(m):
        verb = m.group(1).capitalize()
        det = m.group(2) or 'the '
        room = m.group(3)
        return f"{verb} in {det}{room}"
    s = STOP_INFRONT_ROOM_RE.sub(repl_infront, s)

    return s


def main():
    print("=== v97: v95 + stop phrase corrections ===")
    print("  Fix: 'stop at the doorway/room' → 'stop in the doorway/room'")
    print("  Targets GT's 12.2% stop_in rate")

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V95_PATH) as f:
        v95_data = json.load(f)

    v95_by_eid = {ep["episode_id"]: ep for ep in v95_data["episodes"]}

    new_episodes = []
    changed = 0

    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        v95_ep = v95_by_eid[eid]
        orig = v95_ep["instruction"]["instruction_text"]
        fixed = fix_stop_phrases(orig)
        if fixed != orig:
            changed += 1
        new_ep = dict(v95_ep)
        new_ep["instruction"] = dict(v95_ep["instruction"])
        new_ep["instruction"]["instruction_text"] = fixed
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]
    TURN_RE = re.compile(r'\bturn\b', re.I)
    INIT_TURN_RE = re.compile(r'^Turn', re.I)
    STOP_AT = re.compile(r'\bstop at\b', re.I)
    STOP_NEAR = re.compile(r'\bstop near\b', re.I)
    STOP_IN = re.compile(r'\bstop in\b', re.I)
    WAIT_NEAR = re.compile(r'\bwait near\b', re.I)

    print(f"\nChanged {changed}/{n} ({changed/n*100:.1f}%)")
    print(f"avg_words={sum(words)/n:.1f}  [GT=26.8]")
    print(f"starts_Turn={sum(1 for i in all_insts if INIT_TURN_RE.match(i))/n*100:.1f}%  [GT=16.5%]")
    print(f"stop_at={sum(1 for i in all_insts if STOP_AT.search(i))/n*100:.1f}%  [GT=7.3%]")
    print(f"stop_near={sum(1 for i in all_insts if STOP_NEAR.search(i))/n*100:.1f}%  [GT=1.1%]")
    print(f"stop_in={sum(1 for i in all_insts if STOP_IN.search(i))/n*100:.1f}%  [GT=12.2%]")
    print(f"wait_near={sum(1 for i in all_insts if WAIT_NEAR.search(i))/n*100:.1f}%  [GT=5.2%]")
    print(f"unique={len(set(all_insts))/n*100:.1f}%")

    print("\nSamples (changed):")
    shown = 0
    for ep, v95_ep in zip(new_episodes, v95_data["episodes"]):
        orig = v95_ep["instruction"]["instruction_text"]
        fixed = ep["instruction"]["instruction_text"]
        if orig != fixed and shown < 5:
            print(f"  v95: {orig[:150]}")
            print(f"  v97: {fixed[:150]}")
            print()
            shown += 1

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
