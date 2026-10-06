#!/usr/bin/env python3
"""
Generate v92: Conservative v90 simplification — only remove double-pass clauses.

v91 was too aggressive: "Turn left at the X and walk into the kitchen" → "Turn left" (lost kitchen).
v92 is more conservative:
  1. ONLY remove "pass X AND pass Y" → keep single pass clauses
  2. KEEP "Turn left at the X" → just removes the "at the X" from turns
  3. Simpler stop-when → stop-in conversion (same as v91)

This tests whether just the double-pass removal is enough to improve SR.
"""
import gzip
import json
import re
from pathlib import Path

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
GT_PATH = BASE / "val_unseen" / "val_unseen_patched.json.gz"
V90_PATH = BASE / "val_unseen" / "val_unseen_v90.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v92.json.gz"

# Double pass: "pass X and pass Y" → "walk forward"
DOUBLE_PASS_RE = re.compile(
    r'\b(?:pass(?:ing)?|walk(?:ing)?\s+(?:forward\s+)?past|walk\s+by)\s+the\s+(?:[\w]+\s*){1,5}and\s+(?:pass(?:ing)?|walk(?:ing)?\s+(?:forward\s+)?past|walk\s+by)\s+the\s+(?:[\w]+\s*){1,5}(?=[.,])',
    re.IGNORECASE
)

# "stop/wait X when/once you reach the Y" → "stop in the Y"
STOP_WHEN_RE = re.compile(
    r'(stop|wait)\s+(?:near|by|at|next to|in front of)\s+the\s+[\w\s]+?\s+(?:when|once|after)\s+you\s+reach\s+the\s+([\w\s]+?)(?=\.|$)',
    re.IGNORECASE
)

# Hallway normalization (same as v228)
HALLWAY_THROUGH_RE = re.compile(r'\bwalk through the hallway\b', re.IGNORECASE)
GO_THROUGH_HALL_RE = re.compile(r'\bgo through the hallway\b', re.IGNORECASE)
CONT_THROUGH_HALL_RE = re.compile(r'\bcontinue through the hallway\b', re.IGNORECASE)


def simplify(instruction):
    s = instruction

    # Remove double-pass
    s = DOUBLE_PASS_RE.sub('walk forward', s)

    # Normalize stop-when
    def stop_when_repl(m):
        verb = m.group(1).capitalize()
        room = m.group(2).strip().lower()
        return f"{verb} in the {room}"
    s = STOP_WHEN_RE.sub(stop_when_repl, s)

    # Hallway: through→down (same as auto chain v228)
    s = HALLWAY_THROUGH_RE.sub('walk down the hallway', s)
    s = GO_THROUGH_HALL_RE.sub('go down the hallway', s)
    s = CONT_THROUGH_HALL_RE.sub('continue down the hallway', s)

    # Clean up
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'(?<=\. )([a-z])', lambda m: m.group(1).upper(), s)
    if s and s[0].islower():
        s = s[0].upper() + s[1:]

    return s


def main():
    print("=== v92: Conservative v90 simplification (double-pass removal + stop-when fix) ===")

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V90_PATH) as f:
        v90_data = json.load(f)

    v90_by_eid = {ep["episode_id"]: ep for ep in v90_data["episodes"]}

    changed = 0
    new_episodes = []
    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        v90_ep = v90_by_eid[eid]
        orig = v90_ep["instruction"]["instruction_text"]
        simp = simplify(orig)
        if simp != orig:
            changed += 1
        new_ep = dict(v90_ep)
        new_ep["instruction"] = dict(v90_ep["instruction"])
        new_ep["instruction"]["instruction_text"] = simp
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]
    HALLWAY_RE = re.compile(r'\b(hallway|hall|corridor)\b', re.I)
    TURN_RE = re.compile(r'\bturn\b', re.I)

    print(f"Changed {changed}/{n} ({changed/n*100:.1f}%)")
    print(f"avg_words: {sum(words)/n:.1f}  [v90=25.6, GT=26.8]")
    print(f"hallway:   {sum(1 for i in all_insts if HALLWAY_RE.search(i))/n*100:.1f}%")
    print(f"through:   {sum(1 for i in all_insts if 'through' in i.lower())/n*100:.1f}%")
    print(f"turn%:     {sum(1 for i in all_insts if TURN_RE.search(i))/n*100:.1f}%")
    print(f"unique:    {len(set(all_insts))/n*100:.1f}%")

    print(f"\nSamples:")
    shown = 0
    for ep, v90_ep in zip(new_episodes, v90_data["episodes"]):
        orig = v90_ep["instruction"]["instruction_text"]
        simp = ep["instruction"]["instruction_text"]
        if orig != simp and shown < 6:
            print(f"  v90: {orig[:160]}")
            print(f"  v92: {simp[:160]}")
            print()
            shown += 1

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
