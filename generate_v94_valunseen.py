#!/usr/bin/env python3
"""
Generate v94: v93 (no initial turns) + v92 conservative simplification.
v93 removes initial turn prefixes from v90.
v94 = v93 + double-pass removal + stop-when conversion + hallway through→down.
This combines: vocab normalization + no initial turns + waypoint reduction.
"""
import gzip
import json
import re
from pathlib import Path

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
GT_PATH = BASE / "val_unseen" / "val_unseen_patched.json.gz"
V93_PATH = BASE / "val_unseen" / "val_unseen_v93.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v94.json.gz"

DOUBLE_PASS_RE = re.compile(
    r'\b(?:pass(?:ing)?|walk(?:ing)?\s+(?:forward\s+)?past|walk\s+by)\s+the\s+(?:[\w]+\s*){1,5}and\s+(?:pass(?:ing)?|walk(?:ing)?\s+(?:forward\s+)?past|walk\s+by)\s+the\s+(?:[\w]+\s*){1,5}(?=[.,])',
    re.IGNORECASE
)
STOP_WHEN_RE = re.compile(
    r'(stop|wait)\s+(?:near|by|at|next to|in front of)\s+the\s+[\w\s]+?\s+(?:when|once|after)\s+you\s+reach\s+the\s+([\w\s]+?)(?=\.|$)',
    re.IGNORECASE
)
HALLWAY_THROUGH_RE = re.compile(r'\bwalk through the hallway\b', re.IGNORECASE)
GO_THROUGH_HALL_RE = re.compile(r'\bgo through the hallway\b', re.IGNORECASE)
CONT_THROUGH_HALL_RE = re.compile(r'\bcontinue through the hallway\b', re.IGNORECASE)


def simplify(s):
    s = DOUBLE_PASS_RE.sub('walk forward', s)
    def stop_when_repl(m):
        verb = m.group(1).capitalize()
        room = m.group(2).strip().lower()
        return f"{verb} in the {room}"
    s = STOP_WHEN_RE.sub(stop_when_repl, s)
    s = HALLWAY_THROUGH_RE.sub('walk down the hallway', s)
    s = GO_THROUGH_HALL_RE.sub('go down the hallway', s)
    s = CONT_THROUGH_HALL_RE.sub('continue down the hallway', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'(?<=\. )([a-z])', lambda m: m.group(1).upper(), s)
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s


def main():
    print("=== v94: v93 (no initial turns) + conservative simplification ===")
    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V93_PATH) as f:
        v93_data = json.load(f)
    v93_by_eid = {ep["episode_id"]: ep for ep in v93_data["episodes"]}
    new_episodes = []
    changed = 0
    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        v93_ep = v93_by_eid[eid]
        orig = v93_ep["instruction"]["instruction_text"]
        simp = simplify(orig)
        if simp != orig:
            changed += 1
        new_ep = dict(v93_ep)
        new_ep["instruction"] = dict(v93_ep["instruction"])
        new_ep["instruction"]["instruction_text"] = simp
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]
    TURN_RE = re.compile(r'\bturn\b', re.I)
    print(f"Changed {changed}/{n} ({changed/n*100:.1f}%)")
    print(f"avg_words={sum(words)/n:.1f}  turn%={sum(1 for i in all_insts if TURN_RE.search(i))/n*100:.1f}%  unique={len(set(all_insts))/n*100:.1f}%")

    print("\nSamples (changed):")
    shown = 0
    for ep, v93_ep in zip(new_episodes, v93_data["episodes"]):
        if ep["instruction"]["instruction_text"] != v93_ep["instruction"]["instruction_text"] and shown < 5:
            print(f"  v93: {v93_ep['instruction']['instruction_text'][:150]}")
            print(f"  v94: {ep['instruction']['instruction_text'][:150]}")
            print()
            shown += 1

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
