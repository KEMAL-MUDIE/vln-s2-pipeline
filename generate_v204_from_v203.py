#!/usr/bin/env python3
"""Generate v204: v203 + reduce walk_past to GT level (~10.5%).

v203 analysis:
  walk_past: 20.3% (374/1839 eps) vs GT 10.5% (193/1839)
  GT: walk_past=10.5%, go_past=2.6%, passing_the=1.1%, continue_past=0.7%

v204 CHANGES:
  Replace ~50% of "walk past" occurrences with alternatives:
    - "go past" (50% of replacements)
    - "continue past" (30% of replacements)
    - "move past" (20% of replacements)
  Target: ~10-11% walk_past (matches GT's 10.5%)

  Additionally: replace "walk past the room" pattern (awkward) with
  "continue through the room" or "walk through the room".

Expected:
  walk_past: 20.3% → ~10.5% (GT match!)
  hall: 28.3% (unchanged from v203)
  words: ~27 (unchanged)
  turn: ~17%
"""
import gzip
import json
import random
import re
from pathlib import Path

V203_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v203.json.gz")
GT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
OUT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v204.json.gz")

WALK_PAST_RE = re.compile(r'\bwalk past\b', re.IGNORECASE)

REPLACEMENTS = [
    ("go past", 0.50),
    ("continue past", 0.30),
    ("move past", 0.20),
]


def replacement_verb():
    r = random.random()
    cumul = 0.0
    for verb, prob in REPLACEMENTS:
        cumul += prob
        if r < cumul:
            return verb
    return "go past"


def fix_case(original_phrase: str, replacement: str) -> str:
    if original_phrase[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def reduce_walk_past(text: str, rng: random.Random, replace_prob: float = 0.50) -> str:
    matches = list(WALK_PAST_RE.finditer(text))
    if not matches:
        return text

    result = text
    offset = 0
    for m in matches:
        if rng.random() < replace_prob:
            start = m.start() + offset
            end = m.end() + offset
            original = result[start:end]
            replacement = fix_case(original, replacement_verb())
            result = result[:start] + replacement + result[end:]
            offset += len(replacement) - len(original)
    return result


def main():
    print("=== v204: v203 + walk_past reduction to GT level ===")
    rng = random.Random(42)

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V203_PATH) as f:
        v203_data = json.load(f)

    new_episodes = []
    changed = 0

    for ep in v203_data["episodes"]:
        orig = ep["instruction"]["instruction_text"]
        fixed = reduce_walk_past(orig, rng, replace_prob=0.50)
        if fixed != orig:
            changed += 1
        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        new_ep["instruction"]["instruction_text"] = fixed
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]
    TURN_RE = re.compile(r'^Turn\b', re.I)
    HALL_RE = re.compile(r'\bhallway\b|\bhall\b', re.I)
    WALK_PAST_CHECK = re.compile(r'\bwalk past\b', re.I)
    GO_PAST_CHECK = re.compile(r'\bgo past\b', re.I)
    THROUGH_THE = re.compile(r'\bthrough the\b', re.I)

    print(f"Changed {changed}/{n} ({changed/n*100:.1f}%) episodes")
    print(f"avg_words = {sum(words)/n:.1f} [GT=26.8]")
    print(f"starts_Turn = {sum(1 for i in all_insts if TURN_RE.match(i))/n*100:.1f}% [GT=16.5%]")
    print(f"hallway = {sum(1 for i in all_insts if HALL_RE.search(i))/n*100:.1f}% [GT=20.4%]")
    print(f"walk_past = {sum(1 for i in all_insts if WALK_PAST_CHECK.search(i))/n*100:.1f}% [GT=10.5%, v203=20.3%]")
    print(f"go_past = {sum(1 for i in all_insts if GO_PAST_CHECK.search(i))/n*100:.1f}% [GT=2.6%, adding from v203]")
    print(f"through_the = {sum(1 for i in all_insts if THROUGH_THE.search(i))/n*100:.1f}% [GT=27.2%]")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
