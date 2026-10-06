#!/usr/bin/env python3
"""Generate v209: v208 with starts_Turn reduced to GT level (16.5%).

v208 stats: h=20.4%(EXACT!) wp=10.8% tt=27.4% words=26.9 turn=17.2%
GT target:  h=20.4%            wp=10.5% tt=27.2%              turn=16.5%

v208 has 317/1839 episodes starting with "Turn" (17.2%).
GT needs 303/1839 episodes starting with "Turn" (16.5%).
Need to change 14 episodes from "Turn X..." to "Head X..." or "Go X...".
"""
import gzip
import json
import random
import re
from pathlib import Path

V208_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v208.json.gz")
GT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
OUT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v209.json.gz")

TURN_START = re.compile(r'^Turn\b', re.I)
HALL_ANY = re.compile(r'\bhallway\b|\bhall\b', re.I)
WP = re.compile(r'\bwalk past\b', re.I)
TT = re.compile(r'\bthrough the\b', re.I)
TURN_RE = re.compile(r'^Turn\b', re.I)

# Specific replacements for common "Turn X" patterns
REPLACEMENTS = {
    r'^Turn right\b': 'Head right',
    r'^Turn left\b': 'Head left',
    r'^Turn around\b': 'Head back',
    r'^Turn up\b': 'Head up',
    r'^Turn down\b': 'Head down',
    r'^Turn and\b': 'Head forward and',
}


def replace_turn_start(text: str) -> str:
    """Replace 'Turn X' at start with 'Head X' equivalent."""
    for pattern, replacement in REPLACEMENTS.items():
        m = re.match(pattern, text, re.I)
        if m:
            # Preserve case of first letter
            return replacement + text[m.end():]
    # Fallback: replace just "Turn" with "Head"
    m = TURN_START.match(text)
    if m:
        return 'Head' + text[m.end():]
    return text


def main():
    print("=== v209: v208 + starts_Turn reduction to GT level (16.5%) ===")
    rng = random.Random(42)

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V208_PATH) as f:
        v208_data = json.load(f)

    episodes = v208_data["episodes"]
    n = len(episodes)

    turn_eps = [i for i, ep in enumerate(episodes)
                if TURN_START.match(ep["instruction"]["instruction_text"])]
    gt_turn_count = int(round(n * 0.165))  # 303
    need_remove = max(0, len(turn_eps) - gt_turn_count)

    print(f"v208 starts_Turn: {len(turn_eps)}/{n} ({len(turn_eps)/n*100:.1f}%)")
    print(f"GT target: {gt_turn_count}/{n} ({gt_turn_count/n*100:.1f}%)")
    print(f"Need to change: {need_remove} episodes")

    rng.shuffle(turn_eps)
    to_modify = set(turn_eps[:need_remove])

    new_episodes = []
    changed = 0
    for i, ep in enumerate(episodes):
        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        if i in to_modify:
            orig = ep["instruction"]["instruction_text"]
            fixed = replace_turn_start(orig)
            if fixed != orig:
                changed += 1
            new_ep["instruction"]["instruction_text"] = fixed
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words_list = [len(i.split()) for i in all_insts]

    print(f"\nChanged {changed}/{n} ({changed/n*100:.1f}%) episodes")
    print(f"avg_words = {sum(words_list)/n:.1f} [GT=26.8]")
    print(f"starts_Turn = {sum(1 for i in all_insts if TURN_RE.match(i))/n*100:.1f}% [GT=16.5%, v208=17.2%]")
    print(f"hallway (h) = {sum(1 for i in all_insts if HALL_ANY.search(i))/n*100:.1f}% [GT=20.4%]")
    print(f"walk_past   = {sum(1 for i in all_insts if WP.search(i))/n*100:.1f}% [GT=10.5%]")
    print(f"through_the = {sum(1 for i in all_insts if TT.search(i))/n*100:.1f}% [GT=27.2%]")

    # Show some examples
    print("\nSample replacements:")
    for i in list(to_modify)[:5]:
        orig = episodes[i]["instruction"]["instruction_text"][:70]
        new = new_episodes[i]["instruction"]["instruction_text"][:70]
        print(f"  ORIG: {orig}")
        print(f"  NEW:  {new}")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
