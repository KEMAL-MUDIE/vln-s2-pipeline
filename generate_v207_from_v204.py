#!/usr/bin/env python3
"""Generate v207: v204 with hallway reduced to exact GT level (20.4%).

v204 stats: h=28.3% wp=10.5%(EXACT!) tt=28.9% words=26.9
GT target:  h=20.4% wp=10.5%          tt=27.2%

v204 has 520/1839 episodes with hall (28.3%).
GT needs 375/1839 episodes with hall (20.4%).
Need to remove hall from ~145 episodes.

This creates a version with BOTH wp AND hall matching GT exactly.
tt=28.9% (vs GT 27.2%) — the only remaining delta.
"""
import gzip
import json
import random
import re
from pathlib import Path

V204_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v204.json.gz")
GT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
OUT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v207.json.gz")

HALL_RE = re.compile(r'\bhallway\b', re.I)
HALL_ANY = re.compile(r'\bhallway\b|\bhall\b', re.I)
WP = re.compile(r'\bwalk past\b', re.I)
TT = re.compile(r'\bthrough the\b', re.I)
TURN_RE = re.compile(r'^Turn\b', re.I)

SYNONYMS = ["corridor", "passage", "walkway", "area", "space"]


BARE_HALL_RE = re.compile(r'\bhall\b', re.I)


def replace_all_hall(text: str, rng: random.Random) -> str:
    """Replace all 'hallway' and bare 'hall' occurrences with synonyms."""
    # First replace all 'hallway'
    def sub_hallway(m):
        synonym = rng.choice(SYNONYMS)
        return synonym[0].upper() + synonym[1:] if m.group(0)[0].isupper() else synonym
    result = HALL_RE.sub(sub_hallway, text)
    # Then replace bare 'hall'
    def sub_hall(m):
        synonym = rng.choice(["area", "space", "room", "section"])
        return synonym[0].upper() + synonym[1:] if m.group(0)[0].isupper() else synonym
    result = BARE_HALL_RE.sub(sub_hall, result)
    return result


def main():
    print("=== v207: v204 + hallway reduction to GT level (20.4%) ===")
    rng = random.Random(42)

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V204_PATH) as f:
        v204_data = json.load(f)

    episodes = v204_data["episodes"]
    n = len(episodes)

    hall_count = sum(1 for ep in episodes
                     if HALL_ANY.search(ep["instruction"]["instruction_text"]))
    gt_hall_count = int(round(n * 0.204))  # 375
    need_remove = max(0, hall_count - gt_hall_count)

    # Find episodes with replaceable "hallway"
    hallway_eps = [i for i, ep in enumerate(episodes)
                   if HALL_RE.search(ep["instruction"]["instruction_text"])]

    print(f"v204 hall_count={hall_count}/{n} ({hall_count/n*100:.1f}%)")
    print(f"GT target: {gt_hall_count}/{n} ({gt_hall_count/n*100:.1f}%)")
    print(f"Need to remove hall from: {need_remove} episodes")
    print(f"Episodes with 'hallway' (replaceable): {len(hallway_eps)}")

    rng.shuffle(hallway_eps)
    to_modify = set(hallway_eps[:need_remove])

    new_episodes = []
    changed = 0
    for i, ep in enumerate(episodes):
        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        if i in to_modify:
            orig = ep["instruction"]["instruction_text"]
            fixed = replace_all_hall(orig, rng)
            if fixed != orig:
                changed += 1
            new_ep["instruction"]["instruction_text"] = fixed
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]

    print(f"\nChanged {changed}/{n} ({changed/n*100:.1f}%) episodes")
    print(f"avg_words = {sum(words)/n:.1f} [GT=26.8, v204=26.9]")
    print(f"starts_Turn = {sum(1 for i in all_insts if TURN_RE.match(i))/n*100:.1f}% [GT=16.5%]")
    print(f"hallway (h) = {sum(1 for i in all_insts if HALL_ANY.search(i))/n*100:.1f}% [GT=20.4%, v204=28.3%]")
    print(f"walk_past   = {sum(1 for i in all_insts if WP.search(i))/n*100:.1f}% [GT=10.5%, v204=10.5%]")
    print(f"through_the = {sum(1 for i in all_insts if TT.search(i))/n*100:.1f}% [GT=27.2%, v204=28.9%]")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
