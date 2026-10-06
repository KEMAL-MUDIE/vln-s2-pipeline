#!/usr/bin/env python3
"""Generate v206: v135 with hallway reduced to GT level (20.4%).

v135 stats: h=21.3% wp=9.1% tt=27.2%(EXACT) words=25.6
GT target:  h=20.4% wp=10.5% tt=27.2%

v135 has 392/1839 episodes with hallway (21.3%).
GT needs 375/1839 episodes with hallway (20.4%).
Need to REMOVE hallway from ~17 episodes.

Strategy: In episodes that have "hallway", replace ONE "hallway" occurrence
with "corridor" or "passage" or "walkway" to reduce the count.
"""
import gzip
import json
import random
import re
from pathlib import Path

V135_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v135.json.gz")
GT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
OUT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v206.json.gz")

HALL_RE = re.compile(r'\bhallway\b', re.I)
WP = re.compile(r'\bwalk past\b', re.I)
TT = re.compile(r'\bthrough the\b', re.I)
HALL_ANY = re.compile(r'\bhallway\b|\bhall\b', re.I)
TURN_RE = re.compile(r'^Turn\b', re.I)

SYNONYMS = ["corridor", "passage", "walkway", "area"]


def replace_one_hallway(text: str, rng: random.Random) -> str:
    """Replace the first 'hallway' in text with a synonym."""
    m = HALL_RE.search(text)
    if not m:
        return text
    orig = m.group(0)
    synonym = rng.choice(SYNONYMS)
    if orig[0].isupper():
        synonym = synonym[0].upper() + synonym[1:]
    return text[:m.start()] + synonym + text[m.end():]


def main():
    print("=== v206: v135 + hallway reduction to GT level (20.4%) ===")
    rng = random.Random(42)

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V135_PATH) as f:
        v135_data = json.load(f)

    episodes = v135_data["episodes"]
    n = len(episodes)

    hall_eps = [i for i, ep in enumerate(episodes)
                if HALL_ANY.search(ep["instruction"]["instruction_text"])]
    hall_count = len(hall_eps)
    gt_hall_count = int(n * 0.204)  # 375
    need_remove = max(0, hall_count - gt_hall_count)

    print(f"v135 hall_count={hall_count}/{n} ({hall_count/n*100:.1f}%)")
    print(f"GT target: {gt_hall_count}/{n} ({gt_hall_count/n*100:.1f}%)")
    print(f"Need to remove hall from: {need_remove} episodes")

    # Only replace episodes that have "hallway" (the word we replace), not just "hall"
    hallway_eps = [i for i, ep in enumerate(episodes)
                   if HALL_RE.search(ep["instruction"]["instruction_text"])]
    print(f"Episodes with 'hallway' (replaceable): {len(hallway_eps)}")

    # Sample episodes to modify
    rng.shuffle(hallway_eps)
    to_modify = set(hallway_eps[:need_remove])

    new_episodes = []
    changed = 0
    for i, ep in enumerate(episodes):
        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        if i in to_modify:
            orig = ep["instruction"]["instruction_text"]
            fixed = replace_one_hallway(orig, rng)
            if fixed != orig:
                changed += 1
            new_ep["instruction"]["instruction_text"] = fixed
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]

    print(f"\nChanged {changed}/{n} ({changed/n*100:.1f}%) episodes")
    print(f"avg_words = {sum(words)/n:.1f} [GT=26.8]")
    print(f"starts_Turn = {sum(1 for i in all_insts if TURN_RE.match(i))/n*100:.1f}% [GT=16.5%]")
    print(f"hallway (h) = {sum(1 for i in all_insts if HALL_ANY.search(i))/n*100:.1f}% [GT=20.4%, v135=21.3%]")
    print(f"walk_past   = {sum(1 for i in all_insts if WP.search(i))/n*100:.1f}% [GT=10.5%, v135=9.1%]")
    print(f"through_the = {sum(1 for i in all_insts if TT.search(i))/n*100:.1f}% [GT=27.2%, v135=27.2%]")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
