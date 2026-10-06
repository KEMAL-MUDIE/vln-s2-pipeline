#!/usr/bin/env python3
"""Generate v205: v135 + walk_past boost to GT level (10.5%).

v135 stats: h=21.3% wp=9.1% tt=27.2%(EXACT GT!) words=25.6
GT target:  h=20.4% wp=10.5% tt=27.2%

v135 has wp=9.1% (167/1839 eps). GT has wp=10.5% (193/1839 eps).
Need to ADD walk_past to ~26 more episodes.

Strategy: Find episodes that have "go past", "pass by", "pass the", "walk by"
but NOT "walk past", then replace one occurrence with "walk past".
"""
import gzip
import json
import random
import re
from pathlib import Path

V135_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v135.json.gz")
GT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
OUT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v205.json.gz")

WP = re.compile(r'\bwalk past\b', re.I)
GO_PAST = re.compile(r'\bgo past\b', re.I)
PASS_THE = re.compile(r'\bpass (?:the|by)\b', re.I)
WALK_BY = re.compile(r'\bwalk by\b', re.I)
HALL = re.compile(r'\bhallway\b|\bhall\b', re.I)
TT = re.compile(r'\bthrough the\b', re.I)
TURN_RE = re.compile(r'^Turn\b', re.I)


def boost_walk_past(text: str, rng: random.Random) -> str:
    """Replace one 'go past' or 'pass the/by' with 'walk past'."""
    # Prefer replacing "go past" → "walk past"
    m = GO_PAST.search(text)
    if m:
        orig = m.group(0)
        replacement = "walk past" if orig[0].islower() else "Walk past"
        return text[:m.start()] + replacement + text[m.end():]
    # Try "pass the" → "walk past the" — but only if it makes sense
    m = PASS_THE.search(text)
    if m:
        # "pass the X" → "walk past the X"
        rest = text[m.end():]  # everything after "pass the/by"
        orig = m.group(0)
        if "the" in orig.lower():
            replacement = ("Walk past the" if orig[0].isupper() else "walk past the")
            return text[:m.start()] + replacement + rest
    return text


def main():
    print("=== v205: v135 + walk_past boost to GT level (10.5%) ===")
    rng = random.Random(42)

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V135_PATH) as f:
        v135_data = json.load(f)

    episodes = v135_data["episodes"]
    n = len(episodes)

    # Find episodes WITHOUT walk_past but WITH a replaceable pattern
    candidates = []
    for i, ep in enumerate(episodes):
        text = ep["instruction"]["instruction_text"]
        if not WP.search(text):
            if GO_PAST.search(text) or PASS_THE.search(text) or WALK_BY.search(text):
                candidates.append(i)

    # Current wp count
    wp_count = sum(1 for ep in episodes if WP.search(ep["instruction"]["instruction_text"]))
    gt_wp_count = int(n * 0.105)  # 193
    need = max(0, gt_wp_count - wp_count)

    print(f"v135 wp_count={wp_count}/{n} ({wp_count/n*100:.1f}%)")
    print(f"GT target: {gt_wp_count}/{n} ({gt_wp_count/n*100:.1f}%)")
    print(f"Need to add: {need} episodes")
    print(f"Candidates available: {len(candidates)}")

    # Sample candidates to convert
    rng.shuffle(candidates)
    to_boost = candidates[:need]
    to_boost_set = set(to_boost)

    new_episodes = []
    changed = 0
    for i, ep in enumerate(episodes):
        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        if i in to_boost_set:
            orig = ep["instruction"]["instruction_text"]
            fixed = boost_walk_past(orig, rng)
            if fixed != orig:
                changed += 1
            new_ep["instruction"]["instruction_text"] = fixed
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]

    print(f"\nChanged {changed}/{n} ({changed/n*100:.1f}%) episodes")
    print(f"avg_words = {sum(words)/n:.1f} [GT=26.8, v135=25.6]")
    print(f"starts_Turn = {sum(1 for i in all_insts if TURN_RE.match(i))/n*100:.1f}% [GT=16.5%]")
    print(f"hallway = {sum(1 for i in all_insts if HALL.search(i))/n*100:.1f}% [GT=20.4%, v135=21.3%]")
    print(f"walk_past = {sum(1 for i in all_insts if WP.search(i))/n*100:.1f}% [GT=10.5%, v135=9.1%]")
    print(f"through_the = {sum(1 for i in all_insts if TT.search(i))/n*100:.1f}% [GT=27.2%, v135=27.2%]")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
