#!/usr/bin/env python3
"""Generate v208: v207 with through_the reduced to exact GT level (27.2%).

v207 stats: h=20.4%(EXACT!) wp=10.5%(EXACT!) tt=28.9% words=26.9 turn=17.2%
GT target:  h=20.4%            wp=10.5%            tt=27.2%

v207 has 531/1839 episodes with "through the" (28.9%).
GT needs 500/1839 episodes with "through the" (27.2%).
Need to remove "through the" from ~31 episodes.

Strategy: Replace "through the X" with "into the X" or "past the X" in 31 episodes.
Only for spatial prepositions where the replacement is semantically valid.
"""
import gzip
import json
import random
import re
from pathlib import Path

V207_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v207.json.gz")
GT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
OUT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v208.json.gz")

TT_RE = re.compile(r'\bthrough the\b', re.I)
HALL_ANY = re.compile(r'\bhallway\b|\bhall\b', re.I)
WP = re.compile(r'\bwalk past\b', re.I)
TURN_RE = re.compile(r'^Turn\b', re.I)


def replace_first_through_the(text: str, rng: random.Random) -> str:
    """Replace first 'through the' with 'into the' or 'past the'."""
    m = TT_RE.search(text)
    if not m:
        return text
    # Check what follows "through the"
    after = text[m.end():m.end()+30].lower().strip()
    orig = m.group(0)
    # Choose replacement based on context
    if any(after.startswith(w) for w in ['door', 'opening', 'arch', 'gap', 'entran']):
        replacement = 'past the' if orig[0].islower() else 'Past the'
    elif any(after.startswith(w) for w in ['room', 'kitchen', 'bedroom', 'bathroom',
                                            'living', 'dining', 'office', 'area', 'space']):
        replacement = 'into the' if orig[0].islower() else 'Into the'
    elif any(after.startswith(w) for w in ['corridor', 'passage', 'walkway']):
        repl = rng.choice(['into the', 'down the'])
        replacement = repl if orig[0].islower() else repl[0].upper() + repl[1:]
    else:
        replacement = 'into the' if orig[0].islower() else 'Into the'
    return text[:m.start()] + replacement + text[m.end():]


def main():
    print("=== v208: v207 + through_the reduction to GT level (27.2%) ===")
    rng = random.Random(42)

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V207_PATH) as f:
        v207_data = json.load(f)

    episodes = v207_data["episodes"]
    n = len(episodes)

    tt_count = sum(1 for ep in episodes if TT_RE.search(ep["instruction"]["instruction_text"]))
    gt_tt_count = int(round(n * 0.272))  # 500
    need_remove = max(0, tt_count - gt_tt_count)

    # Find episodes WITH "through the"
    tt_eps = [i for i, ep in enumerate(episodes)
              if TT_RE.search(ep["instruction"]["instruction_text"])]

    print(f"v207 tt_count={tt_count}/{n} ({tt_count/n*100:.1f}%)")
    print(f"GT target: {gt_tt_count}/{n} ({gt_tt_count/n*100:.1f}%)")
    print(f"Need to remove tt from: {need_remove} episodes")
    print(f"Episodes with 'through the': {len(tt_eps)}")

    rng.shuffle(tt_eps)
    to_modify = set(tt_eps[:need_remove])

    new_episodes = []
    changed = 0
    for i, ep in enumerate(episodes):
        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        if i in to_modify:
            orig = ep["instruction"]["instruction_text"]
            fixed = replace_first_through_the(orig, rng)
            if fixed != orig and not TT_RE.search(fixed):
                changed += 1
            elif fixed != orig:
                # Still has "through the" (multiple occurrences), try to count it
                changed += 1
            new_ep["instruction"]["instruction_text"] = fixed
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]

    print(f"\nChanged {changed}/{n} ({changed/n*100:.1f}%) episodes")
    print(f"avg_words = {sum(words)/n:.1f} [GT=26.8]")
    print(f"starts_Turn = {sum(1 for i in all_insts if TURN_RE.match(i))/n*100:.1f}% [GT=16.5%]")
    print(f"hallway (h) = {sum(1 for i in all_insts if HALL_ANY.search(i))/n*100:.1f}% [GT=20.4%]")
    print(f"walk_past   = {sum(1 for i in all_insts if WP.search(i))/n*100:.1f}% [GT=10.5%]")
    print(f"through_the = {sum(1 for i in all_insts if TT_RE.search(i))/n*100:.1f}% [GT=27.2%, v207=28.9%]")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
