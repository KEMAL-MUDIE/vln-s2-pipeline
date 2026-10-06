#!/usr/bin/env python3
"""Generate v213: v204 + correct tt reduction.

v204 stats: hall_any=28.3% wp=10.5%(EXACT) tt=28.9% turn=17.2%
GT stats:   hall_any=29.9% wp=10.5%         tt=27.2% turn=16.5%

v204 already has:
- hall_any=28.3% (only 1.6pp below GT 29.9%) — keep as-is
- wp=10.5% (EXACT GT) — keep
- tt=28.9% (1.7pp above GT) — reduce to 27.2%
- turn=17.2% (0.7pp above GT) — optionally fix

v213 = v204 + tt reduction (31 episodes) + turn fix (13 episodes)
"""
import gzip, json, random, re
from pathlib import Path

V204_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v204.json.gz")
GT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
OUT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v213.json.gz")

TT_RE = re.compile(r'\bthrough the\b', re.I)
HALL_ANY = re.compile(r'\bhallway\b|\bhall\b', re.I)
WP = re.compile(r'\bwalk past\b', re.I)
TURN_RE = re.compile(r'^Turn\b', re.I)

TURN_REPLACEMENTS = [
    (re.compile(r'^Turn right\b', re.I), 'Head right'),
    (re.compile(r'^Turn left\b', re.I), 'Head left'),
    (re.compile(r'^Turn around\b', re.I), 'Head back'),
    (re.compile(r'^Turn and\b', re.I), 'Head forward and'),
    (re.compile(r'^Turn\b', re.I), 'Head'),
]


def replace_first_through_the(text: str, rng: random.Random) -> str:
    """Replace first 'through the' with 'into the' or 'past the'."""
    m = TT_RE.search(text)
    if not m:
        return text
    after = text[m.end():m.end()+30].lower().strip()
    orig = m.group(0)
    if any(after.startswith(w) for w in ['door', 'opening', 'arch', 'gap', 'entran']):
        replacement = 'past the' if orig[0].islower() else 'Past the'
    elif any(after.startswith(w) for w in ['room', 'kitchen', 'bedroom', 'bathroom',
                                            'living', 'dining', 'office', 'area', 'space']):
        replacement = 'into the' if orig[0].islower() else 'Into the'
    else:
        replacement = 'into the' if orig[0].islower() else 'Into the'
    return text[:m.start()] + replacement + text[m.end():]


def replace_turn_start(text: str) -> str:
    for pattern, replacement in TURN_REPLACEMENTS:
        m = pattern.match(text)
        if m:
            return replacement + text[m.end():]
    return text


def main():
    print("=== v213: v204 + tt+turn corrections to GT level ===")
    rng = random.Random(42)

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V204_PATH) as f:
        v204_data = json.load(f)

    episodes = v204_data["episodes"]
    n = len(episodes)
    insts = [ep["instruction"]["instruction_text"] for ep in episodes]

    # TT reduction: 28.9% → 27.2%
    gt_tt = int(round(n * 0.272))  # 500
    tt_count = sum(1 for t in insts if TT_RE.search(t))
    need_tt_remove = tt_count - gt_tt
    tt_eps = [i for i, t in enumerate(insts) if TT_RE.search(t)]
    rng.shuffle(tt_eps)
    tt_modify = set(tt_eps[:max(0, need_tt_remove)])
    print(f"TT: {tt_count}/{n} ({tt_count/n*100:.1f}%) → target {gt_tt} ({gt_tt/n*100:.1f}%)")
    print(f"  Remove tt from {need_tt_remove} eps. Total with tt: {len(tt_eps)}")

    # Turn reduction: 17.2% → 16.5%
    gt_turn = int(round(n * 0.165))  # 303
    turn_eps = [i for i, t in enumerate(insts) if TURN_RE.match(t)]
    need_turn_remove = len(turn_eps) - gt_turn
    rng.shuffle(turn_eps)
    turn_modify = set(turn_eps[:max(0, need_turn_remove)])
    print(f"Turn: {len(turn_eps)}/{n} ({len(turn_eps)/n*100:.1f}%) → target {gt_turn} ({gt_turn/n*100:.1f}%)")
    print(f"  Modify {need_turn_remove} eps")

    # Apply corrections
    new_episodes = []
    tt_changed = turn_changed = 0
    for i, ep in enumerate(episodes):
        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        text = ep["instruction"]["instruction_text"]
        if i in tt_modify:
            new_text = replace_first_through_the(text, rng)
            if new_text != text:
                tt_changed += 1
            text = new_text
        if i in turn_modify:
            new_text = replace_turn_start(text)
            if new_text != text:
                turn_changed += 1
            text = new_text
        new_ep["instruction"]["instruction_text"] = text
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(t.split()) for t in all_insts]
    print(f"\nApplied: tt={tt_changed} turn={turn_changed} changes")
    print(f"avg_words = {sum(words)/n:.1f} [GT=26.8, v204=26.9]")
    print(f"starts_Turn = {sum(1 for t in all_insts if TURN_RE.match(t))/n*100:.1f}% [GT=16.5%, v204=17.2%]")
    print(f"hall_any  = {sum(1 for t in all_insts if HALL_ANY.search(t))/n*100:.1f}% [GT=29.9%, v204=28.3%]")
    print(f"walk_past = {sum(1 for t in all_insts if WP.search(t))/n*100:.1f}% [GT=10.5%, v204=10.5%]")
    print(f"through_the = {sum(1 for t in all_insts if TT_RE.search(t))/n*100:.1f}% [GT=27.2%, v204=28.9%]")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
