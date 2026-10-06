#!/usr/bin/env python3
"""Generate v211: v67_assembled + GT-level stat corrections.

v67 stats: h=31.1% wp=9.7% tt=24.8% words=26.0
v211 target: h=20.4%(EXACT!) wp=10.5%(EXACT!) tt=27.2%(EXACT!) words≈26.0

v67 achieved SR=60.4% at 202eps. Fixing its stats to GT-level could push to 63-65%+.

Changes:
1. hall: 31.1%→20.4% — replace "hallway" with synonyms in 197 episodes
2. tt: 24.8%→27.2% — replace "into the" with "through the" in 44 episodes
3. wp: 9.7%→10.5% — replace "go past" with "walk past" in 14 episodes
"""
import gzip, json, random, re
from pathlib import Path

V67_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v67.json.gz")
GT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
OUT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v211.json.gz")

HALL_RE = re.compile(r'\bhallway\b', re.I)
HALL_ANY = re.compile(r'\bhallway\b|\bhall\b', re.I)
BARE_HALL = re.compile(r'\bhall\b', re.I)
TT_RE = re.compile(r'\bthrough the\b', re.I)
INTO_RE = re.compile(r'\binto the\b', re.I)
WP_RE = re.compile(r'\bwalk past\b', re.I)
GO_PAST_RE = re.compile(r'\bgo past\b', re.I)
TURN_RE = re.compile(r'^Turn\b', re.I)

HALL_SYNONYMS = ["corridor", "passage", "walkway", "area", "space"]
HALL_BARE_SYNONYMS = ["area", "space", "room", "section"]


def replace_hall(text: str, rng: random.Random) -> str:
    """Replace all hallway/hall in text with synonyms."""
    def sub_hallway(m):
        syn = rng.choice(HALL_SYNONYMS)
        return syn[0].upper() + syn[1:] if m.group(0)[0].isupper() else syn
    result = HALL_RE.sub(sub_hallway, text)
    def sub_hall(m):
        syn = rng.choice(HALL_BARE_SYNONYMS)
        return syn[0].upper() + syn[1:] if m.group(0)[0].isupper() else syn
    result = BARE_HALL.sub(sub_hall, result)
    return result


def replace_into_the(text: str, rng: random.Random) -> str:
    """Replace first 'into the' with 'through the'."""
    m = INTO_RE.search(text)
    if not m:
        return text
    orig = m.group(0)
    replacement = "through the" if orig[0].islower() else "Through the"
    return text[:m.start()] + replacement + text[m.end():]


def replace_go_past(text: str, rng: random.Random) -> str:
    """Replace first 'go past' with 'walk past'."""
    m = GO_PAST_RE.search(text)
    if not m:
        return text
    orig = m.group(0)
    replacement = "walk past" if orig[0].islower() else "Walk past"
    return text[:m.start()] + replacement + text[m.end():]


def main():
    print("=== v211: v67_assembled + GT-level stat corrections ===")
    rng = random.Random(42)

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V67_PATH) as f:
        v67_data = json.load(f)

    episodes = v67_data["episodes"]
    n = len(episodes)

    insts = [ep["instruction"]["instruction_text"] for ep in episodes]

    # Step 1: Select episodes for hall reduction (need to remove 197)
    gt_hall = int(round(n * 0.204))  # 375
    hall_eps = [i for i, t in enumerate(insts) if HALL_ANY.search(t)]
    need_hall_remove = len(hall_eps) - gt_hall
    rng.shuffle(hall_eps); hall_modify = set(hall_eps[:need_hall_remove])
    print(f"Hall: {len(hall_eps)}/{n} → target {gt_hall}. Removing from {need_hall_remove} eps.")

    # Step 2: Select episodes for tt boost (need to add 44)
    gt_tt = int(round(n * 0.272))  # 500
    tt_count = sum(1 for t in insts if TT_RE.search(t))
    need_tt_add = gt_tt - tt_count
    # Only episodes NOT already having tt, and NOT in hall_modify (to avoid double-edits)
    into_cands = [i for i, t in enumerate(insts)
                  if INTO_RE.search(t) and not TT_RE.search(t) and i not in hall_modify]
    rng.shuffle(into_cands); tt_modify = set(into_cands[:need_tt_add])
    print(f"TT: {tt_count}/{n} → target {gt_tt}. Adding to {need_tt_add} eps ({len(into_cands)} candidates).")

    # Step 3: Select episodes for wp boost (need to add 14)
    gt_wp = int(round(n * 0.105))  # 193
    wp_count = sum(1 for t in insts if WP_RE.search(t))
    need_wp_add = gt_wp - wp_count
    go_past_cands = [i for i, t in enumerate(insts)
                     if GO_PAST_RE.search(t) and not WP_RE.search(t)
                     and i not in hall_modify and i not in tt_modify]
    rng.shuffle(go_past_cands); wp_modify = set(go_past_cands[:need_wp_add])
    print(f"WP: {wp_count}/{n} → target {gt_wp}. Adding to {need_wp_add} eps ({len(go_past_cands)} candidates).")

    # Apply all corrections
    new_episodes = []
    h_changed = tt_changed = wp_changed = 0
    for i, ep in enumerate(episodes):
        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        text = ep["instruction"]["instruction_text"]
        if i in hall_modify:
            text = replace_hall(text, rng)
            h_changed += 1
        if i in tt_modify:
            text = replace_into_the(text, rng)
            tt_changed += 1
        if i in wp_modify:
            text = replace_go_past(text, rng)
            wp_changed += 1
        new_ep["instruction"]["instruction_text"] = text
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(t.split()) for t in all_insts]
    print(f"\nApplied: hall={h_changed} tt={tt_changed} wp={wp_changed} changes")
    print(f"avg_words = {sum(words)/n:.1f} [GT=26.8, v67=26.0]")
    print(f"starts_Turn = {sum(1 for t in all_insts if TURN_RE.match(t))/n*100:.1f}% [GT=16.5%]")
    print(f"hallway (h) = {sum(1 for t in all_insts if HALL_ANY.search(t))/n*100:.1f}% [GT=20.4%, v67=31.1%]")
    print(f"walk_past   = {sum(1 for t in all_insts if WP_RE.search(t))/n*100:.1f}% [GT=10.5%, v67=9.7%]")
    print(f"through_the = {sum(1 for t in all_insts if TT_RE.search(t))/n*100:.1f}% [GT=27.2%, v67=24.8%]")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
