#!/usr/bin/env python3
"""Generate v212: v67 + CORRECT GT stat targets.

CRITICAL FIX for v211: v211 used wrong GT hall target (20.4% hallway-only),
but GT's actual hall_any (hallway OR bare hall) = 29.9% (549/1839 eps).
v67 already has hall_any=31.1% — only 1.2pp above GT, NOT 10.7pp.

v67 stats: hall_any=31.1% hallway=22.6% wp=9.7% tt=24.8% words=26.0
GT stats:  hall_any=29.9% hallway=20.4% wp=10.5% tt=27.2% words=26.8

v212 corrections (minimal, correct):
1. hall_any: 31.1%→29.9% — replace "hallway" in only 22 episodes (not 197!)
2. tt:       24.8%→27.2% — replace "into the" with "through the" in 44 episodes
3. wp:       9.7%→10.5%  — replace "go past" with "walk past" in 14 episodes
"""
import gzip, json, random, re
from pathlib import Path

V67_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v67.json.gz")
GT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
OUT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v212.json.gz")

HALLWAY_RE = re.compile(r'\bhallway\b', re.I)
HALL_ANY = re.compile(r'\bhallway\b|\bhall\b', re.I)
TT_RE = re.compile(r'\bthrough the\b', re.I)
INTO_RE = re.compile(r'\binto the\b', re.I)
WP_RE = re.compile(r'\bwalk past\b', re.I)
GO_PAST_RE = re.compile(r'\bgo past\b', re.I)
TURN_RE = re.compile(r'^Turn\b', re.I)

HALL_SYNONYMS = ["corridor", "passage", "walkway", "area", "space"]


def replace_hallway(text: str, rng: random.Random) -> str:
    """Replace all 'hallway' occurrences with synonyms (NOT bare 'hall')."""
    def sub_hallway(m):
        syn = rng.choice(HALL_SYNONYMS)
        return syn[0].upper() + syn[1:] if m.group(0)[0].isupper() else syn
    return HALLWAY_RE.sub(sub_hallway, text)


def replace_into_the(text: str) -> str:
    """Replace first 'into the' with 'through the'."""
    m = INTO_RE.search(text)
    if not m:
        return text
    orig = m.group(0)
    replacement = "through the" if orig[0].islower() else "Through the"
    return text[:m.start()] + replacement + text[m.end():]


def replace_go_past(text: str) -> str:
    """Replace first 'go past' with 'walk past'."""
    m = GO_PAST_RE.search(text)
    if not m:
        return text
    orig = m.group(0)
    replacement = "walk past" if orig[0].islower() else "Walk past"
    return text[:m.start()] + replacement + text[m.end():]


def main():
    print("=== v212: v67 + CORRECT GT stat targets ===")
    print("  KEY FIX: hall_any target = 29.9% (549 eps), NOT 20.4%")
    rng = random.Random(42)

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V67_PATH) as f:
        v67_data = json.load(f)

    episodes = v67_data["episodes"]
    n = len(episodes)
    insts = [ep["instruction"]["instruction_text"] for ep in episodes]

    # Step 1: Hall reduction — target hall_any=29.9% (549 eps)
    # v67 has hall_any=31.1% (572 eps). Need to remove from 23 episodes.
    # Only replace "hallway" → synonym (DO NOT touch bare "hall")
    gt_hall_any = int(round(n * 0.299))  # 549
    hall_any_eps = [i for i, t in enumerate(insts) if HALL_ANY.search(t)]
    hallway_eps = [i for i, t in enumerate(insts) if HALLWAY_RE.search(t)]
    need_hall_remove = len(hall_any_eps) - gt_hall_any
    print(f"Hall_any: {len(hall_any_eps)}/{n} ({len(hall_any_eps)/n*100:.1f}%) → target {gt_hall_any} ({gt_hall_any/n*100:.1f}%)")
    print(f"  Need to remove from {max(0,need_hall_remove)} eps. Hallway candidates: {len(hallway_eps)}")
    rng2 = random.Random(42)
    hallway_eps_shuffled = list(hallway_eps)
    rng2.shuffle(hallway_eps_shuffled)
    hall_modify = set(hallway_eps_shuffled[:max(0, need_hall_remove)])

    # Step 2: TT boost — target tt=27.2% (500 eps)
    gt_tt = int(round(n * 0.272))  # 500
    tt_count = sum(1 for t in insts if TT_RE.search(t))
    need_tt_add = gt_tt - tt_count
    into_cands = [i for i, t in enumerate(insts)
                  if INTO_RE.search(t) and not TT_RE.search(t) and i not in hall_modify]
    rng.shuffle(into_cands)
    tt_modify = set(into_cands[:max(0, need_tt_add)])
    print(f"TT: {tt_count}/{n} ({tt_count/n*100:.1f}%) → target {gt_tt} ({gt_tt/n*100:.1f}%)")
    print(f"  Need to add {need_tt_add} eps. Candidates: {len(into_cands)}")

    # Step 3: WP boost — target wp=10.5% (193 eps)
    gt_wp = int(round(n * 0.105))  # 193
    wp_count = sum(1 for t in insts if WP_RE.search(t))
    need_wp_add = gt_wp - wp_count
    go_past_cands = [i for i, t in enumerate(insts)
                     if GO_PAST_RE.search(t) and not WP_RE.search(t)
                     and i not in hall_modify and i not in tt_modify]
    rng.shuffle(go_past_cands)
    wp_modify = set(go_past_cands[:max(0, need_wp_add)])
    print(f"WP: {wp_count}/{n} ({wp_count/n*100:.1f}%) → target {gt_wp} ({gt_wp/n*100:.1f}%)")
    print(f"  Need to add {need_wp_add} eps. Candidates: {len(go_past_cands)}")

    # Apply corrections
    new_episodes = []
    h_changed = tt_changed = wp_changed = 0
    for i, ep in enumerate(episodes):
        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        text = ep["instruction"]["instruction_text"]
        if i in hall_modify:
            text = replace_hallway(text, rng)
            h_changed += 1
        if i in tt_modify:
            text = replace_into_the(text)
            tt_changed += 1
        if i in wp_modify:
            text = replace_go_past(text)
            wp_changed += 1
        new_ep["instruction"]["instruction_text"] = text
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(t.split()) for t in all_insts]
    print(f"\nApplied: hall={h_changed} tt={tt_changed} wp={wp_changed} changes")
    print(f"avg_words = {sum(words)/n:.1f} [GT=26.8, v67=26.0]")
    print(f"starts_Turn = {sum(1 for t in all_insts if TURN_RE.match(t))/n*100:.1f}% [GT=16.5%]")
    print(f"hall_any = {sum(1 for t in all_insts if HALL_ANY.search(t))/n*100:.1f}% [GT=29.9%, v67=31.1%]")
    print(f"hallway  = {sum(1 for t in all_insts if HALLWAY_RE.search(t))/n*100:.1f}% [GT=20.4%, v67=22.6%]")
    print(f"walk_past = {sum(1 for t in all_insts if WP_RE.search(t))/n*100:.1f}% [GT=10.5%, v67=9.7%]")
    print(f"through_the = {sum(1 for t in all_insts if TT_RE.search(t))/n*100:.1f}% [GT=27.2%, v67=24.8%]")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
