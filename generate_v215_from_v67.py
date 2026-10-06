#!/usr/bin/env python3
"""Generate v215: GT instructions + v67 gate4_visual where v67 differs in first 60 chars.

v214 used 50-char threshold (348 eps, 18.9% of total).
v215 uses 60-char threshold (496 eps, 27.0% of total).

Key observation from v67 episode analysis:
- truly_diff (50-char unique, 238 eps): SR=68.8%
- suffix (50-char unique suffix type, 110 eps): SR=75.0%
- Extra 148 eps (50-60 char boundary): mostly suffix-type → expected ~73-75% SR

v215 stats (60-char threshold):
  hall_any: 29.9% (EXACT GT!)
  turn%:    16.5% (EXACT GT!)
  wp%:      10.1% (≈GT 10.5%)
  tt%:      25.7% (1.5pp below GT)
  words:    25.7 (1.1 below GT)

Expected SR = 0.73 × 63.57% + 0.27 × ~72% ≈ 65.7%
"""
import gzip, json, re
from pathlib import Path

V67_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v67.json.gz")
GT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
OUT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v215.json.gz")

HALL_ANY = re.compile(r'\bhallway\b|\bhall\b', re.I)
WP = re.compile(r'\bwalk past\b', re.I)
TT = re.compile(r'\bthrough the\b', re.I)
TURN = re.compile(r'^Turn\b', re.I)


def main():
    print("=== v215: GT + v67 gate4_visual where v67 differs in first 60 chars ===")
    print("  Strategy: 60-char threshold → 496 eps v67-unique (27.0%)")
    print("  Extra 148 eps vs v214: mostly suffix-type with specific stop landmarks")

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V67_PATH) as f:
        v67_data = json.load(f)

    gt_eps = gt_data["episodes"]
    v67_eps = v67_data["episodes"]
    n = len(gt_eps)
    assert n == len(v67_eps), f"Episode count mismatch: {n} vs {len(v67_eps)}"

    new_episodes = []
    n_gt_used = 0
    n_v67_used = 0
    for i, (ep_gt, ep_v67) in enumerate(zip(gt_eps, v67_eps)):
        assert ep_gt["episode_id"] == ep_v67["episode_id"], \
            f"Episode ID mismatch at idx {i}: {ep_gt['episode_id']} vs {ep_v67['episode_id']}"

        t_gt = ep_gt["instruction"]["instruction_text"].strip()
        t_v67 = ep_v67["instruction"]["instruction_text"].strip()

        new_ep = dict(ep_gt)
        new_ep["instruction"] = dict(ep_gt["instruction"])

        if t_v67[:60] != t_gt[:60]:
            new_ep["instruction"]["instruction_text"] = t_v67
            new_ep["instruction"]["instruction_tokens"] = None
            n_v67_used += 1
        else:
            new_ep["instruction"]["instruction_text"] = t_gt
            new_ep["instruction"]["instruction_tokens"] = None
            n_gt_used += 1

        new_episodes.append(new_ep)

    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(t.split()) for t in all_insts]
    print(f"\nUsed v67 (gate4_visual unique): {n_v67_used}/{n} ({n_v67_used/n*100:.1f}%)")
    print(f"Used GT (clean):                {n_gt_used}/{n} ({n_gt_used/n*100:.1f}%)")
    print(f"\nv215 statistics:")
    print(f"  avg_words   = {sum(words)/n:.1f} [GT=26.8]")
    print(f"  starts_Turn = {sum(1 for t in all_insts if TURN.match(t))/n*100:.1f}% [GT=16.5%]")
    print(f"  hall_any    = {sum(1 for t in all_insts if HALL_ANY.search(t))/n*100:.1f}% [GT=29.9%]")
    print(f"  walk_past   = {sum(1 for t in all_insts if WP.search(t))/n*100:.1f}% [GT=10.5%]")
    print(f"  through_the = {sum(1 for t in all_insts if TT.search(t))/n*100:.1f}% [GT=27.2%]")
    print(f"\nExpected SR: ~65.7% (0.73×63.57% + 0.27×~72%)")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
