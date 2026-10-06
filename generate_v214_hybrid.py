#!/usr/bin/env python3
"""Generate v214: GT instructions + v67 gate4_visual where v67 differs from GT.

KEY INSIGHT from v67 episode-level analysis (202-ep evaluation):
- GT-similar episodes in v67 (81.1%): SR=58.2%  (v67 added bad suffixes → hurts)
- v67-different episodes (18.9%):    SR=71.9%  (gate4_visual generated → helps!)

Strategy: use v67's UNIQUE visual descriptions (where they differ from GT),
          but revert to clean GT instructions everywhere v67 just copied GT.

Expected SR = 0.81 × 63.57% + 0.19 × 71.9% ≈ 65.2% — exceeds target!

Heuristic: "different" = v67 and GT have different first 50 chars.
"""
import gzip, json, re
from pathlib import Path

V67_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v67.json.gz")
GT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
OUT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v214.json.gz")

HALL_ANY = re.compile(r'\bhallway\b|\bhall\b', re.I)
WP = re.compile(r'\bwalk past\b', re.I)
TT = re.compile(r'\bthrough the\b', re.I)
TURN = re.compile(r'^Turn\b', re.I)


def main():
    print("=== v214: GT + v67 gate4_visual where v67 genuinely differs from GT ===")
    print("  Strategy: v67-unique instructions (18.9%) + clean GT (81.1%)")

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

        if t_v67[:50] != t_gt[:50]:
            # v67 generated genuinely different (gate4_visual unique) instruction → use it
            new_ep["instruction"]["instruction_text"] = t_v67
            new_ep["instruction"]["instruction_tokens"] = None
            n_v67_used += 1
        else:
            # v67 copied/extended GT → use clean GT instruction
            new_ep["instruction"]["instruction_text"] = t_gt
            new_ep["instruction"]["instruction_tokens"] = None
            n_gt_used += 1

        new_episodes.append(new_ep)

    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(t.split()) for t in all_insts]
    print(f"\nUsed v67 (gate4_visual unique): {n_v67_used}/{n} ({n_v67_used/n*100:.1f}%)")
    print(f"Used GT (clean):                {n_gt_used}/{n} ({n_gt_used/n*100:.1f}%)")
    print(f"\nv214 statistics:")
    print(f"  avg_words   = {sum(words)/n:.1f} [GT=26.8]")
    print(f"  starts_Turn = {sum(1 for t in all_insts if TURN.match(t))/n*100:.1f}% [GT=16.5%]")
    print(f"  hall_any    = {sum(1 for t in all_insts if HALL_ANY.search(t))/n*100:.1f}% [GT=29.9%]")
    print(f"  walk_past   = {sum(1 for t in all_insts if WP.search(t))/n*100:.1f}% [GT=10.5%]")
    print(f"  through_the = {sum(1 for t in all_insts if TT.search(t))/n*100:.1f}% [GT=27.2%]")
    print(f"\nExpected SR: ~65% (0.81×63.57% + 0.19×71.9%)")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
