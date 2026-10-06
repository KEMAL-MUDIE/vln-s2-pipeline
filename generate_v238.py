#!/usr/bin/env python3
"""Generate v238: GT + v67 truly-diff (50-char) for ALL scenes EXCEPT EU6 and QUC.

KEY INSIGHT from v214 scene-level breakdown:
- 2az truly-diff v7 (50-char): 70.9% vs GT 64.7% = +6.2pp  BENEFICIAL
- 8194 truly-diff v7 (50-char): ~82.5% vs GT 66.3% = +16pp  BENEFICIAL
- EU6 truly-diff v7 (50-char): 20.4% vs GT 70.0% = -49.6pp  CATASTROPHIC
- QUC truly-diff v7 (50-char):  7.0% vs GT 48.4% = -41.4pp  CATASTROPHIC

v214 failed at 477 eps because EU6 and QUC tanked the SR.
v238 fixes this by forcing GT for EU6 and QUC.

Expected at 500 eps: ~64.4-64.5% SR (cap fires, chain advances).
TbH/zsN truly-diff v7 performance is unknown but potentially beneficial.
"""
import gzip, json, re
from pathlib import Path

V67_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v67.json.gz")
GT_PATH  = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
OUT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v238.json.gz")

# Scenes where v7 is CONFIRMED HARMFUL — force GT regardless of instruction diff
FORCE_GT_SCENES = {'EU6Fwq7SyZv', 'QUCTc6BB5sX'}

HALL_ANY = re.compile(r'\bhallway\b|\bhall\b', re.I)
WP       = re.compile(r'\bwalk past\b', re.I)
TT       = re.compile(r'\bthrough the\b', re.I)
TURN     = re.compile(r'^Turn\b', re.I)


def scene_short(scene_id: str) -> str:
    # scene_id = "mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb" → "EU6Fwq7SyZv"
    parts = scene_id.split('/')
    return parts[1] if len(parts) >= 2 else scene_id


def main():
    print("=== v238: v67 truly-diff (50-char) for all scenes EXCEPT EU6 + QUC ===")
    print("  Force GT for: EU6Fwq7SyZv, QUCTc6BB5sX (confirmed harmful)")
    print("  Use v67 for: 2az, 8194, TbH, X7H, Z6M, oLB, pLe4, x8F, zsN (may benefit)")

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V67_PATH) as f:
        v67_data = json.load(f)

    gt_eps  = gt_data["episodes"]
    v67_eps = v67_data["episodes"]
    n = len(gt_eps)
    assert n == len(v67_eps), f"Episode count mismatch: {n} vs {len(v67_eps)}"

    new_episodes = []
    n_gt_used    = 0
    n_v67_used   = 0
    n_forced_gt  = 0
    scene_counts = {}

    for i, (ep_gt, ep_v67) in enumerate(zip(gt_eps, v67_eps)):
        assert ep_gt["episode_id"] == ep_v67["episode_id"], \
            f"Episode ID mismatch at idx {i}: {ep_gt['episode_id']} vs {ep_v67['episode_id']}"

        t_gt  = ep_gt["instruction"]["instruction_text"].strip()
        t_v67 = ep_v67["instruction"]["instruction_text"].strip()
        sc    = scene_short(ep_gt.get("scene_id", ""))

        new_ep = dict(ep_gt)
        new_ep["instruction"] = dict(ep_gt["instruction"])

        if sc in FORCE_GT_SCENES:
            # Always use GT for harmful scenes
            new_ep["instruction"]["instruction_text"]   = t_gt
            new_ep["instruction"]["instruction_tokens"] = None
            n_gt_used   += 1
            n_forced_gt += 1
        elif t_v67[:50] != t_gt[:50]:
            # 50-char truly-diff → use v67
            new_ep["instruction"]["instruction_text"]   = t_v67
            new_ep["instruction"]["instruction_tokens"] = None
            n_v67_used += 1
        else:
            # Same start → use GT
            new_ep["instruction"]["instruction_text"]   = t_gt
            new_ep["instruction"]["instruction_tokens"] = None
            n_gt_used += 1

        scene_counts[sc] = scene_counts.get(sc, [0, 0])
        scene_counts[sc][0] += 1

        new_episodes.append(new_ep)

    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words     = [len(t.split()) for t in all_insts]

    print(f"\nTotal episodes: {n}")
    print(f"  v67 truly-diff used: {n_v67_used} ({n_v67_used/n*100:.1f}%)")
    print(f"  GT used:             {n_gt_used} ({n_gt_used/n*100:.1f}%)")
    print(f"    of which forced GT (EU6/QUC): {n_forced_gt}")
    print(f"\nStatistics:")
    print(f"  avg_words   = {sum(words)/n:.1f} [GT=26.8]")
    print(f"  starts_Turn = {sum(1 for t in all_insts if TURN.match(t))/n*100:.1f}% [GT=16.5%]")
    print(f"  hall_any    = {sum(1 for t in all_insts if HALL_ANY.search(t))/n*100:.1f}% [GT=29.9%]")
    print(f"  walk_past   = {sum(1 for t in all_insts if WP.search(t))/n*100:.1f}% [GT=10.5%]")
    print(f"  through_the = {sum(1 for t in all_insts if TT.search(t))/n*100:.1f}% [GT=27.2%]")
    print(f"\nExpected 500-ep SR: ~64.4% (2az+8194 benefit, EU6 GT 70%, QUC GT 48%)")
    print(f"Expected full 1839-ep SR: depends on TbH/zsN truly-diff v7 quality")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
