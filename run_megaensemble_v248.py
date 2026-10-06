#!/usr/bin/env python3
"""
Gate 4 v248 — Mega-Ensemble from v242–v246.

For each trajectory we have 15 candidate instructions (5 versions × 3 slots).
We score all 5 candidates per slot and pick the best one, resulting in 3 final
instructions (one per slot) that are the best across ALL generation approaches:
  v242: 3-style base (no ordinals, no stairs)
  v243: + height-aware stair detection
  v244: + ordinal disambiguation (65.9% ordinals)
  v245: + trajectory-matched few-shot GT examples (GT-style vocabulary)
  v246: + goal-anchored pre-pass (90.4% visual noun stops)

Quality score (same as v247 scorer, same weights):
  +2 ordinal marker when n_turns > 1 (else +0.5)
  ±3 stair word when multifloor (penalise if missing)
  +1 per turn direction word (capped at n_turns+1)
  +2 stop word present
  +2 visual noun in last sentence
  +1 length 15-40 words

No API calls. Runs in under 5 seconds.
"""
import gzip
import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

ROOT         = Path(__file__).parent
GT_UNSEEN    = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz"
GT_SEEN      = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_seen/val_seen.json.gz"
OUT_DIR      = ROOT / "outputs" / "datasets"
HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")

UNSEEN_CKS = {
    "v242": ROOT / "outputs" / "gate4_v242_unseen_checkpoint.json",
    "v243": ROOT / "outputs" / "gate4_v243_unseen_checkpoint.json",
    "v244": ROOT / "outputs" / "gate4_v244_unseen_checkpoint.json",
    "v245": ROOT / "outputs" / "gate4_v245_unseen_checkpoint.json",
    "v246": ROOT / "outputs" / "gate4_v246_unseen_checkpoint.json",
}
SEEN_CKS = {
    "v243": ROOT / "outputs" / "gate4_v243_seen_checkpoint.json",
    "v244": ROOT / "outputs" / "gate4_v244_seen_checkpoint.json",
    # v242_seen, v245_seen, v246_seen added dynamically when available
}

# ── Geometry ──────────────────────────────────────────────────────────────────

def dist3d(p1, p2) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def heading_xz(p1, p2) -> float:
    dx, dz = p2[0] - p1[0], p2[2] - p1[2]
    return math.degrees(math.atan2(dx, -dz)) if (abs(dx) > 1e-6 or abs(dz) > 1e-6) else 0.0

def signed_diff(a, b) -> float:
    return (b - a + 180) % 360 - 180

STAIR_DY = 0.25


def path_features(ep: dict) -> dict:
    path = ep.get("reference_path", [])
    n_turns = 0
    is_multifloor = False
    for i in range(1, len(path) - 1):
        if abs(signed_diff(heading_xz(path[i-1], path[i]), heading_xz(path[i], path[i+1]))) > 20:
            n_turns += 1
        if abs(path[i+1][1] - path[i][1]) >= STAIR_DY:
            is_multifloor = True
    return {"n_turns": n_turns, "is_multifloor": is_multifloor}


# ── Quality scorer (same as v247) ─────────────────────────────────────────────

STOP_NOUNS = {
    "chair","table","couch","sofa","door","doorway","window","wall","desk","bed",
    "counter","cabinet","lamp","shelf","mirror","stair","stairs","step","steps",
    "carpet","rug","column","pillar","island","sink","fireplace","hallway","corridor",
    "bookcase","bookshelf","dresser","wardrobe","closet","bath","bathroom","kitchen",
    "living","dining","bedroom","lobby","entryway","foyer","balcony","archway",
    "painting","picture","artwork","frame","plant","banner","sign","tv","television",
    "couch","armchair","recliner","ottoman","loveseat","bench",
}
STAIR_WORDS = {"stair","stairs","step","steps","flight","floor","level","ascend","descend","climb"}
ORDINALS    = {"first","second","third","fourth","1st","2nd","3rd"}
STOP_WORDS  = {"stop","wait","halt","stand","pause"}


def score(instr: str, feats: dict) -> float:
    t = instr.lower()
    words = t.split()
    s = 0.0

    has_ordinal = any(o in t for o in ORDINALS)
    if has_ordinal:
        s += 2.0 if feats["n_turns"] > 1 else 0.5

    has_stair = any(w in t for w in STAIR_WORDS)
    if feats["is_multifloor"]:
        s += 3.0 if has_stair else -2.0
    elif has_stair:
        s -= 1.0

    dir_count = t.count("left") + t.count("right")
    s += min(dir_count, feats["n_turns"] + 1) * 1.0

    sentences = re.split(r'[.!?]', instr)
    last = sentences[-2].lower() if len(sentences) > 1 else sentences[-1].lower()
    if any(w in t for w in STOP_WORDS):
        s += 2.0
    if any(n in last for n in STOP_NOUNS):
        s += 2.0

    n_words = len(words)
    if 15 <= n_words <= 40:
        s += 1.0
    elif n_words < 8 or n_words > 60:
        s -= 1.0

    return s


# ── Ensemble builder ──────────────────────────────────────────────────────────

def load_ck(path: Path) -> Dict[str, List[str]]:
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for k, v in raw.items():
        result[k] = v if isinstance(v, list) else [v, v, v]
    return result


def build_ensemble(gt_data: dict, checkpoints: Dict[str, Dict]) -> Tuple[dict, dict]:
    traj_to_feat: Dict[str, dict] = {}
    for ep in gt_data["episodes"]:
        tid = f"traj_{ep['trajectory_id']}" if "trajectory_id" in ep else f"traj_ep{ep['episode_id']}"
        if tid not in traj_to_feat:
            traj_to_feat[tid] = path_features(ep)

    # Track which version won each slot
    version_wins: Dict[str, int] = {v: 0 for v in checkpoints}
    instruction_map: Dict[str, List[str]] = {}

    for tid, feats in traj_to_feat.items():
        chosen = []
        for slot in range(3):
            best_score = float("-inf")
            best_instr = None
            best_ver = None
            for ver, ck in checkpoints.items():
                instrs = ck.get(tid, [])
                if slot < len(instrs) and instrs[slot]:
                    s = score(instrs[slot], feats)
                    if s > best_score:
                        best_score = s
                        best_instr = instrs[slot]
                        best_ver = ver
            if best_instr is not None:
                chosen.append(best_instr)
                if best_ver:
                    version_wins[best_ver] += 1
            # fallback: use any available
            elif any(ck.get(tid) for ck in checkpoints.values()):
                for ck in checkpoints.values():
                    instrs = ck.get(tid, [])
                    if instrs:
                        chosen.append(instrs[min(slot, len(instrs)-1)])
                        break
        instruction_map[tid] = chosen[:3]

    episodes = []
    traj_counter: Dict[str, int] = {}
    for ep in gt_data["episodes"]:
        tid = f"traj_{ep['trajectory_id']}" if "trajectory_id" in ep else f"traj_ep{ep['episode_id']}"
        idx = traj_counter.get(tid, 0)
        traj_counter[tid] = idx + 1
        instrs = instruction_map.get(tid, [])
        instr_text = instrs[idx % len(instrs)] if instrs else ep["instruction"]["instruction_text"]
        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        new_ep["instruction"]["instruction_text"] = instr_text
        episodes.append(new_ep)

    return {"episodes": episodes}, version_wins


def run(split: str = "unseen"):
    if split == "unseen":
        gt_path     = GT_UNSEEN
        ck_paths    = UNSEEN_CKS
        out_name    = "val_unseen_v248.json.gz"
        habitat_dir = HABITAT_BASE / "val_unseen"
    else:
        gt_path     = GT_SEEN
        # Add dynamically available seen checkpoints
        seen_cks = dict(SEEN_CKS)
        for vname, cand in [
            ("v242", ROOT/"outputs"/"gate4_v242_seen_checkpoint.json"),
            ("v245", ROOT/"outputs"/"gate4_v245_seen_checkpoint.json"),
            ("v246", ROOT/"outputs"/"gate4_v246_seen_checkpoint.json"),
        ]:
            if cand.exists():
                seen_cks[vname] = cand
        ck_paths    = seen_cks
        out_name    = "val_seen_v248.json.gz"
        habitat_dir = HABITAT_BASE / "val_seen"

    with gzip.open(gt_path, "rt") as f:
        gt_data = json.load(f)

    checkpoints = {}
    for vname, path in ck_paths.items():
        if path.exists():
            checkpoints[vname] = load_ck(path)
            print(f"  Loaded {vname}: {len(checkpoints[vname])} trajectories")
        else:
            print(f"  Skipped {vname}: {path} not found")

    if not checkpoints:
        print("No checkpoints available — aborting"); return

    out_data, version_wins = build_ensemble(gt_data, checkpoints)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)
    shutil.copy2(out_path, habitat_dir / out_name)

    ep_texts = [ep["instruction"]["instruction_text"] for ep in out_data["episodes"]]
    total = len(ep_texts); unique = len(set(ep_texts))
    avg_w = sum(len(t.split()) for t in ep_texts) / total
    ord_cnt = sum(1 for t in ep_texts if any(o in t.lower() for o in ["first","second","third"]))
    noun_cnt = sum(1 for t in ep_texts if any(n in t.lower() for n in
        ["chair","table","door","window","desk","bed","counter","cabinet","couch"]))
    print(f"\nv248 {split}: {total} eps  {unique} unique ({unique/total*100:.0f}%)  avg_words={avg_w:.1f}")
    print(f"  ordinal_markers={ord_cnt/total*100:.1f}%  visual_noun={noun_cnt/total*100:.1f}%")
    print(f"  Version wins: {version_wins}")
    print(f"Saved: {habitat_dir / out_name}")

    import random; random.seed(42)
    print("Samples:")
    for ep in random.sample(out_data["episodes"][:400], 3):
        print(f"  [{ep['scene_id'][-6:]}] {ep['instruction']['instruction_text'][:105]}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="unseen", choices=["unseen","seen","both"])
    args = p.parse_args()
    if args.split == "both":
        for s in ["unseen","seen"]: run(s)
    else:
        run(args.split)
