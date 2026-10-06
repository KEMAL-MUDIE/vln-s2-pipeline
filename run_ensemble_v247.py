#!/usr/bin/env python3
"""
Gate 4 v247 — Quality-Scored Ensemble of v244 and v246.

For each trajectory, we have 3 candidate instructions from v244 (strong ordinals,
65.9% ordinal markers) and 3 from v246 (goal-anchored, 90.4% specific nouns, 21%
ordinals). We score each candidate by how many navigation-relevant features it
covers, and pick the better instruction per slot (A/B/C).

Score function (higher = better):
  +1 per turn direction word (left/right) in expected positions
  +2 if ordinal marker present (first/second/third) when n_turns > 1
  +3 if stair/step/floor mentioned when trajectory is multi-floor
  +2 if stop condition has specific visual noun (chair, table, door, etc.)
  +1 if stop condition word present (stop/wait/halt)
  +1 if length 15-40 words (appropriate for the navigation model)
  Tiebreak: prefer v246 (newer, better goal anchoring)

No API calls required. Runs in <10 seconds.
"""
import gzip
import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent
GT_UNSEEN    = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz"
GT_SEEN      = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_seen/val_seen.json.gz"
CK244_U      = ROOT / "outputs" / "gate4_v244_unseen_checkpoint.json"
CK244_S      = ROOT / "outputs" / "gate4_v244_seen_checkpoint.json"
CK246_U      = ROOT / "outputs" / "gate4_v246_unseen_checkpoint.json"
CK246_S      = ROOT / "outputs" / "gate4_v246_seen_checkpoint.json"
OUT_DIR      = ROOT / "outputs" / "datasets"
HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")

# ── Geometry helpers (minimal) ─────────────────────────────────────────────────

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


# ── Quality scorer ─────────────────────────────────────────────────────────────

STOP_NOUNS = {
    "chair","table","couch","sofa","door","doorway","window","wall","desk","bed",
    "counter","cabinet","lamp","shelf","mirror","stair","stairs","step","steps",
    "carpet","rug","column","pillar","island","sink","fireplace","hallway","corridor",
    "bookcase","bookshelf","dresser","wardrobe","closet","bath","bathroom","kitchen",
    "living","dining","bedroom","lobby","entryway","foyer","balcony","archway",
    "painting","picture","artwork","frame","plant","tree","banner","sign","tv",
    "television","couch","armchair","recliner","ottoman","loveseat","bench","pew",
}

STAIR_WORDS = {"stair","stairs","step","steps","flight","floor","level","ascend","descend","climb"}
ORDINALS    = {"first","second","third","fourth","1st","2nd","3rd"}
STOP_WORDS  = {"stop","wait","halt","stand","pause"}


def score(instr: str, feats: dict) -> float:
    t = instr.lower()
    words = t.split()
    s = 0.0

    # Ordinal markers — more valuable when path has multiple turns
    has_ordinal = any(o in t for o in ORDINALS)
    if has_ordinal:
        s += 2.0 if feats["n_turns"] > 1 else 0.5

    # Stair mention — critical for multi-floor paths
    has_stair = any(w in t for w in STAIR_WORDS)
    if feats["is_multifloor"]:
        s += 3.0 if has_stair else -2.0  # penalise missing stair mention
    elif has_stair:
        s -= 1.0  # hallucinated stair

    # Turn direction coverage (crude: count left/right)
    dir_count = t.count("left") + t.count("right")
    s += min(dir_count, feats["n_turns"] + 1) * 1.0

    # Stop condition quality
    # Find last sentence
    sentences = re.split(r'[.!?]', instr)
    last = sentences[-2].lower() if len(sentences) > 1 else sentences[-1].lower()
    has_stop_word = any(w in t for w in STOP_WORDS)
    has_noun_stop = any(n in last for n in STOP_NOUNS)
    s += 2.0 if has_stop_word else 0.0
    s += 2.0 if has_noun_stop else 0.0  # specific visual goal anchor

    # Length appropriateness
    n_words = len(words)
    if 15 <= n_words <= 40:
        s += 1.0
    elif n_words < 8 or n_words > 60:
        s -= 1.0

    return s


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_ensemble(gt_data: dict, ck244: Dict, ck246: Dict) -> Tuple[dict, dict]:
    traj_to_feat: Dict[str, dict] = {}
    traj_to_first_ep: Dict[str, dict] = {}
    for ep in gt_data["episodes"]:
        tid = f"traj_{ep['trajectory_id']}" if "trajectory_id" in ep else f"traj_ep{ep['episode_id']}"
        if tid not in traj_to_feat:
            traj_to_feat[tid] = path_features(ep)
            traj_to_first_ep[tid] = ep

    # Build instruction map: for each trajectory, pick best-per-slot from v244 vs v246
    instruction_map: Dict[str, List[str]] = {}
    stats = {"v244_wins": 0, "v246_wins": 0, "tied_v246": 0}

    for tid, feats in traj_to_feat.items():
        instrs244 = ck244.get(tid, [])
        instrs246 = ck246.get(tid, [])
        if not instrs244 and not instrs246:
            continue

        chosen = []
        n_slots = max(len(instrs244), len(instrs246), 3)
        for slot in range(n_slots):
            c244 = instrs244[slot] if slot < len(instrs244) else None
            c246 = instrs246[slot] if slot < len(instrs246) else None

            if c244 is None:
                chosen.append(c246); stats["v246_wins"] += 1
            elif c246 is None:
                chosen.append(c244); stats["v244_wins"] += 1
            else:
                s244 = score(c244, feats)
                s246 = score(c246, feats)
                if s244 > s246 + 0.5:  # v244 needs clear advantage to win
                    chosen.append(c244); stats["v244_wins"] += 1
                elif s246 > s244:
                    chosen.append(c246); stats["v246_wins"] += 1
                else:
                    chosen.append(c246); stats["tied_v246"] += 1  # tiebreak: v246
        instruction_map[tid] = chosen[:3]

    # Build dataset
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

    return {"episodes": episodes}, stats


def run(split: str = "unseen"):
    if split == "unseen":
        gt_path  = GT_UNSEEN
        ck244    = CK244_U
        ck246    = CK246_U
        out_name = "val_unseen_v247.json.gz"
        habitat_dir = HABITAT_BASE / "val_unseen"
    else:
        gt_path  = GT_SEEN
        ck244    = CK244_S
        ck246    = CK246_S
        out_name = "val_seen_v247.json.gz"
        habitat_dir = HABITAT_BASE / "val_seen"

    with gzip.open(gt_path, "rt") as f:
        gt_data = json.load(f)

    if not ck244.exists():
        print(f"Missing v244 checkpoint: {ck244}"); return
    if not ck246.exists():
        print(f"Missing v246 checkpoint: {ck246}"); return

    with open(ck244) as f: map244 = json.load(f)
    with open(ck246) as f: map246 = json.load(f)

    # Normalise: each entry should be a list of 3
    def norm(m):
        return {k: (v if isinstance(v, list) else [v,v,v]) for k,v in m.items()}
    map244, map246 = norm(map244), norm(map246)

    out_data, stats = build_ensemble(gt_data, map244, map246)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)
    shutil.copy2(out_path, habitat_dir / out_name)

    ep_texts = [ep["instruction"]["instruction_text"] for ep in out_data["episodes"]]
    unique = len(set(ep_texts)); total = len(ep_texts)
    avg_w = sum(len(t.split()) for t in ep_texts) / total
    ord_cnt = sum(1 for t in ep_texts if any(o in t.lower() for o in ["first","second","third"]))
    noun_cnt = sum(1 for t in ep_texts if any(n in t.lower() for n in
        ["chair","table","door","window","desk","bed","counter","cabinet","couch"]))

    print(f"v247 {split}: {total} eps  {unique} unique ({unique/total*100:.0f}%)  avg_words={avg_w:.1f}")
    print(f"  ordinal_markers={ord_cnt/total*100:.1f}%  visual_noun={noun_cnt/total*100:.1f}%")
    print(f"  selection: v244_wins={stats['v244_wins']}  v246_wins={stats['v246_wins']}  tied→v246={stats['tied_v246']}")
    print(f"Saved: {habitat_dir / out_name}")

    # Sanity sample
    import random; random.seed(7)
    for ep in random.sample(out_data["episodes"][:500], 3):
        print(f"  [{ep['scene_id'][-6:]}] {ep['instruction']['instruction_text'][:100]}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="unseen", choices=["unseen","seen","both"])
    args = p.parse_args()
    if args.split == "both":
        for s in ["unseen","seen"]: run(s)
    else:
        run(args.split)
