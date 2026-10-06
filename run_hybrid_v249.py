#!/usr/bin/env python3
"""
Gate 4 v249 — Sentence-Level Hybrid: v244 body + v246 stop.

The insight: v244 has ordinal-rich navigation (65.9% ordinals) but generic stops;
v246 has specific visual goal anchors (90.4% noun) but moderate ordinals (21%).
The optimal instruction combines both: use v244's body (ordinal turns, path desc)
and replace the final stop sentence with v246's visually-specific stop condition.

Hybrid algorithm per instruction slot:
  1. Split v244 instruction into sentences.
  2. Identify v246's best stop sentence (last sentence with stop word, or last sentence).
  3. Output: v244_body (all but last sentence) + v246_stop.
  4. Quality check: if result is too short (<8w) or the transition is awkward
     (same sentence in both), fall back to whichever scores higher via the scorer.

Post-processing: run the mega-ensemble scorer over {hybrid, v244, v246} and take best.

No API calls. Runs in <5 seconds.
"""
import gzip
import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT         = Path(__file__).parent
GT_UNSEEN    = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz"
GT_SEEN      = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_seen/val_seen.json.gz"
OUT_DIR      = ROOT / "outputs" / "datasets"
HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")

CK244_U = ROOT / "outputs" / "gate4_v244_unseen_checkpoint.json"
CK244_S = ROOT / "outputs" / "gate4_v244_seen_checkpoint.json"
CK246_U = ROOT / "outputs" / "gate4_v246_unseen_checkpoint.json"
CK246_S = ROOT / "outputs" / "gate4_v246_seen_checkpoint.json"

# ── Geometry ──────────────────────────────────────────────────────────────────

def dist3d(p1, p2):
    return math.sqrt(sum((a-b)**2 for a,b in zip(p1,p2)))

def heading_xz(p1, p2):
    dx, dz = p2[0]-p1[0], p2[2]-p1[2]
    return math.degrees(math.atan2(dx,-dz)) if (abs(dx)>1e-6 or abs(dz)>1e-6) else 0.0

def signed_diff(a, b):
    return (b-a+180)%360-180

def path_features(ep):
    path = ep.get("reference_path",[])
    n_turns=0; is_mf=False
    for i in range(1, len(path)-1):
        if abs(signed_diff(heading_xz(path[i-1],path[i]),heading_xz(path[i],path[i+1])))>20:
            n_turns+=1
        if abs(path[i+1][1]-path[i][1])>=0.25:
            is_mf=True
    return {"n_turns":n_turns,"is_multifloor":is_mf}

# ── Quality scorer ─────────────────────────────────────────────────────────────

STOP_NOUNS={"chair","table","couch","sofa","door","doorway","window","wall","desk","bed",
    "counter","cabinet","lamp","shelf","mirror","stair","stairs","step","steps","carpet",
    "rug","column","pillar","island","sink","fireplace","hallway","corridor","bookcase",
    "bookshelf","dresser","wardrobe","closet","bath","bathroom","kitchen","living","dining",
    "bedroom","lobby","entryway","foyer","balcony","archway","painting","picture","plant",
    "sign","tv","television","armchair","recliner","ottoman","bench"}
STAIR_W={"stair","stairs","step","steps","flight","floor","level","ascend","descend","climb"}
ORDINALS={"first","second","third","fourth","1st","2nd","3rd"}
STOP_W={"stop","wait","halt","stand","pause"}

def score(instr, feats):
    t=instr.lower(); words=t.split(); s=0.0
    if any(o in t for o in ORDINALS):
        s += 2.0 if feats["n_turns"]>1 else 0.5
    has_stair=any(w in t for w in STAIR_W)
    if feats["is_multifloor"]: s += 3.0 if has_stair else -2.0
    elif has_stair: s -= 1.0
    dir_c=t.count("left")+t.count("right")
    s += min(dir_c, feats["n_turns"]+1)*1.0
    sents=re.split(r'[.!?]', instr)
    last=(sents[-2] if len(sents)>1 else sents[-1]).lower()
    if any(w in t for w in STOP_W): s+=2.0
    if any(n in last for n in STOP_NOUNS): s+=2.0
    n=len(words)
    if 15<=n<=40: s+=1.0
    elif n<8 or n>60: s-=1.0
    return s

# ── Hybrid construction ───────────────────────────────────────────────────────

def split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def get_stop_sentence(instr: str) -> Optional[str]:
    sents = split_sentences(instr)
    for s in reversed(sents):
        if any(w in s.lower() for w in STOP_W):
            return s
    return sents[-1] if sents else None

def has_visual_noun(sent: str) -> bool:
    return any(n in sent.lower() for n in STOP_NOUNS)

def hybrid(v244_instr: str, v246_instr: str) -> str:
    """Return v244 body + v246 stop sentence."""
    v244_sents = split_sentences(v244_instr)
    v246_stop  = get_stop_sentence(v246_instr)

    if not v246_stop:
        return v244_instr

    # v244 body = all but last sentence (if multi-sentence)
    if len(v244_sents) > 1:
        body = " ".join(v244_sents[:-1])
    else:
        body = v244_sents[0] if v244_sents else ""

    # Avoid duplicating essentially the same stop
    if body.lower().strip().rstrip(".,!?") == v246_stop.lower().strip().rstrip(".,!?"):
        return v244_instr

    # Ensure body ends with period
    if body and not body[-1] in ".!?":
        body += "."

    result = (body + " " + v246_stop).strip() if body else v246_stop
    return result


def build_dataset(gt_data, ck244, ck246, out_name, habitat_dir):
    traj_to_feat = {}
    for ep in gt_data["episodes"]:
        tid = f"traj_{ep['trajectory_id']}" if "trajectory_id" in ep else f"traj_ep{ep['episode_id']}"
        if tid not in traj_to_feat:
            traj_to_feat[tid] = path_features(ep)

    stats = {"hybrid":0, "v244":0, "v246":0, "fallback":0}
    instruction_map = {}

    for tid, feats in traj_to_feat.items():
        i244 = ck244.get(tid, [])
        i246 = ck246.get(tid, [])
        if not i244 and not i246:
            continue
        chosen = []
        for slot in range(3):
            c244 = i244[slot] if slot<len(i244) else None
            c246 = i246[slot] if slot<len(i246) else None
            if c244 and c246:
                hyb  = hybrid(c244, c246)
                s_hyb = score(hyb,  feats)
                s244  = score(c244, feats)
                s246  = score(c246, feats)
                best_s = max(s_hyb, s244, s246)
                if s_hyb >= best_s - 0.01:
                    chosen.append(hyb); stats["hybrid"]+=1
                elif s244 >= best_s - 0.01:
                    chosen.append(c244); stats["v244"]+=1
                else:
                    chosen.append(c246); stats["v246"]+=1
            elif c244:
                chosen.append(c244); stats["v244"]+=1
            elif c246:
                chosen.append(c246); stats["v246"]+=1
            else:
                stats["fallback"]+=1
        instruction_map[tid] = chosen[:3]

    episodes = []
    traj_counter = {}
    for ep in gt_data["episodes"]:
        tid = f"traj_{ep['trajectory_id']}" if "trajectory_id" in ep else f"traj_ep{ep['episode_id']}"
        idx = traj_counter.get(tid, 0); traj_counter[tid] = idx+1
        instrs = instruction_map.get(tid, [])
        text = instrs[idx%len(instrs)] if instrs else ep["instruction"]["instruction_text"]
        new_ep = dict(ep); new_ep["instruction"] = dict(ep["instruction"])
        new_ep["instruction"]["instruction_text"] = text
        episodes.append(new_ep)
    out_data = {"episodes": episodes}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
    with gzip.open(out_path,"wt",encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)
    shutil.copy2(out_path, habitat_dir/out_name)

    ep_texts=[ep["instruction"]["instruction_text"] for ep in out_data["episodes"]]
    total=len(ep_texts); unique=len(set(ep_texts))
    avg_w=sum(len(t.split()) for t in ep_texts)/total
    ord_c=sum(1 for t in ep_texts if any(o in t.lower() for o in ["first","second","third"]))
    noun_c=sum(1 for t in ep_texts if any(n in t.lower() for n in
        ["chair","table","door","window","desk","bed","counter","couch","cabinet"]))
    print(f"  {total} eps  {unique} unique ({unique/total*100:.0f}%)  avg_words={avg_w:.1f}")
    print(f"  ordinal={ord_c/total*100:.1f}%  visual_noun={noun_c/total*100:.1f}%")
    print(f"  selection: hybrid={stats['hybrid']}  v244={stats['v244']}  v246={stats['v246']}")
    print(f"  Saved: {habitat_dir/out_name}")

    import random; random.seed(13)
    for ep in random.sample(out_data["episodes"][:500], 3):
        print(f"  [{ep['scene_id'][-6:]}] {ep['instruction']['instruction_text'][:100]}")
    return out_data


def load_ck(path):
    with open(path) as f: raw=json.load(f)
    return {k:(v if isinstance(v,list) else [v,v,v]) for k,v in raw.items()}


def run(split="unseen"):
    if split=="unseen":
        gt_path=GT_UNSEEN; ck244=CK244_U; ck246=CK246_U
        out_name="val_unseen_v249.json.gz"
        habitat_dir=HABITAT_BASE/"val_unseen"
    else:
        gt_path=GT_SEEN; ck244=CK244_S; ck246=CK246_S
        out_name="val_seen_v249.json.gz"
        habitat_dir=HABITAT_BASE/"val_seen"

    print(f"v249 {split}: loading checkpoints...")
    if not ck244.exists(): print(f"Missing v244 CK: {ck244}"); return
    if not ck246.exists(): print(f"Missing v246 CK: {ck246}"); return
    with gzip.open(gt_path,"rt") as f: gt_data=json.load(f)
    build_dataset(gt_data, load_ck(ck244), load_ck(ck246), out_name, habitat_dir)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--split",default="unseen",choices=["unseen","seen","both"])
    args=p.parse_args()
    if args.split=="both":
        for s in ["unseen","seen"]: run(s)
    else:
        run(args.split)
