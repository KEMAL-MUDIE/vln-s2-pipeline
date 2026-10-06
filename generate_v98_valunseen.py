#!/usr/bin/env python3
"""Generate v98: v96 + hallway movement reduction.

v96: starts_Turn=21.9% (close to GT 16.5%), but hallway=32.4% (GT=20.4%)
Apply v32's hallway fix to reduce hallway from 32.4% toward GT's 20.4%.
"""
import gzip, json, re
from pathlib import Path

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
GT_PATH = BASE / "val_unseen" / "val_unseen_patched.json.gz"
V96_PATH = BASE / "val_unseen" / "val_unseen_v96.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v98.json.gz"

def fix_hallway(text):
    if 'hallway' not in text.lower(): return text, 0
    changes = 0
    for pat, repl in [
        (r'\binto the hallway\b', 'forward'),
        (r'\bthrough the hallway\b', 'forward'),
        (r'\balong the hallway\b', 'forward'),
        (r'\bout of the hallway\b', 'out'),
    ]:
        n = len(re.findall(pat, text, re.IGNORECASE))
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
        changes += n
    return text, changes

def fix_artifacts(text):
    text = re.sub(r'\bforward\s+forward\b', 'forward', text, re.IGNORECASE)
    text = re.sub(r'\bout\s+out\b', 'out', text, re.IGNORECASE)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r',\s*\.', '.', text)
    return text.strip()

def main():
    print("=== v98: v96 + hallway movement reduction ===")
    with gzip.open(GT_PATH) as f: gt_data = json.load(f)
    with gzip.open(V96_PATH) as f: v96_data = json.load(f)
    v96_by_eid = {ep["episode_id"]: ep for ep in v96_data["episodes"]}
    new_episodes = []; changed = 0; hallway_total = 0
    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        v96_ep = v96_by_eid[eid]
        orig = v96_ep["instruction"]["instruction_text"]
        fixed, h_cnt = fix_hallway(orig)
        fixed = fix_artifacts(fixed)
        if fixed != orig: changed += 1
        hallway_total += h_cnt
        new_ep = dict(v96_ep)
        new_ep["instruction"] = dict(v96_ep["instruction"])
        new_ep["instruction"]["instruction_text"] = fixed
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)
    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]
    TURN_RE = re.compile(r'\bturn\b', re.I)
    INIT_TURN_RE = re.compile(r'^Turn', re.I)
    HALLWAY_RE = re.compile(r'\bhallway\b', re.I)
    print(f"Changed {changed}/{n} ({changed/n*100:.1f}%) — {hallway_total} hallway instances replaced")
    print(f"avg_words={sum(words)/n:.1f}  [GT=26.8]")
    print(f"starts_Turn={sum(1 for i in all_insts if INIT_TURN_RE.match(i))/n*100:.1f}%  [GT=16.5%]")
    print(f"hallway={sum(1 for i in all_insts if HALLWAY_RE.search(i))/n*100:.1f}%  [GT=20.4%]")
    print(f"unique={len(set(all_insts))/n*100:.1f}%")
    print("\nSamples (changed):")
    shown = 0
    for ep, v96_ep in zip(new_episodes, v96_data["episodes"]):
        if ep["instruction"]["instruction_text"] != v96_ep["instruction"]["instruction_text"] and shown < 3:
            print(f"  v96: {v96_ep['instruction']['instruction_text'][:150]}")
            print(f"  v98: {ep['instruction']['instruction_text'][:150]}")
            print()
            shown += 1
    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f: json.dump(out_data, f)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__": main()
