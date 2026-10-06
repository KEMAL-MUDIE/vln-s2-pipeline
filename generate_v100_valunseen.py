#!/usr/bin/env python3
"""Generate v100: v96 + sentence merger.

v96: starts_Turn=21.9%, hallway=32.4% (close to GT but hallway high)
v99: v95 + merger = avg_sents 2.77→2.42

v100 = v96 + same sentence merger
  Expected: starts_Turn=21.9%, avg_sents~2.4, hallway=32.4% (unchanged)
  Tests: does merger help when combined with threshold-based initial turns?
"""
import gzip, json, re
from pathlib import Path

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
GT_PATH = BASE / "val_unseen" / "val_unseen_patched.json.gz"
V96_PATH = BASE / "val_unseen" / "val_unseen_v96.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v100.json.gz"

SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
MOVE_START_RE = re.compile(r'^(walk|go|continue|head|move|proceed|take|exit|leave|enter|turn|pass)', re.IGNORECASE)
STOP_START_RE = re.compile(r'^(stop|wait|stand|halt)', re.IGNORECASE)

def should_merge(s1, s2):
    words1 = s1.split()
    if len(words1) > 8: return False
    if STOP_START_RE.match(s1): return False
    if not MOVE_START_RE.match(s2): return False
    if STOP_START_RE.match(s2): return False
    return True

def merge_sentences(text):
    sentences = [s.strip() for s in SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    if len(sentences) <= 1: return text
    merged = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        s_clean = s.rstrip('.!?')
        if i + 1 < len(sentences) and should_merge(s, sentences[i + 1]):
            s2_clean = sentences[i + 1].rstrip('.!?')
            combined = s_clean + ' and ' + s2_clean[0].lower() + s2_clean[1:]
            merged.append(combined)
            i += 2
        else:
            merged.append(s_clean)
            i += 1
    result = '. '.join(s[0].upper() + s[1:] if s else s for s in merged) + '.'
    result = re.sub(r'\.+', '.', result)
    result = re.sub(r'\s+', ' ', result).strip()
    return result

def main():
    print("=== v100: v96 + sentence merger ===")
    with gzip.open(GT_PATH) as f: gt_data = json.load(f)
    with gzip.open(V96_PATH) as f: v96_data = json.load(f)
    v96_by_eid = {ep["episode_id"]: ep for ep in v96_data["episodes"]}
    new_episodes = []; changed = 0
    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        v96_ep = v96_by_eid[eid]
        orig = v96_ep["instruction"]["instruction_text"]
        merged = merge_sentences(orig)
        if merged != orig: changed += 1
        new_ep = dict(v96_ep)
        new_ep["instruction"] = dict(v96_ep["instruction"])
        new_ep["instruction"]["instruction_text"] = merged
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)
    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]
    INIT_TURN_RE = re.compile(r'^Turn', re.I)
    HALLWAY_RE = re.compile(r'\bhallway\b', re.I)
    sents_per = [len([s for s in SENT_SPLIT_RE.split(i.strip()) if s]) for i in all_insts]
    print(f"Changed {changed}/{n} ({changed/n*100:.1f}%)")
    print(f"avg_words={sum(words)/n:.1f}  [GT=26.8]")
    print(f"avg_sents={sum(sents_per)/n:.2f}  [GT=2.50, v96=2.77]")
    print(f"starts_Turn={sum(1 for i in all_insts if INIT_TURN_RE.match(i))/n*100:.1f}%  [GT=16.5%]")
    print(f"hallway={sum(1 for i in all_insts if HALLWAY_RE.search(i))/n*100:.1f}%  [GT=20.4%]")
    print(f"unique={len(set(all_insts))/n*100:.1f}%")
    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f: json.dump(out_data, f)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__": main()
