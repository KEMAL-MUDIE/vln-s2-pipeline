#!/usr/bin/env python3
"""Generate v99: v95 + sentence merger (reduce avg_sents from 2.77 → ~2.5).

GT: avg_sents=2.50, avg_last_sent_words=12.3
v95: avg_sents=2.77, avg_last_sent_words=8.0

v95 has too many short 3-sentence instructions vs GT's 2-sentence preference.

MERGE RULE: When a short sentence (≤8 words, not a stop) is followed by another
movement sentence, merge with "and" connector.
  "Walk straight. Turn left at the rug." → "Walk straight and turn left at the rug."
  "Continue forward. Stop near the table." → (don't merge stop with prior move)

RESULT: avg_sents closer to GT's 2.50, more natural flow.
"""
import gzip, json, re
from pathlib import Path

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
GT_PATH = BASE / "val_unseen" / "val_unseen_patched.json.gz"
V95_PATH = BASE / "val_unseen" / "val_unseen_v95.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v99.json.gz"

SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
MOVE_START_RE = re.compile(r'^(walk|go|continue|head|move|proceed|take|exit|leave|enter|turn|pass)', re.IGNORECASE)
STOP_START_RE = re.compile(r'^(stop|wait|stand|halt)', re.IGNORECASE)


def should_merge(s1, s2):
    """Check if s1 and s2 should be merged with 'and'."""
    words1 = s1.split()
    if len(words1) > 8:  # s1 too long to merge
        return False
    if STOP_START_RE.match(s1):  # don't merge stop sentences
        return False
    if not MOVE_START_RE.match(s2):  # s2 must start with movement
        return False
    if STOP_START_RE.match(s2):  # don't merge if s2 is stop
        return False
    return True


def merge_sentences(text):
    sentences = [s.strip() for s in SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    if len(sentences) <= 1:
        return text

    merged = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        # Remove trailing punctuation for merging
        s_clean = s.rstrip('.!?')
        if i + 1 < len(sentences) and should_merge(s, sentences[i + 1]):
            # Merge with next sentence
            s2_clean = sentences[i + 1].rstrip('.!?')
            # Make s1 lowercase at end, keep s2 as is
            combined = s_clean + ' and ' + s2_clean[0].lower() + s2_clean[1:]
            merged.append(combined)
            i += 2
        else:
            merged.append(s_clean)
            i += 1

    # Reconstruct with periods, capitalize each sentence start
    result = '. '.join(s[0].upper() + s[1:] if s else s for s in merged) + '.'
    # Fix double periods
    result = re.sub(r'\.+', '.', result)
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def main():
    print("=== v99: v95 + sentence merger (reduce avg_sents 2.77 → ~2.5) ===")
    with gzip.open(GT_PATH) as f: gt_data = json.load(f)
    with gzip.open(V95_PATH) as f: v95_data = json.load(f)
    v95_by_eid = {ep["episode_id"]: ep for ep in v95_data["episodes"]}
    new_episodes = []; changed = 0
    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        v95_ep = v95_by_eid[eid]
        orig = v95_ep["instruction"]["instruction_text"]
        merged = merge_sentences(orig)
        if merged != orig: changed += 1
        new_ep = dict(v95_ep)
        new_ep["instruction"] = dict(v95_ep["instruction"])
        new_ep["instruction"]["instruction_text"] = merged
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)
    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]
    TURN_RE = re.compile(r'\bturn\b', re.I)
    INIT_TURN_RE = re.compile(r'^Turn', re.I)
    sents_per = [len([s for s in SENT_SPLIT_RE.split(i.strip()) if s]) for i in all_insts]
    last_sents = []
    for i in all_insts:
        parts = [s for s in SENT_SPLIT_RE.split(i.strip()) if s]
        if parts: last_sents.append(parts[-1])
    last_words = [len(s.split()) for s in last_sents]
    print(f"Changed {changed}/{n} ({changed/n*100:.1f}%)")
    print(f"avg_words={sum(words)/n:.1f}  [GT=26.8]")
    print(f"avg_sents={sum(sents_per)/n:.2f}  [GT=2.50, v95=2.77]")
    print(f"avg_last_sent={sum(last_words)/n:.1f}  [GT=12.3, v95=8.0]")
    print(f"starts_Turn={sum(1 for i in all_insts if INIT_TURN_RE.match(i))/n*100:.1f}%  [GT=16.5%]")
    print(f"unique={len(set(all_insts))/n*100:.1f}%")
    print("\nSamples (changed):")
    shown = 0
    for ep, v95_ep in zip(new_episodes, v95_data["episodes"]):
        if ep["instruction"]["instruction_text"] != v95_ep["instruction"]["instruction_text"] and shown < 5:
            print(f"  v95: {v95_ep['instruction']['instruction_text'][:160]}")
            print(f"  v99: {ep['instruction']['instruction_text'][:160]}")
            print()
            shown += 1
    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f: json.dump(out_data, f)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__": main()
