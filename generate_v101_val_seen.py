#!/usr/bin/env python3
"""
Generate v101 for VAL_SEEN — val_seen v100 + far-wall fix + opener fix + sentence merge P=0.75

Applies the same post-processing as generate_v101_valunseen.py, but for val_seen split.

Val_seen v100 stats (before):
  avg_words: 26.184  (GT=27.093)  → still short, but hallway is NEAR-PERFECT
  hallway%:  19.0%   (GT=21.6%)   ← NEAR PERFECT
  far_wall%: 23.9%   → needs fix
  depart%:    2.7%,  proceed%: 2.7% → needs fix
  pct_3+sent: 42.8%  → too high

Expected after v101:
  avg_words: ~26.2 (word-neutral transforms)
  hallway%: ~19%  (unchanged)
  far_wall%: 0%   (FIXED)
  depart%: 0%,  proceed%: 0% (FIXED)
  pct_3+sent: ~21% (75% of 3+sent episodes merged)
"""

import gzip, json, re, hashlib
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent
HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
OUT_DIR = PIPELINE_ROOT / "outputs" / "datasets"

INPUT_FILE = OUT_DIR / "val_seen_gate3_gemma_v100.json.gz"
OUT_NAME = "val_seen_v101.json.gz"
DEPLOY_PATH = HABITAT_BASE / "val_seen" / OUT_NAME

SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
MERGE_PROB = 75  # P=0.75 (same hash as val_unseen v101)


def fix_far_wall(text: str) -> tuple:
    if 'far wall' not in text.lower():
        return text, False
    new = re.sub(r'\bfar wall\b', 'far end', text, flags=re.IGNORECASE)
    return new, new != text


def fix_opener(text: str) -> tuple:
    changed = False
    if re.match(r'^Depart from the ', text):
        text = re.sub(r'^Depart from the ', 'Head out of the ', text); changed = True
    elif re.match(r'^Depart from ', text):
        text = re.sub(r'^Depart from ', 'Head out of the ', text); changed = True
    elif re.match(r'^Proceed from the ', text):
        text = re.sub(r'^Proceed from the ', 'Leave the ', text); changed = True
    elif re.match(r'^Proceed from ', text):
        text = re.sub(r'^Proceed from ', 'Leave the ', text); changed = True
    return text, changed


def merge_sentences(text: str, eid: int) -> tuple:
    sents = [s.strip() for s in SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    if len(sents) < 3:
        return text, False
    s1, s2 = sents[0], sents[1]
    s3_rest = sents[2:]
    if len(s1.split()) + len(s2.split()) > 50:
        return text, False
    if len(s2.split()) + len(s3_rest[0].split()) > 40:
        return text, False
    h = int(hashlib.md5(f"{eid}_v205_merge".encode()).hexdigest(), 16) % 100
    if h >= MERGE_PROB:
        return text, False
    s2_clean = s2.rstrip('.!?')
    s3_0_lower = s3_rest[0][0].lower() + s3_rest[0][1:]
    merged_middle = f"{s2_clean}, {s3_0_lower}"
    remaining = ' '.join(s3_rest[1:]) if len(s3_rest) > 1 else ''
    result = f"{s1} {merged_middle} {remaining}".strip()
    return result, True


def analyze(texts, label, gt_avg_words, gt_hallway):
    n = len(texts)
    avg_words = sum(len(t.split()) for t in texts) / n
    turns = [len(re.findall(r'\bturn (?:left|right)\b', t, re.I)) for t in texts]
    avg_turns = sum(turns) / n
    hallway = sum(1 for t in texts if re.search(r'\bhallway\b', t, re.I)) / n * 100
    far_wall = sum(1 for t in texts if re.search(r'\bfar wall\b', t, re.I)) / n * 100
    depart = sum(1 for t in texts if re.match(r'^depart', t, re.I)) / n * 100
    proceed = sum(1 for t in texts if re.match(r'^proceed', t, re.I)) / n * 100
    walk = sum(1 for t in texts if re.match(r'^walk', t, re.I)) / n * 100
    sents = [len([s for s in SENT_SPLIT_RE.split(t.strip()) if s.strip()]) for t in texts]
    pct_3plus = sum(1 for s in sents if s >= 3) / n * 100
    print(f"\n{label} (n={n}):")
    print(f"  avg_words: {avg_words:.3f}  (GT={gt_avg_words})")
    print(f"  avg_turns: {avg_turns:.3f}  (GT=0.636)")
    print(f"  hallway%:  {hallway:.1f}%   (GT={gt_hallway}%)")
    print(f"  far_wall%: {far_wall:.1f}%")
    print(f"  depart%:   {depart:.1f}%   proceed%: {proceed:.1f}%")
    print(f"  walk%:     {walk:.1f}%")
    print(f"  pct_3+sent:{pct_3plus:.1f}%")


def main():
    print("=" * 70)
    print("val_seen v100 → v101 (far-wall + opener + sentence merge)")
    print("=" * 70)

    with gzip.open(INPUT_FILE, "rt") as f:
        d = json.load(f)

    vocab = d.get("instruction_vocab", {})
    episodes = d["episodes"]
    n = len(episodes)
    print(f"Loaded {n} val_seen episodes from v100")

    far_wall_fixed = 0
    opener_fixed = 0
    merged_count = 0
    results = []

    for ep in episodes:
        eid = ep.get("episode_id", 0)
        instr = ep["instruction"]["instruction_text"]

        instr, fw = fix_far_wall(instr)
        if fw: far_wall_fixed += 1

        instr, op = fix_opener(instr)
        if op: opener_fixed += 1

        instr, mg = merge_sentences(instr, eid)
        if mg: merged_count += 1

        out_ep = dict(ep)
        out_ep["instruction"] = {"instruction_text": instr}
        results.append(out_ep)

    texts = [e["instruction"]["instruction_text"] for e in results]
    analyze(texts, "val_seen v101", gt_avg_words=27.093, gt_hallway=21.6)

    print(f"\n  far_wall_fixed: {far_wall_fixed}")
    print(f"  opener_fixed:   {opener_fixed}")
    print(f"  merged:         {merged_count}/{n}")

    out_data = {"episodes": results, "instruction_vocab": vocab}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = OUT_DIR / OUT_NAME
    with gzip.open(local_path, "wt") as f:
        json.dump(out_data, f)
    print(f"\nSaved: {local_path} ({local_path.stat().st_size // 1024} KB)")

    DEPLOY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(DEPLOY_PATH, "wt") as f:
        json.dump(out_data, f)
    print(f"Deployed: {DEPLOY_PATH}")


if __name__ == "__main__":
    main()
