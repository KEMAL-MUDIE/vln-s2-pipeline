#!/usr/bin/env python3
"""
Generate v205 — gate4 visual v24 + sentence merging (3→2 sentences).

v24 issue: pct_3sent=66.2% (GT=29.3%). Too many 3-sentence instructions.
v24 advantage: avg_words=26.774 (PERFECT), walk=34% go=18.6% exit=10.7% ALL PERFECT.

v205 fix: merge S2+S3 with ", then" connector (word-count-neutral).
  - Only merge if len(s1.split()) + len(s2.split()) ≤ 50 words (avoid run-ons)
  - P=0.57 (hash-based) → reduces 3-sent from 66.2% to ~29% (GT=29.3%)
  - Opener (S1) is NEVER modified → preserves PERFECT opener distribution
  - avg_words stays ≈ 26.77 (same words, different sentence structure)

Expected v205 stats:
  avg_words: 26.77 (unchanged PERFECT)
  walk: 34.0% go: 18.6% exit: 10.7% (ALL UNCHANGED PERFECT)
  pct_3sent: ~29% (GT=29.3%)
  avg_turns: ~1.546 (unchanged)
"""

import gzip, json, re, hashlib
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent
HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
OUT_DIR = PIPELINE_ROOT / "outputs" / "datasets"

VERSION = "v205"
INPUT_FILE = OUT_DIR / "val_unseen_generated_gemma_visual_v24.json.gz"
OUT_NAME = f"val_unseen_auto_{VERSION}.json.gz"
DEPLOY_PATH = HABITAT_BASE / "val_unseen" / OUT_NAME

SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def merge_sentences(text: str, eid: int) -> tuple:
    """
    For 3-sentence instructions, merge S2+S3 with ', then ' connector.
    Returns (modified_text, merged: bool).
    """
    sents = [s.strip() for s in SENT_SPLIT_RE.split(text.strip()) if s.strip()]

    if len(sents) < 3:
        return text, False

    s1, s2, s3 = sents[0], sents[1], sents[2:]
    # Only merge s2+s3[0] if combined length is manageable
    if len(s1.split()) + len(s2.split()) > 50:
        return text, False
    if len(s2.split()) + len(s3[0].split()) > 40:
        return text, False

    # Hash-based decision: P=0.57
    h = int(hashlib.md5(f"{eid}_v205_merge".encode()).hexdigest(), 16) % 100
    if h >= 57:
        return text, False

    # Merge s2 and s3[0] with ", then"
    s2_clean = s2.rstrip('.!?')
    s3_0_clean = s3[0][0].lower() + s3[0][1:]  # lowercase start of s3[0]
    merged_middle = f"{s2_clean}, then {s3_0_clean}"

    remaining = ' '.join(s3[1:]) if len(s3) > 1 else ''
    if remaining:
        result = f"{s1} {merged_middle} {remaining}"
    else:
        result = f"{s1} {merged_middle}"

    return result, True


def count_sents(text):
    return len([s for s in SENT_SPLIT_RE.split(text.strip()) if s.strip()])


def main():
    print("=" * 70)
    print(f"Gate4-Visual {VERSION} — sentence merge on v24")
    print("Target: pct_3sent 66.2%→29% (GT=29.3%), preserve avg_words and openers")
    print("=" * 70)

    with gzip.open(INPUT_FILE, "rt") as f:
        d = json.load(f)

    vocab = d.get("instruction_vocab", {})
    episodes = d["episodes"]
    n = len(episodes)
    print(f"Loaded {n} episodes from gate4 visual v24")

    results = []
    merged_count = 0
    eligible_count = 0

    for ep in episodes:
        eid = ep.get("episode_id", 0)
        instr = ep["instruction"]["instruction_text"]

        sents = [s.strip() for s in SENT_SPLIT_RE.split(instr.strip()) if s.strip()]
        if len(sents) >= 3:
            eligible_count += 1

        new_instr, merged = merge_sentences(instr, eid)
        if merged:
            merged_count += 1

        out_ep = dict(ep)
        out_ep["instruction"] = {"instruction_text": new_instr}
        results.append(out_ep)

    texts = [e["instruction"]["instruction_text"] for e in results]
    avg_words = sum(len(t.split()) for t in texts) / n
    turns = [len(re.findall(r'\bturn (?:left|right|around|slightly)\b', t, re.I)) for t in texts]
    avg_turns = sum(turns) / n
    pct_3plus_turns = sum(1 for t in turns if t >= 3) / n * 100
    turn_openers = sum(1 for t in texts if t.lower().startswith('turn')) / n * 100

    sent_dist = {}
    for t in texts:
        k = count_sents(t)
        sent_dist[k] = sent_dist.get(k, 0) + 1

    walk = sum(1 for t in texts if re.match(r'^walk', t, re.I)) / n * 100
    go   = sum(1 for t in texts if re.match(r'^go', t, re.I)) / n * 100
    exit_ = sum(1 for t in texts if re.match(r'^exit', t, re.I)) / n * 100

    print(f"\nResults (GT: avg_words=26.78, walk=34%, go=18.5%, exit=10.7%, 3sent=29.3%):")
    print(f"  avg_words:     {avg_words:.3f}  (v24=26.774)")
    print(f"  avg_turns:     {avg_turns:.3f}  (v24=1.546, GT=0.66)")
    print(f"  pct_3+_turns:  {pct_3plus_turns:.1f}%   (v24=18.6%, GT=3.8%)")
    print(f"  turn_openers:  {turn_openers:.1f}%  (v24=16.6%, GT=16.5%)")
    print(f"  walk: {walk:.1f}%  go: {go:.1f}%  exit: {exit_:.1f}%  (GT: 34/18.5/10.7)")
    print(f"  sent_dist: {dict(sorted(sent_dist.items()))}")
    pct_3sent = sum(v for k, v in sent_dist.items() if k >= 3) / n * 100
    print(f"  pct_3+_sent:   {pct_3sent:.1f}%   (v24=66.2%, GT=29.3%)")
    print(f"  merged: {merged_count}/{eligible_count} eligible episodes")

    out_data = {"episodes": results, "instruction_vocab": vocab}
    local_path = OUT_DIR / OUT_NAME
    with gzip.open(local_path, "wt") as f:
        json.dump(out_data, f)
    print(f"\nSaved:    {local_path} ({local_path.stat().st_size // 1024} KB)")
    DEPLOY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(DEPLOY_PATH, "wt") as f:
        json.dump(out_data, f)
    print(f"Deployed: {DEPLOY_PATH}")


if __name__ == "__main__":
    main()
