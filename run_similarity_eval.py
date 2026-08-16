#!/usr/bin/env python3
"""
Metadata Similarity Evaluator
Compares reproduced instructions vs GT across 10+ metrics.
Run against all versions to track progress toward SR > 65%.

Metrics:
  1.  BLEU-1        n-gram precision (unigrams)
  2.  BLEU-2        n-gram precision (bigrams)
  3.  BLEU-4        n-gram precision (4-grams)
  4.  ROUGE-L       longest common subsequence F1
  5.  METEOR        synonym + stemming aware recall
  6.  Jaccard       word set overlap
  7.  Edit-dist     normalized Levenshtein word-level distance (lower=better, shown as similarity)
  8.  Len-ratio     |reproduced_words| / |gt_words| (1.0 = perfect)
  9.  Verb-match    first word matches GT start verb
  10. Stop-match    has stop/wait/halt word
  11. Turn-accuracy turn directions in reproduced match GT turns (left/right)
  12. Noun-F1       content noun overlap (proxy for landmark accuracy)
"""
import gzip, json, math, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"

# ── Text normalisation ─────────────────────────────────────────────────────────

def tokenize(text: str):
    return re.findall(r"[a-z']+", text.lower())

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# ── BLEU (per-instance, smoothed) ─────────────────────────────────────────────

def bleu_n(hyp_tokens, ref_tokens, n):
    hyp_ng = Counter(ngrams(hyp_tokens, n))
    ref_ng = Counter(ngrams(ref_tokens, n))
    clip   = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
    total  = max(sum(hyp_ng.values()), 1)
    return clip / total

def bleu(hyp, ref, max_n=4):
    h = tokenize(hyp); r = tokenize(ref)
    if not h or not r: return 0.0
    # brevity penalty
    bp = 1.0 if len(h) >= len(r) else math.exp(1 - len(r)/len(h))
    precs = []
    for n in range(1, max_n+1):
        p = bleu_n(h, r, n)
        precs.append(p + 1e-10)   # add-1 smoothing
    geo = math.exp(sum(math.log(p) for p in precs) / max_n)
    return bp * geo

def bleu1(hyp, ref):
    h = tokenize(hyp); r = tokenize(ref)
    if not h or not r: return 0.0
    ref_counts = Counter(r)
    clip = sum(min(c, ref_counts[w]) for w, c in Counter(h).items())
    return clip / len(h)

def bleu2(hyp, ref):
    h = tokenize(hyp); r = tokenize(ref)
    return bleu_n(h, r, 2)

# ── ROUGE-L ────────────────────────────────────────────────────────────────────

def lcs_len(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(2)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i%2][j] = dp[(i-1)%2][j-1]+1 if a[i-1]==b[j-1] else max(dp[(i-1)%2][j], dp[i%2][j-1])
    return dp[m%2][n]

def rouge_l(hyp, ref):
    h = tokenize(hyp); r = tokenize(ref)
    if not h or not r: return 0.0
    l = lcs_len(h, r)
    p = l / len(h); rec = l / len(r)
    if p + rec == 0: return 0.0
    return 2*p*rec / (p+rec)

# ── METEOR (approx, no WordNet) ────────────────────────────────────────────────

def stem(w):
    for suf in ("ing","tion","ly","ed","er","est","s"):
        if w.endswith(suf) and len(w)-len(suf)>2:
            return w[:-len(suf)]
    return w

def meteor(hyp, ref, alpha=0.9, beta=3, gamma=0.5):
    h = tokenize(hyp); r = tokenize(ref)
    if not h or not r: return 0.0
    # exact match
    r_left = list(r)
    matched = 0
    h_matched = []
    for i, w in enumerate(h):
        if w in r_left:
            r_left.remove(w); matched += 1; h_matched.append(i)
    # stem match on unmatched
    r_stems = Counter(stem(w) for w in r_left)
    for i, w in enumerate(h):
        if i not in h_matched:
            s = stem(w)
            if r_stems[s] > 0:
                r_stems[s] -= 1; matched += 1; h_matched.append(i)
    if matched == 0: return 0.0
    prec = matched / len(h)
    rec  = matched / len(r)
    f    = prec * rec / (alpha*prec + (1-alpha)*rec)
    # chunk penalty
    chunks = 1
    for i in range(1, len(h_matched)):
        if h_matched[i] != h_matched[i-1]+1:
            chunks += 1
    pen = gamma * (chunks/matched)**beta
    return f * (1 - pen)

# ── Jaccard ────────────────────────────────────────────────────────────────────

def jaccard(hyp, ref):
    h = set(tokenize(hyp)); r = set(tokenize(ref))
    if not h | r: return 0.0
    return len(h & r) / len(h | r)

# ── Edit distance (word-level, normalised) ─────────────────────────────────────

def edit_dist_sim(hyp, ref):
    h = tokenize(hyp); r = tokenize(ref)
    m, n = len(h), len(r)
    if m == 0 and n == 0: return 1.0
    dp = list(range(n+1))
    for i in range(1, m+1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n+1):
            dp[j] = prev[j-1] if h[i-1]==r[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return 1 - dp[n] / max(m+n, 1)

# ── Verb / stop / turn / noun helpers ────────────────────────────────────────

STOP_WORDS = {"stop","wait","halt","stand","pause"}
TURN_WORDS = {"left","right"}
SKIP_NOUNS = {"the","a","an","and","or","to","of","in","at","on","by","with","into",
              "through","past","walk","go","turn","exit","leave","enter","straight",
              "forward","you","your","door","room","hallway","stairs","step","steps",
              "floor","wall","side","front","back","end","up","down","out","off","over"}

def first_verb(text):
    toks = tokenize(text)
    return toks[0] if toks else ""

def has_stop(text):
    return any(w in STOP_WORDS for w in tokenize(text))

def turn_mentions(text):
    toks = tokenize(text)
    turns = []
    for i, w in enumerate(toks):
        if w == "left" and i>0 and tokenize(text)[max(0,i-2):i]:
            turns.append("left")
        elif w == "right" and i>0:
            turns.append("right")
    return turns

def noun_tokens(text):
    stop = {"the","a","an","and","or","to","of","in","at","on","by","with","into",
            "through","past","out","up","down","then","you","your","is","are","was","be"}
    return [w for w in tokenize(text) if w not in stop and len(w)>2]

def noun_f1(hyp, ref):
    h = Counter(noun_tokens(hyp)); r = Counter(noun_tokens(ref))
    if not h or not r: return 0.0
    tp = sum(min(h[w], r[w]) for w in h)
    prec = tp/sum(h.values()); rec = tp/sum(r.values())
    if prec+rec == 0: return 0.0
    return 2*prec*rec/(prec+rec)

def turn_accuracy(hyp_turns, ref_turns):
    if not ref_turns: return 1.0
    n = min(len(hyp_turns), len(ref_turns))
    if n == 0: return 0.0
    matches = sum(1 for i in range(n) if hyp_turns[i]==ref_turns[i])
    return matches / len(ref_turns)

# ── Main evaluator ────────────────────────────────────────────────────────────

def eval_dataset(reproduced_path: str, gt_path: str = GT_PATH, n_sample: int = None) -> dict:
    with gzip.open(gt_path, "rt") as f:
        gt_data = json.load(f)
    gt_map = {
        ep["episode_id"]: (ep["instruction"]["instruction_text"]
                           if isinstance(ep.get("instruction"), dict)
                           else ep.get("instruction", ""))
        for ep in gt_data["episodes"]
    }

    with gzip.open(reproduced_path, "rt") as f:
        rep_data = json.load(f)

    episodes = rep_data["episodes"]
    if n_sample:
        import random; random.seed(42)
        episodes = random.sample(episodes, min(n_sample, len(episodes)))

    scores = defaultdict(list)
    for ep in episodes:
        eid = ep["episode_id"]
        rep_instr = (ep["instruction"]["instruction_text"]
                     if isinstance(ep.get("instruction"), dict)
                     else ep.get("instruction", ""))
        gt_instr  = gt_map.get(eid, "")
        if not gt_instr or not rep_instr:
            continue

        scores["bleu1"].append(bleu1(rep_instr, gt_instr))
        scores["bleu2"].append(bleu2(rep_instr, gt_instr))
        scores["bleu4"].append(bleu(rep_instr,  gt_instr, 4))
        scores["rouge_l"].append(rouge_l(rep_instr, gt_instr))
        scores["meteor"].append(meteor(rep_instr, gt_instr))
        scores["jaccard"].append(jaccard(rep_instr, gt_instr))
        scores["edit_sim"].append(edit_dist_sim(rep_instr, gt_instr))
        h_words = tokenize(rep_instr); g_words = tokenize(gt_instr)
        scores["len_ratio"].append(len(h_words)/max(len(g_words),1))
        scores["verb_match"].append(float(first_verb(rep_instr)==first_verb(gt_instr)))
        scores["stop_match"].append(float(has_stop(rep_instr)))
        scores["noun_f1"].append(noun_f1(rep_instr, gt_instr))
        ht = turn_mentions(rep_instr); gt_t = turn_mentions(gt_instr)
        scores["turn_acc"].append(turn_accuracy(ht, gt_t))

    n = len(scores["bleu1"])
    result = {k: sum(v)/len(v) for k, v in scores.items()}
    result["n_episodes"] = n
    # composite score: average of key metrics (excluding len_ratio which is already good at 1.0)
    result["composite"] = (result["bleu1"]+result["bleu2"]+result["rouge_l"]+
                           result["meteor"]+result["jaccard"]+result["noun_f1"]) / 6
    return result


def print_report(name: str, r: dict):
    print(f"\n{'='*60}")
    print(f"  {name}  (n={r['n_episodes']})")
    print(f"{'='*60}")
    print(f"  BLEU-1    {r['bleu1']:.4f}   | BLEU-2   {r['bleu2']:.4f}")
    print(f"  BLEU-4    {r['bleu4']:.4f}   | ROUGE-L  {r['rouge_l']:.4f}")
    print(f"  METEOR    {r['meteor']:.4f}   | Jaccard  {r['jaccard']:.4f}")
    print(f"  Edit-sim  {r['edit_sim']:.4f}   | Noun-F1  {r['noun_f1']:.4f}")
    print(f"  Len-ratio {r['len_ratio']:.3f}    | Turn-acc {r['turn_acc']:.4f}")
    print(f"  Verb-match {r['verb_match']:.3f}   | Stop%   {r['stop_match']:.3f}")
    print(f"  ── Composite: {r['composite']:.4f} ──")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None, help="Subsample n episodes")
    p.add_argument("--versions", nargs="+", default=None,
                   help="Specific version names to eval (e.g. v4 v5 v7)")
    args = p.parse_args()

    ds_dir = ROOT / "outputs" / "datasets"
    versions = []
    if args.versions:
        for v in args.versions:
            p2 = ds_dir / f"val_unseen_generated_gemma_visual_{v}.json.gz"
            if p2.exists(): versions.append((v, str(p2)))
            else: print(f"[SKIP] not found: {p2.name}")
    else:
        for f in sorted(ds_dir.glob("val_unseen_generated_gemma_visual_*.json.gz")):
            name = f.stem.replace("val_unseen_generated_gemma_visual_","").replace(".json","")
            versions.append((name, str(f)))

    if not versions:
        print("No versions found."); sys.exit(1)

    print(f"Evaluating {len(versions)} version(s) vs GT  (n={args.n or 'all'} eps)")
    all_results = {}
    for name, path in versions:
        print(f"  Scoring {name} ...", flush=True)
        r = eval_dataset(path, n_sample=args.n)
        all_results[name] = r
        print_report(name, r)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY (sorted by composite)")
    print(f"{'='*60}")
    fmt = "{:<8}  BLEU1={:.3f}  ROUGE={:.3f}  METEOR={:.3f}  Jacc={:.3f}  NounF1={:.3f}  Comp={:.4f}"
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]["composite"]):
        print(fmt.format(name, r["bleu1"], r["rouge_l"], r["meteor"],
                         r["jaccard"], r["noun_f1"], r["composite"]))
