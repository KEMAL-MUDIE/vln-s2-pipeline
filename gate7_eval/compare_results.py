#!/usr/bin/env python3
"""
Gate 7: Results Comparison
Compares GT eval vs generated-instruction eval, computes semantic similarity metrics.
"""
import json
import gzip
import sys
from pathlib import Path

GT_BASELINE = {
    "SR": 0.6389, "SPL": 0.5855, "OS": 0.7036, "NE": 4.027, "Count": 1839,
    "label": "GT instructions (DualVLN async)"
}

def load_result(path: str) -> dict:
    with open(path) as f:
        r = json.load(f)
    # Handle split-key wrapper
    if isinstance(r, dict) and not any(k in r for k in ("SR", "SPL", "NE")):
        r = list(r.values())[0]
    return r


def compute_semantic_similarity(generated_path: str, gt_path: str) -> dict:
    """Compute BERTScore and BLEU between generated and GT instructions."""
    try:
        import evaluate
        from bert_score import score as bert_score
    except ImportError:
        return {"error": "bert_score / evaluate not installed. Run: pip install bert-score evaluate sacrebleu"}

    with gzip.open(generated_path, "rt") as f:
        gen_data = json.load(f)
    with gzip.open(gt_path, "rt") as f:
        gt_data = json.load(f)

    gt_map = {e["episode_id"]: e["instruction"]["instruction_text"] for e in gt_data["episodes"]}
    gen_eps = gen_data["episodes"]

    cands = []
    refs = []
    for ep in gen_eps:
        eid = ep["episode_id"]
        if eid in gt_map:
            cands.append(ep["instruction"]["instruction_text"])
            refs.append(gt_map[eid])

    if not cands:
        return {"error": "No matching episodes found"}

    # BERTScore
    P, R, F1 = bert_score(cands, refs, lang="en", verbose=False)
    bertscore_f1 = float(F1.mean())

    # BLEU-4
    bleu = evaluate.load("sacrebleu")
    bleu_result = bleu.compute(predictions=cands, references=[[r] for r in refs])
    bleu4 = bleu_result["score"] / 100  # normalize to 0-1

    # METEOR
    try:
        meteor = evaluate.load("meteor")
        meteor_result = meteor.compute(predictions=cands, references=refs)
        meteor_score = meteor_result["meteor"]
    except Exception:
        meteor_score = None

    return {
        "n_episodes": len(cands),
        "bertscore_f1": round(bertscore_f1, 4),
        "bleu4": round(bleu4, 4),
        "meteor": round(meteor_score, 4) if meteor_score else None,
        "bertscore_target": 0.85,
        "bleu4_target": 0.15,
        "bertscore_pass": bertscore_f1 >= 0.85,
        "bleu4_pass": bleu4 >= 0.15,
    }


def print_comparison(results: list):
    """Pretty-print comparison table."""
    print("\n" + "=" * 75)
    print(f"{'Config':<35} {'SR':>7} {'SPL':>7} {'NE':>7} {'Count':>7}")
    print("-" * 75)
    for r in results:
        label = r.get("label", "unknown")[:34]
        sr = r.get("SR", 0) * 100
        spl = r.get("SPL", 0) * 100
        ne = r.get("NE", 0)
        count = r.get("Count", 0)
        print(f"{label:<35} {sr:>6.2f}% {spl:>6.2f}% {ne:>6.3f}m {count:>7}")
    print("=" * 75)

    # Delta vs GT
    gt = results[0]
    for r in results[1:]:
        d_sr = (r.get("SR", 0) - gt["SR"]) * 100
        d_spl = (r.get("SPL", 0) - gt["SPL"]) * 100
        d_ne = r.get("NE", 0) - gt["NE"]
        label = r.get("label", "generated")[:34]
        print(f"\nDelta [{label}] vs GT:")
        print(f"  SR:  {d_sr:+.2f}pp  |  SPL: {d_spl:+.2f}pp  |  NE: {d_ne:+.3f}m")
        gate7_pass = abs(d_sr) <= 3.0
        print(f"  Gate 7 criterion (SR ±3pp): {'PASS ✓' if gate7_pass else 'FAIL ✗'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("result_files", nargs="+", help="result.json files to compare")
    parser.add_argument("--labels", nargs="+", help="Labels for each result file")
    parser.add_argument("--semantic", action="store_true", help="Compute semantic similarity metrics")
    parser.add_argument("--generated-gz", help="Generated .json.gz (for semantic similarity)")
    parser.add_argument("--gt-gz", default="/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
    args = parser.parse_args()

    results = [GT_BASELINE]
    for i, path in enumerate(args.result_files):
        r = load_result(path)
        label = args.labels[i] if args.labels and i < len(args.labels) else Path(path).parent.name
        r["label"] = label
        results.append(r)

    print_comparison(results)

    if args.semantic and args.generated_gz:
        print("\nComputing semantic similarity metrics...")
        sim = compute_semantic_similarity(args.generated_gz, args.gt_gz)
        print(json.dumps(sim, indent=2))
