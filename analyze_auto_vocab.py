#!/usr/bin/env python3
"""
Quick vocab analysis for auto-annotated instructions.
Computes key metrics vs GT to predict SR.

Usage:
  python3 analyze_auto_vocab.py outputs/auto_metadata/
  python3 analyze_auto_vocab.py outputs/auto_metadata/ --dataset outputs/auto_metadata/val_unseen_auto_2_0-auto.json.gz
"""
import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Dict, List


GT_TARGETS = {
    "through_the":      27.2,   # % of episodes containing "through the"
    "hallway":          20.4,
    "into_hallway":      4.4,   # "into the hallway" / "into the hall"
    "continue":         10.4,   # "continue"
    "walk_past":         8.7,   # "walk past"
    "go_straight":       7.94,  # "go straight" (extended metric)
    "walk_forward":      6.09,
    "walk_into_the":     6.25,
    "take_a_few_steps":  0.22,
    "walk_down_the":     8.54,
    "toward_the":        3.75,
    "avg_words":        26.8,   # average word count
}

KEY_FIVE = ["through_the", "hallway", "into_hallway", "continue", "walk_past"]

PRED_COEFF = 0.610   # SR = 63.89 - 0.610 × vocab_gap
GT_SR = 63.89


def episode_phrases(text: str) -> Dict[str, bool]:
    t = text.lower()
    return {
        "through_the":      bool(re.search(r'\bthrough the\b', t)),
        "hallway":          bool(re.search(r'\bhall(?:way)?\b', t)),
        "into_hallway":     bool(re.search(r'\binto the hall(?:way)?\b', t)),
        "continue":         bool(re.search(r'\bcontinue\b', t)),
        "walk_past":        bool(re.search(r'\bwalk past\b', t)),
        "go_straight":      bool(re.search(r'\bgo straight\b', t)),
        "walk_forward":     bool(re.search(r'\bwalk forward\b', t)),
        "walk_into_the":    bool(re.search(r'\bwalk into the\b', t)),
        "take_a_few_steps": bool(re.search(r'\btake a few steps\b', t)),
        "walk_down_the":    bool(re.search(r'\bwalk down the\b', t)),
        "toward_the":       bool(re.search(r'\btoward the\b', t)),
    }


def analyze(instructions: List[str]) -> Dict[str, float]:
    n = len(instructions)
    if n == 0:
        return {}
    counts = {k: 0 for k in GT_TARGETS if k != "avg_words"}
    total_words = 0
    for text in instructions:
        phrases = episode_phrases(text)
        for k, v in phrases.items():
            if k in counts and v:
                counts[k] += 1
        total_words += len(text.split())

    result = {k: 100 * v / n for k, v in counts.items()}
    result["avg_words"] = total_words / n
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata_dir", type=Path, nargs="?",
                    default=Path(__file__).parent / "outputs" / "auto_metadata")
    ap.add_argument("--dataset", type=Path, default=None,
                    help="Optionally analyze from .json.gz dataset file directly")
    args = ap.parse_args()

    instructions = []

    if args.dataset and args.dataset.exists():
        print(f"Loading from dataset: {args.dataset}")
        with gzip.open(args.dataset, "rt") as f:
            data = json.load(f)
        instructions = [ep["instruction"]["instruction_text"] for ep in data["episodes"]]
    elif args.metadata_dir.exists():
        print(f"Loading from metadata dir: {args.metadata_dir}")
        for p in sorted(args.metadata_dir.glob("episode_*.json")):
            try:
                d = json.load(open(p))
                text = d.get("generated_instruction", {})
                if isinstance(text, dict):
                    text = text.get("text", "")
                if text:
                    instructions.append(text)
            except Exception:
                continue
    else:
        print("ERROR: No data found.")
        return

    print(f"Analyzing {len(instructions)} instructions...")
    metrics = analyze(instructions)

    print(f"\n{'Metric':<22} {'Generated':>12} {'GT':>12} {'Gap':>10}")
    print("-" * 58)

    vocab_gap_5 = 0.0
    for key in list(GT_TARGETS.keys()):
        if key not in metrics:
            continue
        gt = GT_TARGETS[key]
        gen = metrics[key]
        if key == "avg_words":
            gap = gen - gt
            flag = "✓" if abs(gap) < 1.5 else "✗"
            print(f"  {'avg_words':<20} {gen:>12.1f} {gt:>12.1f} {gap:>+9.1f} {flag}")
        else:
            gap = abs(gen - gt)
            if key in KEY_FIVE:
                vocab_gap_5 += gap
                flag = "✓" if gap < 1.0 else ("~" if gap < 3.0 else "✗")
            else:
                flag = ""
            print(f"  {key:<20} {gen:>12.2f}% {gt:>11.2f}% {gap:>+10.2f}pp {flag}")

    pred_sr = GT_SR - PRED_COEFF * vocab_gap_5
    print(f"\n  5-metric vocab_gap:  {vocab_gap_5:.2f}pp")
    print(f"  Predicted SR:        {pred_sr:.2f}%  (GT={GT_SR}%)")
    print(f"\n  Target: pred_SR ≥ 60% → {'✓ DEPLOY' if pred_sr >= 60 else '✗ DO NOT DEPLOY'}")
    print(f"  Target: pred_SR ≥ 65% → {'✓ EXCELLENT' if pred_sr >= 65 else '  not reached'}")

    # Show sample instructions
    print(f"\n=== Sample Instructions (first 3) ===")
    for i, text in enumerate(instructions[:3]):
        print(f"  [{i+1}] {text}")

    return metrics


if __name__ == "__main__":
    main()
