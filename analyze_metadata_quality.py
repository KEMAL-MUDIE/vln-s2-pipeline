#!/usr/bin/env python3
"""
Metadata Quality Analyzer — cross-version + GT comparison.

Measures instruction TEXT quality across annotator versions and GT.
NOT eval metrics (SR%). Focus: linguistic properties that affect
how well the model can learn from instructions.

Usage:
  python3 analyze_metadata_quality.py                       # all splits
  python3 analyze_metadata_quality.py --split val_unseen    # one split
  python3 analyze_metadata_quality.py --report html         # save HTML report
  python3 analyze_metadata_quality.py --samples 5           # show instruction samples
"""
import argparse
import gzip
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PIPELINE_ROOT = Path(__file__).parent
OUT_TOP5 = PIPELINE_ROOT / "outputs/annotated_datasets_top5"

GT_PATHS = {
    "val_unseen": "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz",
    "val_seen":   "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_seen/val_seen.json.gz",
    "train":      "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/train/train.json.gz",
}

VERSIONS = ["v264", "v272", "v273", "v277", "v278"]

SPLIT_PREFIXES = {
    "val_unseen": "unseen",
    "val_seen": "seen",
    "train": "train",
}

ROOM_WORDS = {
    "bedroom", "kitchen", "hallway", "living", "dining", "bathroom",
    "office", "staircase", "stairs", "foyer", "lobby", "lounge",
    "corridor", "pantry", "laundry", "garage", "study",
}

LANDMARK_NOUNS = {
    "table", "chair", "couch", "sofa", "desk", "bed", "dresser",
    "bookcase", "bookshelf", "shelf", "cabinet", "counter", "island",
    "fireplace", "door", "doorway", "window", "rug", "carpet",
    "lamp", "mirror", "sink", "toilet", "bathtub", "shower",
    "refrigerator", "fridge", "stove", "oven", "microwave",
    "television", "tv", "computer", "monitor", "piano",
    "railing", "banister", "pillar", "column", "archway",
}

TURN_WORDS = {"left", "right", "turn", "turning", "veer", "pivot", "rotate"}
STOP_WORDS_NAV = {"stop", "wait", "stand", "halt", "remain", "pause", "end"}
ACTION_VERBS   = {"walk", "go", "move", "proceed", "head", "continue", "pass",
                  "enter", "exit", "cross", "navigate", "advance"}


def load_instructions(path: str, field: str = "primary") -> List[str]:
    """Load instructions from a .json.gz dataset.
    field="primary"   → instruction.instruction_text (what the model trains on)
    field="generated" → _generated_instruction.text (raw VLM output, if stored)
    """
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    if field == "generated":
        out = []
        for ep in data["episodes"]:
            t = ep.get("_generated_instruction", {}).get("text", "")
            if t and t.strip():
                out.append(t.strip())
        return out
    return [
        ep["instruction"]["instruction_text"]
        for ep in data["episodes"]
        if ep.get("instruction", {}).get("instruction_text", "").strip()
    ]


def load_meta_generated(meta_dir: Path) -> List[str]:
    """Load generated_instruction.text from per-episode meta files."""
    texts = []
    for mf in sorted(meta_dir.glob("ep_*.json")):
        try:
            with open(mf) as f:
                meta = json.load(f)
            t = meta.get("generated_instruction", {}).get("text", "").strip()
            if t:
                texts.append(t)
        except Exception:
            pass
    return texts


def analyze(instructions: List[str]) -> Dict:
    if not instructions:
        return {"n": 0}

    word_counts = [len(t.split()) for t in instructions]
    char_counts = [len(t) for t in instructions]

    # Vocabulary
    all_words = []
    for t in instructions:
        all_words.extend(re.findall(r"\b[a-z]+\b", t.lower()))
    vocab = set(all_words)
    ttr = len(vocab) / len(all_words) if all_words else 0.0

    # Content checks
    room_rate = sum(1 for t in instructions
                    if any(r in t.lower() for r in ROOM_WORDS)) / len(instructions)
    landmark_rate = sum(1 for t in instructions
                        if any(l in t.lower() for l in LANDMARK_NOUNS)) / len(instructions)
    turn_rate = sum(1 for t in instructions
                    if any(w in t.lower() for w in TURN_WORDS)) / len(instructions)
    stop_rate = sum(1 for t in instructions
                    if any(w in t.lower() for w in STOP_WORDS_NAV)) / len(instructions)
    action_rate = sum(1 for t in instructions
                      if any(w in t.lower() for w in ACTION_VERBS)) / len(instructions)

    # Sentence count (rough: split on .!?)
    sent_counts = [max(1, len(re.split(r"[.!?]+", t.strip()))) for t in instructions]

    # Landmark density: avg distinct landmark nouns per instruction
    def landmark_count(t):
        return len({w for w in re.findall(r"\b[a-z]+\b", t.lower()) if w in LANDMARK_NOUNS})
    lm_densities = [landmark_count(t) for t in instructions]

    # Top words (content words only, excl. stopwords)
    STOPWORDS = {"the", "a", "an", "and", "or", "to", "in", "at", "of", "on",
                 "with", "by", "is", "it", "you", "your", "then", "from",
                 "that", "this", "into", "up", "down", "as", "for", "are",
                 "be", "will", "can", "there", "through", "until", "past", "near"}
    content = [w for w in all_words if w not in STOPWORDS and len(w) > 2]
    top_words = [w for w, _ in Counter(content).most_common(15)]

    n = len(instructions)
    return {
        "n": n,
        "len_mean":     round(sum(word_counts) / n, 1),
        "len_median":   round(sorted(word_counts)[n // 2], 1),
        "len_min":      min(word_counts),
        "len_max":      max(word_counts),
        "char_mean":    round(sum(char_counts) / n, 1),
        "vocab_size":   len(vocab),
        "ttr":          round(ttr, 4),
        "room_rate":    round(room_rate, 3),
        "landmark_rate":round(landmark_rate, 3),
        "landmark_density_mean": round(sum(lm_densities) / n, 2),
        "turn_rate":    round(turn_rate, 3),
        "stop_rate":    round(stop_rate, 3),
        "action_rate":  round(action_rate, 3),
        "sentences_mean": round(sum(sent_counts) / n, 2),
        "top_words":    top_words,
    }


def compare_bleu1(hyp_list: List[str], ref_list: List[str]) -> float:
    """Rough BLEU-1 (unigram precision) of hyp vs ref, averaged over matched pairs."""
    if not hyp_list or not ref_list:
        return 0.0
    n = min(len(hyp_list), len(ref_list))
    scores = []
    for h, r in zip(hyp_list[:n], ref_list[:n]):
        h_words = set(re.findall(r"\b[a-z]+\b", h.lower()))
        r_words = set(re.findall(r"\b[a-z]+\b", r.lower()))
        if not h_words:
            scores.append(0.0)
            continue
        scores.append(len(h_words & r_words) / len(h_words))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def print_table(results: Dict[str, Dict], split: str, gt_stats: Dict):
    print(f"\n{'='*90}")
    print(f"  METADATA QUALITY REPORT — {split.upper()}")
    print(f"{'='*90}")
    print(f"  {'Source':<22} {'N':>5} {'Len(avg)':>8} {'Len(med)':>8} "
          f"{'Room%':>7} {'Lmrk%':>7} {'LmDens':>7} {'Turn%':>7} {'Stop%':>7} "
          f"{'Act%':>6} {'Sents':>6} {'TTR':>7}")
    print(f"  {'-'*88}")

    def row(name, s, bleu=None):
        if s.get("n", 0) == 0:
            print(f"  {name:<22}  [no data]")
            return
        b = f" BLEU1={bleu:.3f}" if bleu is not None else ""
        print(
            f"  {name:<22} {s['n']:>5} {s['len_mean']:>8} {s['len_median']:>8} "
            f"{s['room_rate']*100:>6.1f}% {s['landmark_rate']*100:>6.1f}% "
            f"{s['landmark_density_mean']:>7} {s['turn_rate']*100:>6.1f}% "
            f"{s['stop_rate']*100:>6.1f}% {s['action_rate']*100:>5.1f}% "
            f"{s['sentences_mean']:>6} {s['ttr']:>7}{b}"
        )

    row("GT (R2R human)", gt_stats)

    # ChronoNav backups if present
    for v in VERSIONS:
        prefix = SPLIT_PREFIXES.get(split, split.replace("val_", ""))
        chrono = OUT_TOP5 / f"{prefix}_{v}_chrono.json.gz"
        if chrono.exists():
            try:
                instrs = load_instructions(str(chrono))
                s = analyze(instrs)
                bleu = compare_bleu1(instrs, _gt_instrs)
                row(f"ChronoNav {v} (pre-vlm)", s, bleu)
                break  # only show one chrono example (they're similar)
            except Exception:
                pass

    print(f"  {'-'*88}")
    print(f"  --- TRAINING INSTRUCTION (instruction.instruction_text) ---")
    for v in VERSIONS:
        prefix = SPLIT_PREFIXES.get(split, split.replace("val_", ""))
        gz = OUT_TOP5 / f"{prefix}_{v}.json.gz"
        meta_dir = OUT_TOP5 / f"{prefix}_{v}_meta"
        if gz.exists():
            try:
                instrs = load_instructions(str(gz), field="primary")
                s = analyze(instrs)
                bleu = compare_bleu1(instrs, _gt_instrs)
                row(f"  {v} (train)", s, bleu)
            except Exception as e:
                print(f"  {f'  {v} (train)':<22}  [error: {e}]")
        elif meta_dir.exists():
            n_meta = len(list(meta_dir.glob("ep_*.json")))
            print(f"  {'  '+v+' (train)':<22}  [in progress: {n_meta} done]")
        else:
            print(f"  {'  '+v+' (train)':<22}  [not yet annotated]")
    print(f"  --- GENERATED VLM INSTRUCTION (_generated_instruction.text) ---")
    for v in VERSIONS:
        prefix = SPLIT_PREFIXES.get(split, split.replace("val_", ""))
        gz = OUT_TOP5 / f"{prefix}_{v}.json.gz"
        if gz.exists():
            try:
                gen_instrs = load_instructions(str(gz), field="generated")
                if gen_instrs:
                    sg = analyze(gen_instrs)
                    bleu_g = compare_bleu1(gen_instrs, _gt_instrs)
                    row(f"  {v} (VLM gen)", sg, bleu_g)
                else:
                    print(f"  {'  '+v+' (VLM gen)':<22}  [no _generated_instruction field]")
            except Exception as e:
                print(f"  {f'  {v} (VLM gen)':<22}  [error: {e}]")

    print()
    print("  Column guide:")
    print("    Len(avg/med) : words per instruction")
    print("    Room%        : % instructions mentioning a room type (kitchen/hallway/...)")
    print("    Lmrk%        : % instructions mentioning a physical landmark (table/door/...)")
    print("    LmDens       : avg distinct landmark nouns per instruction")
    print("    Turn%        : % instructions with directional turn words (left/right/turn)")
    print("    Stop%        : % instructions with a stop condition (stop/wait/stand)")
    print("    Act%         : % instructions with an action verb (walk/go/proceed)")
    print("    Sents        : avg sentences per instruction")
    print("    TTR          : type-token ratio (vocabulary diversity)")
    print("    BLEU1        : unigram overlap with GT (style similarity, not quality)")


def print_samples(split: str, n: int = 5):
    prefix = SPLIT_PREFIXES.get(split, split.replace("val_", ""))
    print(f"\n{'='*90}")
    print(f"  INSTRUCTION SAMPLES — {split.upper()} (first {n} episodes)")
    print(f"{'='*90}")

    # GT
    gt_path = GT_PATHS.get(split)
    if gt_path and Path(gt_path).exists():
        gt_instrs = load_instructions(gt_path)[:n]
        print(f"\n  [GT R2R human-written]")
        for i, t in enumerate(gt_instrs):
            print(f"  ep{i+1}: {t}")

    # VLM versions
    for v in VERSIONS:
        gz = OUT_TOP5 / f"{prefix}_{v}.json.gz"
        if gz.exists():
            try:
                instrs = load_instructions(str(gz))[:n]
                print(f"\n  [VLM {v}]")
                for i, t in enumerate(instrs):
                    print(f"  ep{i+1}: {t}")
            except Exception as e:
                print(f"\n  [VLM {v}]  error: {e}")

    # ChronoNav (best version)
    chrono = OUT_TOP5 / f"{prefix}_v278_chrono.json.gz"
    if chrono.exists():
        try:
            instrs = load_instructions(str(chrono))[:n]
            print(f"\n  [ChronoNav v278 (pre-VLM backup)]")
            for i, t in enumerate(instrs):
                print(f"  ep{i+1}: {t}")
        except Exception:
            pass


def print_notes(results: Dict[str, Dict], gt: Dict, split: str):
    """Qualitative quality notes comparing VLM versions to GT."""
    print(f"\n{'='*90}")
    print(f"  QUALITY NOTES — {split.upper()}")
    print(f"{'='*90}")

    gt_len = gt.get("len_mean", 20)
    gt_stop = gt.get("stop_rate", 0.7)
    gt_lmrk = gt.get("landmark_rate", 0.8)
    gt_turn = gt.get("turn_rate", 0.7)
    gt_room = gt.get("room_rate", 0.3)
    gt_ttr  = gt.get("ttr", 0.3)

    for v, s in results.items():
        if s.get("n", 0) == 0:
            continue
        print(f"\n  Version {v}:")
        notes = []

        # Length
        dl = s["len_mean"] - gt_len
        if abs(dl) <= 3:
            notes.append(f"  length={s['len_mean']}w  → CLOSE to GT ({gt_len}w), good match")
        elif dl > 3:
            notes.append(f"  length={s['len_mean']}w  → LONGER than GT ({gt_len}w) by {dl:.1f}w — more verbose, may dilute signal")
        else:
            notes.append(f"  length={s['len_mean']}w  → SHORTER than GT ({gt_len}w) by {abs(dl):.1f}w — terse, may lack context")

        # Stop condition
        stop_pct = s["stop_rate"] * 100
        gt_stop_pct = gt_stop * 100
        if stop_pct >= 90:
            notes.append(f"  stop={stop_pct:.0f}%  → EXCELLENT stop-condition coverage (GT={gt_stop_pct:.0f}%)")
        elif stop_pct >= 70:
            notes.append(f"  stop={stop_pct:.0f}%  → GOOD stop-condition coverage (GT={gt_stop_pct:.0f}%)")
        else:
            notes.append(f"  stop={stop_pct:.0f}%  → LOW stop-condition rate (GT={gt_stop_pct:.0f}%) — many episodes lack a stop cue")

        # Landmarks
        lmrk_pct = s["landmark_rate"] * 100
        gt_lmrk_pct = gt_lmrk * 100
        lm_dens = s["landmark_density_mean"]
        if lmrk_pct >= 85:
            notes.append(f"  landmark={lmrk_pct:.0f}%  density={lm_dens}  → HIGH landmark coverage (GT={gt_lmrk_pct:.0f}%) — model gets rich spatial anchors")
        elif lmrk_pct >= 65:
            notes.append(f"  landmark={lmrk_pct:.0f}%  density={lm_dens}  → MODERATE landmark coverage (GT={gt_lmrk_pct:.0f}%)")
        else:
            notes.append(f"  landmark={lmrk_pct:.0f}%  density={lm_dens}  → LOW landmark coverage (GT={gt_lmrk_pct:.0f}%) — instructions too generic")

        # Turn words
        turn_pct = s["turn_rate"] * 100
        gt_turn_pct = gt_turn * 100
        if turn_pct >= gt_turn_pct - 5:
            notes.append(f"  turns={turn_pct:.0f}%  → MATCHES GT directional language ({gt_turn_pct:.0f}%)")
        elif turn_pct < 50:
            notes.append(f"  turns={turn_pct:.0f}%  → LOW turn mention (GT={gt_turn_pct:.0f}%) — missing directional guidance for turns in path")
        else:
            notes.append(f"  turns={turn_pct:.0f}%  → BELOW GT directional coverage ({gt_turn_pct:.0f}%)")

        # Room mentions
        room_pct = s["room_rate"] * 100
        gt_room_pct = gt_room * 100
        if room_pct > gt_room_pct + 10:
            notes.append(f"  room={room_pct:.0f}%  → HIGHER than GT ({gt_room_pct:.0f}%) — VLM over-emphasizes room context")
        elif room_pct < gt_room_pct - 10:
            notes.append(f"  room={room_pct:.0f}%  → LOWER than GT ({gt_room_pct:.0f}%) — under-contextualizing environment")
        else:
            notes.append(f"  room={room_pct:.0f}%  → COMPARABLE to GT ({gt_room_pct:.0f}%) room context")

        # Vocabulary diversity
        ttr_diff = s["ttr"] - gt_ttr
        if abs(ttr_diff) < 0.02:
            notes.append(f"  TTR={s['ttr']:.3f}  → MATCHES GT vocabulary diversity ({gt_ttr:.3f})")
        elif ttr_diff > 0.02:
            notes.append(f"  TTR={s['ttr']:.3f}  → MORE diverse than GT ({gt_ttr:.3f}) — varied vocabulary, may help generalization")
        else:
            notes.append(f"  TTR={s['ttr']:.3f}  → LESS diverse than GT ({gt_ttr:.3f}) — repetitive phrasing, may hurt training")

        for note in notes:
            print(note)

    print(f"\n  Summary ranking by landmark quality (higher = better training signal):")
    ranked = sorted(
        [(v, s) for v, s in results.items() if s.get("n", 0) > 0],
        key=lambda x: (x[1]["landmark_rate"] + x[1]["stop_rate"] * 0.5), reverse=True
    )
    for rank, (v, s) in enumerate(ranked, 1):
        score = s["landmark_rate"] * 100
        stop  = s["stop_rate"] * 100
        print(f"    #{rank}  {v}:  landmark={score:.1f}%  stop={stop:.1f}%  len={s['len_mean']}w")


_gt_instrs: List[str] = []


def main():
    global _gt_instrs

    parser = argparse.ArgumentParser(description="Metadata quality analyzer")
    parser.add_argument("--split", default=None, choices=["val_unseen", "val_seen", "train"],
                        help="Analyze one split only (default: all)")
    parser.add_argument("--samples", type=int, default=0,
                        help="Print N sample instructions per version")
    parser.add_argument("--report", default=None, choices=["html"],
                        help="Save report (html)")
    args = parser.parse_args()

    splits = [args.split] if args.split else ["val_unseen", "val_seen", "train"]

    for split in splits:
        gt_path = GT_PATHS.get(split)
        if not gt_path or not Path(gt_path).exists():
            print(f"\nSkipping {split}: GT not found at {gt_path}")
            continue

        _gt_instrs = load_instructions(gt_path)
        gt_stats = analyze(_gt_instrs)

        version_results = {}
        prefix = SPLIT_PREFIXES.get(split, split.replace("val_", ""))
        for v in VERSIONS:
            gz = OUT_TOP5 / f"{prefix}_{v}.json.gz"
            if gz.exists():
                try:
                    instrs = load_instructions(str(gz))
                    version_results[v] = analyze(instrs)
                except Exception as e:
                    version_results[v] = {"n": 0, "error": str(e)}
            else:
                version_results[v] = {"n": 0}

        print_table(version_results, split, gt_stats)
        print_notes(version_results, gt_stats, split)

        if args.samples > 0:
            print_samples(split, args.samples)

    # Per-version meta stats (quality_ok rate from meta files)
    print(f"\n{'='*90}")
    print(f"  VLM GENERATION QUALITY RATES (from meta files)")
    print(f"{'='*90}")
    print(f"  {'Version':<20} {'Split':<12} {'Total':>6} {'quality_ok':>10} {'rate':>7}")
    print(f"  {'-'*58}")
    for split in splits:
        prefix = SPLIT_PREFIXES.get(split, split.replace("val_", ""))
        for v in VERSIONS:
            meta_dir = OUT_TOP5 / f"{prefix}_{v}_meta"
            if not meta_dir.exists():
                continue
            total = 0
            ok = 0
            for mf in meta_dir.glob("ep_*.json"):
                try:
                    with open(mf) as f:
                        meta = json.load(f)
                    total += 1
                    if meta.get("generated_instruction", {}).get("quality_ok", False):
                        ok += 1
                except Exception:
                    pass
            if total > 0:
                rate = ok / total * 100
                print(f"  {f'VLM {v}':<20} {split:<12} {total:>6} {ok:>10} {rate:>6.1f}%")


if __name__ == "__main__":
    main()
