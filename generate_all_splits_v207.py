#!/usr/bin/env python3
"""
Generate v207 auto-annotated datasets for ALL splits (train, val_seen, val_unseen)
and run comprehensive similarity analysis.

v207 changes:
  - Adaptive turn threshold: 45° (with gate3 visual context) | 60° (path-only)
  - val_unseen: uses gate3 perframe/landmark → 45° threshold
  - val_seen + train: path-only (no gate3) → 60° threshold

Usage:
  PYTHONHASHSEED=42 python3 generate_all_splits_v207.py
"""

import gzip, json, re, time, sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT))

# Import the fully updated metadata_reproducer (already has v207 adaptive threshold)
from metadata_reproducer import reproduce_instruction, _make_rng

HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
OUT_DIR = PIPELINE_ROOT / "outputs" / "datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": {
        "gt_path": HABITAT_BASE / "train" / "train.json.gz",
        "has_gate3": False,
        "out_name": "train_auto_v207.json.gz",
        "deploy_dir": HABITAT_BASE / "train",
    },
    "val_seen": {
        "gt_path": HABITAT_BASE / "val_seen" / "val_seen.json.gz",
        "has_gate3": False,
        "out_name": "val_seen_auto_v207.json.gz",
        "deploy_dir": HABITAT_BASE / "val_seen",
    },
    "val_unseen": {
        # Use existing v207 if already generated, else use patched GT
        "gt_path": HABITAT_BASE / "val_unseen" / "val_unseen_patched.json.gz",
        "has_gate3": True,
        "out_name": "val_unseen_auto_v207.json.gz",
        "deploy_dir": HABITAT_BASE / "val_unseen",
        "pregenerated": PIPELINE_ROOT / "outputs" / "datasets" / "val_unseen_generated_meta_v207.json.gz",
    },
}

PERFRAME_DIR = PIPELINE_ROOT / "outputs" / "gate3_perframe"
LANDMARK_DIR = PIPELINE_ROOT / "outputs" / "gate3_landmarks"


def generate_split(split_name: str, cfg: dict, version: str = "v207") -> list:
    """Generate auto-annotated instructions for a split. Returns list of episodes."""
    # Check for pregenerated file
    if "pregenerated" in cfg and cfg["pregenerated"].exists():
        print(f"\n[{split_name}] Using pregenerated: {cfg['pregenerated']}")
        with gzip.open(cfg["pregenerated"], "rt") as f:
            data = json.load(f)
        return data["episodes"]

    print(f"\n[{split_name}] Loading GT from {cfg['gt_path']}...")
    t0 = time.time()
    with gzip.open(cfg["gt_path"], "rt") as f:
        gt = json.load(f)

    episodes = gt["episodes"]
    print(f"[{split_name}] {len(episodes)} GT episodes loaded.")

    results = []
    missing_gate3 = 0

    for i, ep in enumerate(episodes):
        eid = ep.get("episode_id", i)

        if cfg["has_gate3"]:
            pf_path = PERFRAME_DIR / f"episode_{eid:06d}.json"
            lm_path = LANDMARK_DIR / f"episode_{eid:06d}.json"
            perframe = json.loads(pf_path.read_text()) if pf_path.exists() else {}
            landmark = json.loads(lm_path.read_text()) if lm_path.exists() else {}
            if not perframe:
                missing_gate3 += 1
        else:
            perframe = {}
            landmark = {}

        rng = _make_rng(eid)
        instr = reproduce_instruction(
            episode_id=eid,
            reference_path=ep["reference_path"],
            start_rotation=ep.get("start_rotation"),
            perframe=perframe,
            landmark=landmark,
            rng=rng,
        )

        out_ep = dict(ep)
        out_ep["instruction"] = {
            "instruction_text": instr,
            "_source": f"metadata_reproducer:{version}",
            "_gt_instruction": ep["instruction"]["instruction_text"],
        }
        results.append(out_ep)

        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(episodes)} done ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"[{split_name}] Generated {len(results)} episodes in {elapsed:.1f}s")
    if missing_gate3:
        print(f"[{split_name}] WARNING: {missing_gate3} episodes missing gate3 data (path-only fallback)")

    return results


def vocab_analysis(episodes: list, split_name: str, gt_label: str = "GT"):
    """Print vocabulary statistics."""
    n = len(episodes)
    if n == 0:
        return

    def _pct(pat):
        return sum(1 for e in episodes
                   if re.search(pat, e["instruction"]["instruction_text"], re.I)) / n * 100

    avg_words = sum(len(e["instruction"]["instruction_text"].split()) for e in episodes) / n

    sent_counts = {}
    for e in episodes:
        sents = re.split(r'(?<=[.!?])\s+', e["instruction"]["instruction_text"].strip())
        k = len(sents)
        sent_counts[k] = sent_counts.get(k, 0) + 1
    avg_sents = sum(k * v for k, v in sent_counts.items()) / n

    print(f"\n  [{split_name} vocab]  n={n}")
    print(f"    avg_words: {avg_words:.1f}  avg_sents: {avg_sents:.2f}")
    print(f"    walk_past: {_pct(r'walk past'):.1f}%  walk_toward: {_pct(r'walk toward'):.1f}%")
    print(f"    go_past:   {_pct(r'go past'):.1f}%   pass_the:   {_pct(r'pass the'):.1f}%")
    print(f"    continue:  {_pct('continue'):.1f}%   stop:       {_pct(r'\bstop\b'):.1f}%")
    print(f"    wait:      {_pct(r'\bwait\b'):.1f}%   turn:       {_pct(r'\bturn\b'):.1f}%")
    print(f"    sent_dist: {dict(sorted(sent_counts.items()))}")


def similarity_analysis(episodes: list, split_name: str):
    """Compute word-level F1, BLEU-1, ROUGE-1R vs GT instructions."""
    f1s, bleu1s, rouge_rs = [], [], []
    turn_over = 0  # auto says turn when GT doesn't
    turn_under = 0  # GT says turn when auto doesn't
    turn_exact = 0  # both agree (both mention or both don't)
    n = len(episodes)

    for ep in episodes:
        auto_text = ep["instruction"]["instruction_text"].lower()
        gt_text = ep["instruction"].get("_gt_instruction", "").lower()
        if not gt_text:
            continue

        auto_words = re.findall(r"\b\w+\b", auto_text)
        gt_words = re.findall(r"\b\w+\b", gt_text)
        auto_set = set(auto_words)
        gt_set = set(gt_words)

        # Word F1
        if gt_set and auto_set:
            prec = len(auto_set & gt_set) / len(auto_set)
            rec = len(auto_set & gt_set) / len(gt_set)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            f1s.append(f1)
        else:
            f1s.append(0.0)

        # BLEU-1 (precision of auto words in GT)
        if auto_words:
            gt_counter = {}
            for w in gt_words:
                gt_counter[w] = gt_counter.get(w, 0) + 1
            auto_counter = {}
            for w in auto_words:
                auto_counter[w] = auto_counter.get(w, 0) + 1
            clipped = sum(min(c, gt_counter.get(w, 0)) for w, c in auto_counter.items())
            bleu1s.append(clipped / len(auto_words))
        else:
            bleu1s.append(0.0)

        # ROUGE-1 Recall (coverage of GT words by auto)
        if gt_words:
            gt_counter = {}
            for w in gt_words:
                gt_counter[w] = gt_counter.get(w, 0) + 1
            auto_counter = {}
            for w in auto_words:
                auto_counter[w] = auto_counter.get(w, 0) + 1
            matched = sum(min(c, auto_counter.get(w, 0)) for w, c in gt_counter.items())
            rouge_rs.append(matched / len(gt_words))
        else:
            rouge_rs.append(0.0)

        # Turn over/under detection
        auto_has_turn = bool(re.search(r'\b(turn|left|right|make a)\b', auto_text))
        gt_has_turn = bool(re.search(r'\b(turn|left|right|make a)\b', gt_text))
        if auto_has_turn and not gt_has_turn:
            turn_over += 1
        elif gt_has_turn and not auto_has_turn:
            turn_under += 1
        else:
            turn_exact += 1

    total = len(f1s)
    print(f"\n  [{split_name} similarity]  n={total}")
    print(f"    F1:       {sum(f1s)/total:.3f}")
    print(f"    BLEU-1:   {sum(bleu1s)/total:.3f}  (auto precision in GT vocab)")
    print(f"    ROUGE-1R: {sum(rouge_rs)/total:.3f}  (GT coverage by auto)")
    over_pct = turn_over / total * 100
    under_pct = turn_under / total * 100
    exact_pct = turn_exact / total * 100
    print(f"    Turn: over={over_pct:.1f}% under={under_pct:.1f}% exact={exact_pct:.1f}%")

    return {"f1": sum(f1s)/total, "bleu1": sum(bleu1s)/total, "rouge_r": sum(rouge_rs)/total,
            "turn_over_pct": over_pct, "turn_under_pct": under_pct, "turn_exact_pct": exact_pct}


def save_split(episodes: list, out_name: str, deploy_dir: Path | None = None, version: str = "v207"):
    """Save dataset to outputs/ and optionally deploy to Habitat path."""
    local_path = OUT_DIR / out_name
    out_data = {"episodes": episodes}
    with gzip.open(local_path, "wt") as f:
        json.dump(out_data, f)
    size_kb = local_path.stat().st_size // 1024
    print(f"  [save] {local_path} ({len(episodes)} eps, {size_kb}KB)")

    if deploy_dir:
        deploy_path = deploy_dir / out_name
        with gzip.open(deploy_path, "wt") as f:
            json.dump(out_data, f)
        print(f"  [deploy] {deploy_path}")

    return local_path


def main():
    print("=" * 70)
    print("MetadataReproducer v207 — ALL SPLITS GENERATION")
    print("Adaptive threshold: 45° (with gate3) | 60° (path-only)")
    print("=" * 70)

    all_results = {}
    all_metrics = {}

    for split_name, cfg in SPLITS.items():
        print(f"\n{'='*50}")
        print(f"SPLIT: {split_name.upper()}")
        print(f"{'='*50}")

        episodes = generate_split(split_name, cfg)
        all_results[split_name] = episodes

        vocab_analysis(episodes, split_name)
        metrics = similarity_analysis(episodes, split_name)
        all_metrics[split_name] = metrics

        # Save locally + deploy to Habitat
        save_split(episodes, cfg["out_name"], cfg.get("deploy_dir"), version="v207")

    # ── Cross-split summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-SPLIT SIMILARITY SUMMARY (v207 Auto vs GT)")
    print("=" * 70)
    header = f"{'Split':<12} {'N':>6} {'F1':>6} {'BLEU-1':>8} {'ROUGE-R':>8} {'Over%':>7} {'Under%':>7} {'Exact%':>7}"
    print(header)
    print("-" * len(header))

    split_sizes = {
        "train": len(all_results["train"]),
        "val_seen": len(all_results["val_seen"]),
        "val_unseen": len(all_results["val_unseen"]),
    }

    for sn, m in all_metrics.items():
        n = split_sizes[sn]
        print(f"{sn:<12} {n:>6} {m['f1']:>6.3f} {m['bleu1']:>8.3f} {m['rouge_r']:>8.3f} "
              f"{m['turn_over_pct']:>7.1f} {m['turn_under_pct']:>7.1f} {m['turn_exact_pct']:>7.1f}")

    print("\nNotes:")
    print("  Over%  = auto says 'turn' when GT doesn't (over-detection)")
    print("  Under% = GT says 'turn' when auto doesn't (under-detection)")
    print("  Exact% = both agree on presence/absence of turn language")
    print("\nSR Prediction:")
    print("  val_seen GT baseline: ~79.70%")
    print("  val_seen auto v207 (path-only, 60° thresh): predicted 65-75%")
    print("  val_unseen auto v207 (gate3+45° thresh): predicted 60-64%")
    print("  Gemma val_unseen: 62.12% (reference ceiling)")
    print("\nDeployed datasets:")
    for split_name, cfg in SPLITS.items():
        if cfg.get("deploy_dir"):
            print(f"  {split_name}: {cfg['deploy_dir'] / cfg['out_name']}")

    # Save metrics JSON
    metrics_path = OUT_DIR / "v207_cross_split_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "version": "v207",
            "splits": {
                sn: {"n": split_sizes[sn], **m}
                for sn, m in all_metrics.items()
            }
        }, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
