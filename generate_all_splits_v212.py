#!/usr/bin/env python3
"""
Generate v212 auto-annotated datasets for ALL splits: train, val_seen, val_unseen.

v212 key improvement: initial turn detection from start_rotation.
  - 82.2% of episodes have initial agent-path misalignment >20°
  - Instructions now start with "Turn left/right/around and ..." where needed
  - avg_words=27.0 matches GT 26.8 almost exactly
  - F1 improved from 0.318 → 0.331

Split modes:
  - val_unseen: gate3 perframe/landmark available → rich visual landmark instructions
  - val_seen: path-only (no gate3 VLM data) → room/hallway vocabulary
  - train: path-only (no gate3 VLM data) → room/hallway vocabulary

Output naming: train_auto_v212, val_seen_auto_v212, val_unseen_auto_v212

Usage:
  python3 generate_all_splits_v212.py
"""

import gzip, json, re, time, sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT))

from metadata_reproducer import reproduce_instruction, _make_rng

HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
OUT_DIR = PIPELINE_ROOT / "outputs" / "datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERFRAME_DIR = PIPELINE_ROOT / "outputs" / "gate3_perframe"
LANDMARK_DIR = PIPELINE_ROOT / "outputs" / "gate3_landmarks"

VERSION = "v212"

SPLITS = {
    "train": {
        "gt_path": HABITAT_BASE / "train" / "train.json.gz",
        "has_gate3": False,
        "out_name": f"train_auto_{VERSION}.json.gz",
        "deploy_path": HABITAT_BASE / "train" / f"train_auto_{VERSION}.json.gz",
    },
    "val_seen": {
        "gt_path": HABITAT_BASE / "val_seen" / "val_seen.json.gz",
        "has_gate3": False,
        "out_name": f"val_seen_auto_{VERSION}.json.gz",
        "deploy_path": HABITAT_BASE / "val_seen" / f"val_seen_auto_{VERSION}.json.gz",
    },
    "val_unseen": {
        "gt_path": HABITAT_BASE / "val_unseen" / "val_unseen_patched.json.gz",
        "has_gate3": True,
        "out_name": f"val_unseen_auto_{VERSION}.json.gz",
        "deploy_path": HABITAT_BASE / "val_unseen" / f"val_unseen_auto_{VERSION}.json.gz",
        # Already generated — use pregenerated to keep RNG consistent
        "pregenerated": PIPELINE_ROOT / "outputs" / "datasets" / f"val_unseen_generated_meta_{VERSION}.json.gz",
    },
}


def load_instruction_vocab():
    """Load instruction_vocab from val_unseen_patched (authoritative source)."""
    vocab_path = HABITAT_BASE / "val_unseen" / "val_unseen_patched.json.gz"
    with gzip.open(vocab_path, "rt") as f:
        data = json.load(f)
    return data.get("instruction_vocab", {})


def generate_split(split_name: str, cfg: dict) -> list:
    """Generate auto-annotated instructions for a split. Returns episode list."""
    # val_unseen: use pregenerated to keep RNG/seed consistent with standalone run
    if "pregenerated" in cfg and cfg["pregenerated"].exists():
        print(f"[{split_name}] Using pregenerated: {cfg['pregenerated']}")
        with gzip.open(cfg["pregenerated"], "rt") as f:
            data = json.load(f)
        # Strip debug keys from instruction dict
        for ep in data["episodes"]:
            ep["instruction"] = {"instruction_text": ep["instruction"]["instruction_text"]}
        return data["episodes"]

    print(f"\n[{split_name}] Loading GT from {cfg['gt_path']} ...")
    t0 = time.time()
    with gzip.open(cfg["gt_path"], "rt") as f:
        gt = json.load(f)

    episodes = gt["episodes"]
    print(f"[{split_name}] {len(episodes)} episodes")

    results = []
    missing_gate3 = 0
    initial_turns = 0

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

        if instr.startswith("Turn"):
            initial_turns += 1

        out_ep = dict(ep)
        out_ep["instruction"] = {"instruction_text": instr}
        results.append(out_ep)

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(episodes)} done ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"[{split_name}] Generated {len(results)} episodes in {elapsed:.1f}s")
    print(f"  initial_turns: {initial_turns} ({initial_turns/len(results)*100:.1f}%)")
    if missing_gate3:
        print(f"  WARNING: {missing_gate3} episodes missing gate3 data → path-only fallback")

    return results


def vocab_stats(episodes: list, split_name: str):
    """Print vocabulary statistics for a split."""
    n = len(episodes)
    if n == 0:
        return {}

    def _pct(pat):
        return sum(1 for e in episodes
                   if re.search(pat, e["instruction"]["instruction_text"], re.I)) / n * 100

    texts = [e["instruction"]["instruction_text"] for e in episodes]
    avg_words = sum(len(t.split()) for t in texts) / n

    sent_counts = {}
    for t in texts:
        k = len(re.split(r'(?<=[.!?])\s+', t.strip()))
        sent_counts[k] = sent_counts.get(k, 0) + 1
    avg_sents = sum(k * v for k, v in sent_counts.items()) / n

    turn_around = sum(1 for t in texts if t.startswith("Turn around")) / n * 100
    turn_left = sum(1 for t in texts if t.startswith("Turn left")) / n * 100
    turn_right = sum(1 for t in texts if t.startswith("Turn right")) / n * 100
    turn_slightly = sum(1 for t in texts if t.startswith("Turn slightly")) / n * 100
    no_turn = 100 - turn_around - turn_left - turn_right - turn_slightly

    print(f"\n[{split_name} vocab] n={n}")
    print(f"  avg_words: {avg_words:.1f}  avg_sents: {avg_sents:.2f}")
    print(f"  walk_past: {_pct(r'walk past'):.1f}%  turn: {_pct(r'\bturn\b'):.1f}%")
    print(f"  stop: {_pct(r'\bstop\b'):.1f}%  wait: {_pct(r'\bwait\b'):.1f}%")
    print(f"  continue: {_pct('continue'):.1f}%  hallway: {_pct('hallway'):.1f}%")
    print(f"  Initial turn prefix: around={turn_around:.1f}% left={turn_left:.1f}% right={turn_right:.1f}% slightly={turn_slightly:.1f}% none={no_turn:.1f}%")
    print(f"  sent_dist: {dict(sorted(sent_counts.items()))}")

    return {
        "n": n, "avg_words": round(avg_words, 1), "avg_sents": round(avg_sents, 2),
        "walk_past_pct": round(_pct(r'walk past'), 1),
        "turn_pct": round(_pct(r'\bturn\b'), 1),
        "stop_pct": round(_pct(r'\bstop\b'), 1),
        "wait_pct": round(_pct(r'\bwait\b'), 1),
        "initial_turn_pct": round(100 - no_turn, 1),
    }


def save_and_deploy(episodes: list, cfg: dict, vocab: dict):
    """Save to outputs/ and deploy to Habitat NVMe path, both with instruction_vocab."""
    out_data = {"episodes": episodes, "instruction_vocab": vocab}

    # Local save
    local_path = OUT_DIR / cfg["out_name"]
    with gzip.open(local_path, "wt") as f:
        json.dump(out_data, f)
    size_kb = local_path.stat().st_size // 1024
    print(f"  Saved: {local_path} ({len(episodes)} eps, {size_kb} KB)")

    # Deploy to Habitat NVMe
    deploy_path = cfg["deploy_path"]
    deploy_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(deploy_path, "wt") as f:
        json.dump(out_data, f)
    print(f"  Deployed: {deploy_path}")

    return local_path


def main():
    print("=" * 70)
    print(f"MetadataReproducer {VERSION} — ALL SPLITS GENERATION")
    print("v212: initial turn fix, avg_words≈27.0 matches GT 26.8")
    print("=" * 70)

    vocab = load_instruction_vocab()
    print(f"Loaded instruction_vocab: {len(vocab.get('word_list', []))} words")

    all_stats = {}

    for split_name, cfg in SPLITS.items():
        print(f"\n{'='*50}")
        print(f"SPLIT: {split_name.upper()}")
        print(f"{'='*50}")

        episodes = generate_split(split_name, cfg)
        stats = vocab_stats(episodes, split_name)
        all_stats[split_name] = stats
        save_and_deploy(episodes, cfg, vocab)

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY — {VERSION}")
    print("=" * 70)
    print(f"{'Split':<12} {'N':>6} {'avg_words':>10} {'turn%':>7} {'init_turn%':>11}")
    print("-" * 50)
    for sn, s in all_stats.items():
        print(f"{sn:<12} {s['n']:>6} {s['avg_words']:>10.1f} {s['turn_pct']:>7.1f} {s['initial_turn_pct']:>11.1f}")

    print("\nDeployed to:")
    for split_name, cfg in SPLITS.items():
        print(f"  {split_name}: {cfg['deploy_path']}")

    print(f"\nAll splits generated with {VERSION}. Ready for training/eval.")


if __name__ == "__main__":
    main()
