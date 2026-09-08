#!/usr/bin/env python3
"""
Generate v217 auto-annotated datasets for ALL splits: train, val_seen, val_unseen.

v217 key changes from v216:
  - bedroom excluded from Rule 3 (18 eps fixed in val_unseen — bed/bed-frame preserved)
  - train/val_seen: path-only (no gate3 VLM data) → room/hallway vocabulary
  - val_unseen: gate3 perframe/landmark → full instruction quality

Usage:
  python3 generate_v217_all_splits.py
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

VERSION = "v217"

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
        "pregenerated": OUT_DIR / f"val_unseen_auto_{VERSION}.json.gz",
    },
}


def load_vocab():
    with gzip.open(HABITAT_BASE / "val_unseen" / "val_unseen_patched.json.gz", "rt") as f:
        return json.load(f).get("instruction_vocab", {})


def generate_split(split_name: str, cfg: dict) -> list:
    if "pregenerated" in cfg and cfg["pregenerated"].exists():
        print(f"[{split_name}] Using pregenerated: {cfg['pregenerated']}")
        with gzip.open(cfg["pregenerated"], "rt") as f:
            data = json.load(f)
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
            print(f"  {i+1}/{len(episodes)} ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"[{split_name}] Done: {len(results)} eps, {elapsed:.1f}s, "
          f"init_turns={initial_turns/len(results)*100:.1f}%"
          + (f", missing_gate3={missing_gate3}" if missing_gate3 else ""))
    return results


def vocab_stats(episodes: list, split_name: str):
    texts = [e["instruction"]["instruction_text"] for e in episodes]
    n = len(texts)
    avg_words = sum(len(t.split()) for t in texts) / n

    def _pct(pat):
        return sum(1 for t in texts if re.search(pat, t, re.I)) / n * 100

    turn_around = _pct(r'^Turn around')
    turn_left = _pct(r'^Turn left')
    turn_right = _pct(r'^Turn right')
    turn_slightly = _pct(r'^Turn slightly')
    no_turn = 100 - turn_around - turn_left - turn_right - turn_slightly

    print(f"\n[{split_name}] n={n}, avg_words={avg_words:.1f}")
    print(f"  walk_through={_pct(r'walk through'):.1f}% walk_past={_pct(r'walk past'):.1f}%")
    print(f"  into_the={_pct(r'into the'):.1f}%  stop={_pct(r'\\bstop\\b'):.1f}%")
    print(f"  init_turn: around={turn_around:.1f}% left={turn_left:.1f}% "
          f"right={turn_right:.1f}% slightly={turn_slightly:.1f}% none={no_turn:.1f}%")

    return {"n": n, "avg_words": round(avg_words, 1)}


def main():
    print("=" * 70)
    print(f"MetadataReproducer {VERSION} — ALL SPLITS")
    print("v217: bedroom excluded from Rule 3, room-at-turn, door-through, stop quality")
    print("=" * 70)

    vocab = load_vocab()
    all_stats = {}

    for split_name, cfg in SPLITS.items():
        episodes = generate_split(split_name, cfg)
        stats = vocab_stats(episodes, split_name)
        all_stats[split_name] = stats

        out_data = {"episodes": episodes, "instruction_vocab": vocab}
        local_path = OUT_DIR / cfg["out_name"]
        with gzip.open(local_path, "wt") as f:
            json.dump(out_data, f)
        print(f"  Saved: {local_path} ({local_path.stat().st_size//1024} KB)")

        deploy_path = cfg["deploy_path"]
        deploy_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(deploy_path, "wt") as f:
            json.dump(out_data, f)
        print(f"  Deployed: {deploy_path}")

    print("\n" + "=" * 70)
    print(f"SUMMARY — {VERSION}")
    print("=" * 70)
    for sn, s in all_stats.items():
        print(f"  {sn:<12} n={s['n']:>6}  avg_words={s['avg_words']:.1f}")

    print(f"\nAll {VERSION} splits ready for training/eval.")

    # Create tarball
    import subprocess
    tarball = OUT_DIR / f"vln_r2r_auto_{VERSION}_all_splits.tar.gz"
    files = [str(OUT_DIR / cfg["out_name"]) for cfg in SPLITS.values()]
    subprocess.run(["tar", "czf", str(tarball)] + files, check=True)
    print(f"\nTarball: {tarball} ({tarball.stat().st_size//1024//1024} MB)")


if __name__ == "__main__":
    main()
