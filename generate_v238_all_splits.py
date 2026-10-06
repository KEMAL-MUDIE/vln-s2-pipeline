#!/usr/bin/env python3
"""
Generate v238 auto-annotated datasets for ALL splits: train, val_seen, val_unseen.

v238 = v237 settings applied to ALL THREE SPLITS for comprehensive calibration.

v237 improvements (all inherited):
  - turn_threshold=75° → avg_explicit_turns=0.737 (GT=0.66, NEAR PERFECT!)
  - walk_past h%3 diversification (GT=1.12x, was 3.7x)
  - stop_at_the h%3 diversification (GT=1.71x, was 4.52x)
  - 75% turn opener strip: turn_openers=20.2% (GT=16.5%)
    + start_room prefix for val_unseen (has gate3 perframe)
  - All v222-v231 inherited post-processing

v238 NEW:
  - Runs on train (10,819 eps) + val_seen (783 eps) + val_unseen (1,839 eps)
  - train/val_seen: path-only (no gate3 perframe/landmark) → opener strip w/o prefix
  - val_unseen: reuses existing v237 output for RNG consistency

Output naming: train_auto_v238, val_seen_auto_v238, val_unseen_auto_v238

Usage:
  python3 generate_v238_all_splits.py
"""

import gzip, json, re, time, sys, hashlib
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT))
from metadata_reproducer import reproduce_instruction, _make_rng

HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
OUT_DIR = PIPELINE_ROOT / "outputs" / "datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERFRAME_DIR = PIPELINE_ROOT / "outputs" / "gate3_perframe"
LANDMARK_DIR = PIPELINE_ROOT / "outputs" / "gate3_landmarks"

VERSION = "v238"

SPLITS = {
    "val_unseen": {
        "gt_path": HABITAT_BASE / "val_unseen" / "val_unseen_patched.json.gz",
        "has_gate3": True,
        "pregenerated": OUT_DIR / "val_unseen_auto_v237.json.gz",
        "out_name": f"val_unseen_auto_{VERSION}.json.gz",
        "deploy_path": HABITAT_BASE / "val_unseen" / f"val_unseen_auto_{VERSION}.json.gz",
    },
    "val_seen": {
        "gt_path": HABITAT_BASE / "val_seen" / "val_seen.json.gz",
        "has_gate3": False,
        "out_name": f"val_seen_auto_{VERSION}.json.gz",
        "deploy_path": HABITAT_BASE / "val_seen" / f"val_seen_auto_{VERSION}.json.gz",
    },
    "train": {
        "gt_path": HABITAT_BASE / "train" / "train.json.gz",
        "has_gate3": False,
        "out_name": f"train_auto_{VERSION}.json.gz",
        "deploy_path": HABITAT_BASE / "train" / f"train_auto_{VERSION}.json.gz",
    },
}

TURN_OPENER_RE = re.compile(
    r'^Turn (?:around|slightly )?(?:left|right)? ?and (\w)',
    re.IGNORECASE
)


def apply_v237_postproc(instr: str, eid: int, perframe: dict) -> tuple:
    """Apply all v237 post-processing chain. Returns (instr, counters_dict)."""
    ct_conv = wt_conv = opener_conv = 0

    # v222 inherited: "turn X into the room" → "walk into the room"
    instr = re.sub(r'\bTurn (?:left|right) into the\b', 'Walk into the', instr)
    instr = re.sub(r'\bturn (?:left|right) into the\b', 'walk into the', instr)

    # v234: 75% turn opener strip with start_room prefix (if available)
    if TURN_OPENER_RE.match(instr):
        _hash = int(hashlib.md5(f"{eid}_v233_opener".encode()).hexdigest(), 16)
        if _hash % 4 != 0:
            start_room = perframe.get("start", {}).get("room", "") if perframe else ""
            if start_room:
                instr = TURN_OPENER_RE.sub(
                    lambda m: "From the " + start_room + ", " + m.group(1).lower(), instr
                )
            else:
                instr = TURN_OPENER_RE.sub(lambda m: m.group(1).upper(), instr)
            opener_conv += 1

    # v225: "toward the" → "to the"
    instr = re.sub(r'\btoward the\b', 'to the', instr)

    # v226: hash%2==0 "Walk into the" → "Enter the"
    _hash = int(hashlib.md5(f"{eid}_v226_enter".encode()).hexdigest(), 16)
    if _hash % 2 == 0:
        instr = re.sub(r'\bWalk into the\b', 'Enter the', instr)
        instr = re.sub(r'\bwalk into the\b', 'enter the', instr)

    # v227: "Walk out of the" → "Exit the"
    instr = re.sub(r'\bWalk out of the\b', 'Exit the', instr)
    instr = re.sub(r'\bwalk out of the\b', 'exit the', instr)

    # v228: hallway through → hallway down
    instr = re.sub(r'\bwalk through the hallway\b', 'walk down the hallway', instr, flags=re.IGNORECASE)
    instr = re.sub(r'\bWalk through the hallway\b', 'Walk down the hallway', instr)
    instr = re.sub(r'\bgo through the hallway\b', 'go down the hallway', instr, flags=re.IGNORECASE)
    instr = re.sub(r'\bGo through the hallway\b', 'Go down the hallway', instr)

    # v228: "go out of the" → "exit the"
    instr = re.sub(r'\bGo out of the\b', 'Exit the', instr)
    instr = re.sub(r'\bgo out of the\b', 'exit the', instr)

    # v228: hash%2==0 "when you reach" → "once you reach"
    _hash = int(hashlib.md5(f"{eid}_v228_reach".encode()).hexdigest(), 16)
    if _hash % 2 == 0:
        instr = re.sub(r'\bwhen you reach the\b', 'once you reach the', instr, flags=re.IGNORECASE)

    # v229: "Walk forward past" → "Walk past"
    instr = re.sub(r'\bWalk forward past\b', 'Walk past', instr)
    instr = re.sub(r'\bwalk forward past\b', 'walk past', instr)

    # v229: "continue through the hallway" → "continue down the hallway"
    instr = re.sub(r'\bcontinue through the hallway\b', 'continue down the hallway', instr, flags=re.IGNORECASE)
    instr = re.sub(r'\bContinue through the hallway\b', 'Continue down the hallway', instr)

    # v229: "stop near the" → "stop at the"
    instr = re.sub(r'\bStop near the\b', 'Stop at the', instr)
    instr = re.sub(r'\bstop near the\b', 'stop at the', instr)

    # v230: "Walk straight past" → "Walk past"
    instr = re.sub(r'\bWalk straight past\b', 'Walk past', instr)
    instr = re.sub(r'\bwalk straight past\b', 'walk past', instr)

    # v231: hash%2==0 "continue through the" → "walk through the"
    _hash = int(hashlib.md5(f"{eid}_v231_ct".encode()).hexdigest(), 16)
    if _hash % 2 == 0:
        prev = instr
        instr = re.sub(r'\bContinue through the\b', 'Walk through the', instr)
        instr = re.sub(r'\bcontinue through the\b', 'walk through the', instr)
        if instr != prev:
            ct_conv += 1

    # v231: hash%2==0 "Walk to the" → "Go to the"
    _hash = int(hashlib.md5(f"{eid}_v231_wt".encode()).hexdigest(), 16)
    if _hash % 2 == 0:
        prev = instr
        instr = re.sub(r'\bWalk to the\b', 'Go to the', instr)
        instr = re.sub(r'\bwalk to the\b', 'go to the', instr)
        if instr != prev:
            wt_conv += 1

    # v232: hash%3 "stop at the" diversification
    _hash_stop = int(hashlib.md5(f"{eid}_v232_stop".encode()).hexdigest(), 16)
    h_stop = _hash_stop % 3
    if h_stop == 0:
        instr = re.sub(r'\bStop at the\b', 'Stop in front of the', instr)
        instr = re.sub(r'\bstop at the\b', 'stop in front of the', instr)
    elif h_stop == 1:
        instr = re.sub(r'\bStop at the\b', 'Stop next to the', instr)
        instr = re.sub(r'\bstop at the\b', 'stop next to the', instr)

    # v232: hash%3 "walk past" diversification
    _hash_wp = int(hashlib.md5(f"{eid}_v232_walkpast".encode()).hexdigest(), 16)
    h_wp = _hash_wp % 3
    if h_wp == 0:
        instr = re.sub(r'\bWalk past\b', 'Go past', instr)
        instr = re.sub(r'\bwalk past\b', 'go past', instr)
    elif h_wp == 1:
        instr = re.sub(r'\bWalk past\b', 'Walk by', instr)
        instr = re.sub(r'\bwalk past\b', 'walk by', instr)

    return instr, {"ct": ct_conv, "wt": wt_conv, "opener": opener_conv}


def generate_split(split_name: str, cfg: dict, vocab: dict) -> list:
    # val_unseen: reuse v237 output for RNG/seed consistency
    if "pregenerated" in cfg and Path(cfg["pregenerated"]).exists():
        print(f"[{split_name}] Reusing pregenerated: {cfg['pregenerated']}")
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
    print(f"[{split_name}] {len(episodes)} episodes, has_gate3={cfg['has_gate3']}")

    results = []
    totals = {"ct": 0, "wt": 0, "opener": 0}
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
            turn_threshold=75.0,
        )

        instr, cnts = apply_v237_postproc(instr, eid, perframe)
        for k in totals:
            totals[k] += cnts[k]

        out_ep = dict(ep)
        out_ep["instruction"] = {"instruction_text": instr}
        results.append(out_ep)

        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(episodes)} done ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"[{split_name}] {len(results)} eps in {elapsed:.1f}s  ct={totals['ct']} wt={totals['wt']} opener={totals['opener']}")
    if missing_gate3:
        print(f"  WARNING: {missing_gate3} episodes missing gate3 perframe → path-only")

    return results


def vocab_stats(episodes: list, split_name: str) -> dict:
    n = len(episodes)
    if n == 0:
        return {}

    def pct(pat):
        return sum(1 for e in episodes
                   if re.search(pat, e["instruction"]["instruction_text"], re.I)) / n * 100

    def cnt(pat):
        texts = [e["instruction"]["instruction_text"] for e in episodes]
        return sum(len(re.findall(pat, t, re.I)) for t in texts)

    texts = [e["instruction"]["instruction_text"] for e in episodes]
    avg_words = sum(len(t.split()) for t in texts) / n

    def count_explicit_turns(text):
        return len(re.findall(r'\bturn (?:left|right|around|slightly)\b', text, re.I))

    explicit_turns = [count_explicit_turns(t) for t in texts]
    avg_turns = sum(explicit_turns) / n
    pct_3plus = sum(1 for t in explicit_turns if t >= 3) / n * 100
    turn_openers = sum(1 for t in texts if t.lower().startswith('turn'))

    print(f"\n[{split_name} stats] n={n}")
    print(f"  avg_words: {avg_words:.3f}  (GT=26.78)")
    print(f"  avg_explicit_turns: {avg_turns:.3f}  (GT=0.66)")
    print(f"  pct_3+_turns: {pct_3plus:.1f}%  (GT=3.8%)")
    print(f"  turn_openers: {turn_openers} ({turn_openers/n*100:.1f}%)  (GT=16.5%)")
    print(f"  walk_past: {cnt(r'walk past')}  go_past: {cnt(r'go past')}  walk_by: {cnt(r'walk by')}")
    print(f"  stop_at_the: {cnt(r'stop at the')}  stop_in_front: {cnt(r'stop in front of')}  stop_next_to: {cnt(r'stop next to')}")
    print(f"  walk_to_the: {cnt(r'walk to the')}  (GT=98)")
    print(f"  go_to_the: {cnt(r'go to the')}  (GT=152)")
    print(f"  continue_through: {cnt(r'continue through')}  (GT=15)")
    print(f"  walk_opener%: {pct(r'^walk'):.1f}%  go_opener%: {pct(r'^go'):.1f}%  exit_opener%: {pct(r'^exit'):.1f}%")

    return {
        "n": n, "avg_words": round(avg_words, 3), "avg_turns": round(avg_turns, 3),
        "pct_3plus": round(pct_3plus, 1), "turn_openers_pct": round(turn_openers / n * 100, 1),
    }


def save_and_deploy(episodes: list, cfg: dict, vocab: dict):
    out_data = {"episodes": episodes, "instruction_vocab": vocab}

    local_path = OUT_DIR / cfg["out_name"]
    with gzip.open(local_path, "wt") as f:
        json.dump(out_data, f)
    size_kb = local_path.stat().st_size // 1024
    print(f"  Saved: {local_path} ({len(episodes)} eps, {size_kb} KB)")

    deploy_path = Path(cfg["deploy_path"])
    deploy_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(deploy_path, "wt") as f:
        json.dump(out_data, f)
    print(f"  Deployed: {deploy_path}")


def main():
    print("=" * 70)
    print(f"MetadataReproducer {VERSION} — ALL SPLITS")
    print("v238: v237 settings (turn_threshold=75°) extended to train + val_seen + val_unseen")
    print("=" * 70)

    # Load vocab from val_unseen_patched
    vocab_path = HABITAT_BASE / "val_unseen" / "val_unseen_patched.json.gz"
    with gzip.open(vocab_path, "rt") as f:
        vocab_data = json.load(f)
    vocab = vocab_data.get("instruction_vocab", {})
    print(f"Loaded instruction_vocab: {len(vocab.get('word_list', []))} words")

    all_stats = {}

    for split_name, cfg in SPLITS.items():
        print(f"\n{'='*60}")
        print(f"SPLIT: {split_name.upper()}")
        print(f"{'='*60}")

        episodes = generate_split(split_name, cfg, vocab)
        stats = vocab_stats(episodes, split_name)
        all_stats[split_name] = stats
        save_and_deploy(episodes, cfg, vocab)

    # Summary table
    print("\n" + "=" * 70)
    print(f"SUMMARY — {VERSION} (GT: avg_words=26.78, avg_turns=0.66, turn_openers=16.5%)")
    print("=" * 70)
    print(f"{'Split':<12} {'N':>6} {'avg_words':>10} {'avg_turns':>10} {'turn_open%':>11}")
    print("-" * 55)
    for sn, s in all_stats.items():
        print(f"{sn:<12} {s['n']:>6} {s['avg_words']:>10.3f} {s['avg_turns']:>10.3f} {s['turn_openers_pct']:>11.1f}%")

    print("\nDeployed:")
    for split_name, cfg in SPLITS.items():
        print(f"  {split_name}: {cfg['deploy_path']}")

    print(f"\nAll splits generated with {VERSION}.")


if __name__ == "__main__":
    main()
