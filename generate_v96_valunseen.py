#!/usr/bin/env python3
"""Generate v96: v90 with initial turns only for large rotations (>150°).

KEY INSIGHT from data analysis:
  - v32/v24 achieves 42% SR with starts_Turn=16.6% matching GT (16.5%)
  - v213/v90 achieves only 36.87% with starts_Turn=82.5% (too many initial turns)
  - v93 removes ALL initial turns → starts_Turn=1.1% (too few)
  - 21.2% of episodes have initial_turn.angle_deg > 150° (near GT's 16.5%)

v96 = v90 + remove initial turns UNLESS angle > 150° (only state true turn-arounds)
  Expected starts_Turn: ~21.2% (close to GT's 16.5% and v32's 16.6%)
  Expected SR: comparable to v32's 42%+, but from metadata_reproducer base

This tests whether the initial_turn content (GT-like rate) is the key variable,
independent of instruction source (gate4_visual vs metadata_reproducer).
"""
import gzip
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "gate2_path"))
from path_analyzer import analyze_path

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
GT_PATH = BASE / "val_unseen" / "val_unseen_patched.json.gz"
V90_PATH = BASE / "val_unseen" / "val_unseen_v90.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v96.json.gz"

# THRESHOLD: only state initial turn if |angle| > this
# 150° gives 21.2% → close to GT's 16.5% and v32's 16.6%
ANGLE_THRESHOLD_DEG = 150.0

# Pattern to detect initial turn prefix (same as v93)
INIT_TURN_PREFIX_RE = re.compile(
    r'^Turn\s+(?:slightly\s+)?(?:left|right|around)\s+and\s+'
    r'(?=walk|go|exit|leave|pass|continue|head|move|proceed|step|travel)',
    re.IGNORECASE
)


def remove_initial_turn(instruction):
    m = INIT_TURN_PREFIX_RE.match(instruction)
    if not m:
        return instruction
    rest = instruction[m.end():]
    if not rest:
        return instruction
    return rest[0].upper() + rest[1:]


def get_initial_angle(ep):
    """Return absolute initial turn angle for episode, or 0 if none."""
    path = ep.get('reference_path', [])
    qxyzw = ep.get('start_rotation', [0, 0, 0, 1])
    if len(path) < 2:
        return 0.0
    try:
        pa = analyze_path(path, qxyzw)
        it = pa.get('initial_turn')
        if it:
            return abs(it.get('angle_deg', 0))
    except Exception:
        pass
    return 0.0


def main():
    print("=== v96: v90 + selective initial turns (angle > 150° only) ===")
    print(f"  Threshold: {ANGLE_THRESHOLD_DEG}° (targets ~21% starts_Turn, GT=16.5%)")

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V90_PATH) as f:
        v90_data = json.load(f)

    # Build GT episode lookup for path data
    gt_by_eid = {ep["episode_id"]: ep for ep in gt_data["episodes"]}
    v90_by_eid = {ep["episode_id"]: ep for ep in v90_data["episodes"]}

    new_episodes = []
    removed = 0
    kept = 0
    total_had_prefix = 0

    print("Computing initial turn angles...", flush=True)
    for i, ep in enumerate(gt_data["episodes"]):
        if i % 200 == 0:
            print(f"  {i}/{len(gt_data['episodes'])}...", flush=True)

        eid = ep["episode_id"]
        v90_ep = v90_by_eid[eid]
        orig = v90_ep["instruction"]["instruction_text"]

        # Check if instruction has initial turn prefix
        has_prefix = bool(INIT_TURN_PREFIX_RE.match(orig))
        if has_prefix:
            total_had_prefix += 1
            # Get angle
            angle = get_initial_angle(ep)
            if angle <= ANGLE_THRESHOLD_DEG:
                # Remove initial turn (angle too small to state)
                cleaned = remove_initial_turn(orig)
                removed += 1
            else:
                # Keep initial turn (large rotation, worth stating)
                cleaned = orig
                kept += 1
        else:
            cleaned = orig

        new_ep = dict(v90_ep)
        new_ep["instruction"] = dict(v90_ep["instruction"])
        new_ep["instruction"]["instruction_text"] = cleaned
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    print(f"\nInitial turn prefixes: {total_had_prefix}/1839")
    print(f"  Removed (angle ≤ {ANGLE_THRESHOLD_DEG}°): {removed}")
    print(f"  Kept (angle > {ANGLE_THRESHOLD_DEG}°): {kept}")

    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]
    TURN_RE = re.compile(r'\bturn\b', re.I)
    INIT_TURN_RE = re.compile(r'^Turn', re.I)
    HALLWAY_RE = re.compile(r'\bhallway\b', re.I)

    print(f"\nQuality metrics (n={n}):")
    print(f"  avg_words={sum(words)/n:.1f}  [GT=26.8]")
    print(f"  turn%={sum(1 for i in all_insts if TURN_RE.search(i))/n*100:.1f}%  [GT=51.1%]")
    print(f"  starts_Turn={sum(1 for i in all_insts if INIT_TURN_RE.match(i))/n*100:.1f}%  [GT=16.5%, v32=16.6%]")
    print(f"  hallway={sum(1 for i in all_insts if HALLWAY_RE.search(i))/n*100:.1f}%  [GT=20.4%]")
    print(f"  unique={len(set(all_insts))/n*100:.1f}%")

    print("\nSamples (removed initial turn):")
    shown_rm = 0
    shown_kp = 0
    for ep, v90_ep in zip(new_episodes, v90_data["episodes"]):
        orig = v90_ep["instruction"]["instruction_text"]
        cleaned = ep["instruction"]["instruction_text"]
        if orig != cleaned and shown_rm < 3:
            print(f"  v90: {orig[:150]}")
            print(f"  v96: {cleaned[:150]}")
            print()
            shown_rm += 1
    print("Samples (kept initial turn — large rotation):")
    for ep, v90_ep in zip(new_episodes, v90_data["episodes"]):
        orig = v90_ep["instruction"]["instruction_text"]
        cleaned = ep["instruction"]["instruction_text"]
        if orig == cleaned and INIT_TURN_PREFIX_RE.match(orig) and shown_kp < 3:
            print(f"  v96 (kept): {cleaned[:150]}")
            shown_kp += 1

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
