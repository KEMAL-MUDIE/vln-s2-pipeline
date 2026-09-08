#!/usr/bin/env python3
"""
Convert auto-annotator Phase C outputs → Phase 1 checkpoint format.

The auto-annotator generates per-episode vision descriptions (Phase C).
This script converts them to the Phase 1 checkpoint format used by run_gate4_visual_*.py.

Phase 1 checkpoint format: {episode_id_str: {label: desc_or_None}}
  - "start": starting area description
  - "turn_N": turn point description (N = waypoint index)
  - "goal": goal/stop area description

Auto-annotator metadata format:
  episode_XXXXXX.json → {"vision_descriptions": {"start": "...", "turn_2": "...", "goal": "..."}}

Usage:
  python3 convert_auto_to_phase1_ckpt.py \\
    outputs/auto_metadata/ \\
    --output outputs/gate4_v143_phase1_checkpoint.json

The output checkpoint can then be used in run_gate4_visual_v143.py by setting:
  P1_CKPT = ROOT / "outputs" / "gate4_v143_phase1_checkpoint.json"
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Optional


def convert(metadata_dir: Path, output_path: Path) -> None:
    metadata_dir = Path(metadata_dir)
    output_path  = Path(output_path)

    checkpoint: Dict[str, Dict[str, Optional[str]]] = {}
    n_loaded = 0
    n_missing = 0

    for meta_file in sorted(metadata_dir.glob("episode_*.json")):
        try:
            d = json.load(open(meta_file))
        except Exception as e:
            print(f"  SKIP {meta_file.name}: {e}")
            continue

        ep_id = str(d.get("episode_id", meta_file.stem.split("_")[-1]))
        descs = d.get("vision_descriptions", {})

        if not descs:
            n_missing += 1
            continue

        checkpoint[ep_id] = {label: desc for label, desc in descs.items()}
        n_loaded += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(checkpoint, f)

    print(f"Phase 1 checkpoint: {n_loaded} episodes converted, {n_missing} missing")
    print(f"Saved: {output_path}")

    # Spot-check
    sample_ids = sorted(checkpoint.keys())[:3]
    for eid in sample_ids:
        print(f"\n  Episode {eid}:")
        for label, desc in checkpoint[eid].items():
            print(f"    [{label}] {desc or 'None'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata_dir", type=Path,
                    default=Path(__file__).parent / "outputs" / "auto_metadata")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).parent / "outputs" / "gate4_v143_phase1_checkpoint.json")
    args = ap.parse_args()
    convert(args.metadata_dir, args.output)
