#!/usr/bin/env python3
"""
Gate3-Gemma v100 for VAL_SEEN split.

Thin wrapper: overrides path constants in run_gate3_gemma_v100 and calls main().
All generation logic is shared — only split-specific paths differ.

val_seen:   778 episodes, vision_checkpoint fully pre-built (gate3_gemma_v22_val_seen_vision_checkpoint.json)
Generation: ~78 sec at 10 eps/s (vLLM server at 10.77.32.231:8000)

After generation, run:
    python3 generate_v101_val_seen.py
to apply v101 post-processing (far-wall, opener, sentence merge).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")

# Add pipeline root to path for gate2/gate5/gate6 submodules
sys.path.insert(0, str(ROOT))

# Import the shared v100 module
import run_gate3_gemma_v100 as v100

# Override path constants for val_seen
v100.VAL_UNSEEN_PATH = str(HABITAT_BASE / "val_seen" / "val_seen.json.gz")
v100.VOCAB_SOURCE_PATH = str(HABITAT_BASE / "val_seen" / "val_seen_v24.json.gz")
v100.VISION_CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v22_val_seen_vision_checkpoint.json"
v100.PREV_CHECKPOINT_PATH = None  # fresh generation
v100.CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v100_val_seen_checkpoint.json"
v100.OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_seen_gate3_gemma_v100.json.gz"
v100.DEPLOY_PATH = HABITAT_BASE / "val_seen" / "val_seen_v100.json.gz"

# Phase1 checkpoints: val_unseen p1 is fine as fallback since vision checkpoint
# covers all 778 val_seen episodes (Phase V is fully skipped; p1 only used as fallback)
# p73/p74 don't exist for val_seen → {} automatically (checked with .exists())

if __name__ == "__main__":
    print("=== Gate3-Gemma v100 — VAL_SEEN ===")
    print(f"Input: {v100.VAL_UNSEEN_PATH}")
    print(f"Vision ck: {v100.VISION_CHECKPOINT_PATH}")
    print(f"Output: {v100.OUTPUT_PATH}")
    print(f"Deploy: {v100.DEPLOY_PATH}")
    print()
    v100.main()
