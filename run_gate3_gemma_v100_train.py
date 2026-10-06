#!/usr/bin/env python3
"""
Gate3-Gemma v100 for TRAIN split.

Thin wrapper: overrides path constants in run_gate3_gemma_v100 and calls main().
All generation logic is shared — only split-specific paths differ.

train: 10819 episodes, vision_checkpoint fully pre-built (gate3_gemma_v22_train_vision_checkpoint.json)
Generation: ~22 min at 8 eps/s (vLLM server at 10.77.32.231:8000)

After generation, run:
    python3 generate_v101_train.py
to apply v101 post-processing (far-wall, opener, sentence merge).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")

sys.path.insert(0, str(ROOT))

import run_gate3_gemma_v100 as v100

# Override path constants for train
v100.VAL_UNSEEN_PATH = str(HABITAT_BASE / "train" / "train.json.gz")
v100.VOCAB_SOURCE_PATH = str(HABITAT_BASE / "train" / "train.json.gz")
v100.VISION_CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v22_train_vision_checkpoint.json"
v100.PREV_CHECKPOINT_PATH = None  # fresh generation
v100.CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v100_train_checkpoint.json"
v100.OUTPUT_PATH = ROOT / "outputs" / "datasets" / "train_gate3_gemma_v100.json.gz"
v100.DEPLOY_PATH = HABITAT_BASE / "train" / "train_v100.json.gz"

# PHASE1_CHECKPOINT stays as val_unseen (safe: p1.get(eid, {}) returns {} for train eids,
# since vision checkpoint covers 100% of train episodes and Phase V is fully skipped)

if __name__ == "__main__":
    print("=== Gate3-Gemma v100 — TRAIN ===")
    print(f"Input: {v100.VAL_UNSEEN_PATH}")
    print(f"Vision ck: {v100.VISION_CHECKPOINT_PATH}")
    print(f"Output: {v100.OUTPUT_PATH}")
    print(f"Deploy: {v100.DEPLOY_PATH}")
    print(f"Episodes: 10819 (ETA ~22 min at 8 eps/s)")
    print()
    v100.main()
