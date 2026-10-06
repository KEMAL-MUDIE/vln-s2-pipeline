#!/usr/bin/env python3
"""
Gate3-Gemma v61 for TRAIN split.

Thin wrapper: overrides path constants in run_gate3_gemma_v61 and calls main().
All generation logic is shared — only split-specific paths differ.

train: 10819 episodes (IDs 1-10837), vision_checkpoint pre-built
       (gate3_gemma_v22_train_vision_checkpoint.json, 10822 entries)
Generation: ~17-20 min at ~10 eps/s (vLLM server at 10.77.32.231:8000)

Notes on phase1 checkpoint:
- gate4_v19_phase1_checkpoint covers only val_unseen IDs 1-1839
- 83% of train IDs are > 1839 → phase1 returns {} (correct: no data)
- For train IDs 1-1839, phase1 may give wrong descriptions (val_unseen
  scene data for same ID); however phase1 is only used as VISUAL LANDMARK
  FALLBACK when the vision checkpoint doesn't have data. Since the train
  vision checkpoint covers all 10822 train entries, phase1 fallback is
  rarely triggered, and any fallback for shared IDs 1-1839 is minor.

After generation, apply hallway injection + opener corrections (like val_unseen
v61hpp post-processing) to create train_gate3_gemma_v61hpp.json.gz.

Expected raw v61 train stats (before post-processing):
  avg_turns≈0.55-0.60, hallway≈10-20%, wooden≈0-5%, through_the≈25-30%
GT train targets:
  avg_turns=0.648, hallway=23%, wooden=2%, through_the=25%
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")

sys.path.insert(0, str(ROOT))

import run_gate3_gemma_v61 as v61

# Override path constants for train split
v61.VAL_UNSEEN_PATH = str(HABITAT_BASE / "train" / "train.json.gz")
v61.VOCAB_SOURCE_PATH = str(HABITAT_BASE / "train" / "train.json.gz")
v61.VISION_CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v22_train_vision_checkpoint.json"
v61.PREV_CHECKPOINT_PATH = None   # fresh generation
v61.CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v61_train_checkpoint.json"
v61.OUTPUT_PATH = ROOT / "outputs" / "datasets" / "train_gate3_gemma_v61.json.gz"
v61.DEPLOY_PATH = HABITAT_BASE / "train" / "train_v61.json.gz"

if __name__ == "__main__":
    print("=== Gate3-Gemma v61 — TRAIN ===")
    print(f"Input:     {v61.VAL_UNSEEN_PATH}")
    print(f"Vision ck: {v61.VISION_CHECKPOINT_PATH}")
    print(f"Prev ck:   {v61.PREV_CHECKPOINT_PATH}")
    print(f"Checkpoint:{v61.CHECKPOINT_PATH}")
    print(f"Output:    {v61.OUTPUT_PATH}")
    print(f"Deploy:    {v61.DEPLOY_PATH}")
    print()
    v61.main()
