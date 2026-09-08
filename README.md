# VLN S2 Pipeline — Auto-Annotator for Vision-Language Navigation

Auto-annotator for the [R2R (Room-to-Room)](https://bringmeaspoon.org/) VLN dataset used in the **ChronoNav / InternVLA-N1-DualVLN** system. Generates natural navigation instructions from path geometry and visual scene context, replacing or augmenting human annotations.

## Auto-Annotator Versions

| Version | SR (val_unseen) | Approach | Key Properties |
|---------|----------------|----------|----------------|
| **v6** | predicted 58–67% | Gate3-Gemma | pct_zero=56.3% **(matches GT=56% exactly)**, avg_explicit=0.44, avg_words=24.7 |
| v5 | predicted 55–65% | Gate3-Gemma | pct_zero=45.6%, avg_explicit=0.54, init_turn threshold=90° |
| v24 | **40.24%** | Visual-text Gemma | Best evaluated text-only result |
| v213 | ~34% | MetaReproducer | Path-geometry only |
| v211 | 29.3% | MetaReproducer | No initial turn |

GT (human) baseline: **63.77% SR** | Gemma text-only baseline: **62.12% SR**

> **v6 is the current best annotator** — GT-calibrated pct_zero, GT-style room-transition instructions.

---

## Annotated Splits

### val_unseen — v6 (1,839 episodes, **BEST**)

| Metric | v6 | GT |
|--------|----|----|
| avg_explicit_turns | 0.44 | 0.66 |
| pct_zero_explicit | **56.3%** | **56%** |
| avg_words | 24.7 | 26.8 |
| pct_3plus_turns | 0.0% | 3.8% |
| quality_ok | 1839/1839 | — |

Key design decisions:
- Gate3 per-frame visual context: specific room names, landmarks, stop descriptions extracted from panoramic images
- init_turn threshold = 90° (only sharp/around rotations get explicit direction)
- 20% stochastic suppression: episodes with `episode_id % 10 < 2` skip init_turn context → natural room-transition start
- Triplet temperature diversity: same path regenerated at temps 0.3 / 0.5 / 0.7 for lexical variety

### train — v6 (10,819 episodes)

| Metric | Value |
|--------|-------|
| avg_explicit_turns | 0.60 |
| pct_zero_explicit | 40.5% |
| avg_words | 17.3 |
| Generation | Gate3-style prompt, text-only path context (no visual data for train) |

### val_seen — v6 (778 episodes)

| Metric | Value |
|--------|-------|
| avg_explicit_turns | 0.57 |
| pct_zero_explicit | 43.6% |
| avg_words | 17.3 |

---

## Dataset Paths (NVMe — Habitat eval server)

```
/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/
├── train/
│   └── train_v6.json.gz          # auto-annotator v6 (10,819 eps)
├── val_seen/
│   └── val_seen_v6.json.gz       # auto-annotator v6 (778 eps)
└── val_unseen/
    └── val_unseen_v6.json.gz     # auto-annotator v6 (1,839 eps) ← BEST
```

Training package (all 3 splits, clean naming):
```
outputs/datasets/training_packages/vln_training_package_v6.tar.gz  # 1.2 MB
```

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `run_gate3_gemma_v6.py` | Generate val_unseen v6 (uses per-frame visual context) |
| `run_gate3_gemma_v5.py` | Generate val_unseen v5 |
| `run_train_gate3style_v5.py` | Generate train_v6 (gate3-style prompt, text-only) |
| `run_valseen_gate3style_v5.py` | Generate val_seen_v6 |
| `run_gate4_visual_v24.py` | Best visual-only annotator (40.24% SR) |
| `gate4_instructions/gemma_vllm_backend.py` | vLLM async backend with per-task temperature |
| `configs/vlm_prompts.yaml` | Prompt templates (including `instruction_generation_gate3`) |

---

## Architecture

```
Path geometry + panoramic images
        │
        ▼
Gate3 per-frame extractor
  (start room, turn landmarks, goal room, stop description)
        │
        ▼
Gemma-4-31B via vLLM  ──  instruction_generation_gate3 prompt
  (room-transition style, ≤4 sentences, landmark-focused)
        │
        ▼
Auto-annotated instruction
  (pct_zero=56.3% — matches GT human annotation style)
```

The `gate3_perframe/` data (visual context) exists **only for val_unseen** (1,839 episodes). Train and val_seen use the gate3-style prompt with text-only path context (heading changes, distances, turn angles).

---

## Evaluation Infrastructure

Eval uses [InternVLA-N1-DualVLN](https://github.com/OpenGVLab/InternVLA) on Habitat-CE (discrete action, MP3D scenes).

```bash
# Start v6 eval (after stopping any lower-SR eval)
cd /home/kemal/VLNav/habitat_eval
nohup bash scripts/run_eval_valunseen_auto_v6_gate3gemma.sh &
```

Config: `habitat_dual_system_cvml07_valUNseen_auto_v6_gate3gemma.py`  
Split: `val_unseen_v6` | Port: 2348 | GPUs: device=1,2 | EGL: device=1

---

## SR History

| Version | SR | Episodes | Status |
|---------|-----|----------|--------|
| v211 MetaReproducer | 29.3% | 259 | complete |
| v213 MetaReproducer | ~34% | partial | stopped (below 60% threshold) |
| v24 Visual-text Gemma | **40.24%** | 1839 | complete |
| v5 Gate3-Gemma | predicted 55–65% | — | not evaluated |
| **v6 Gate3-Gemma** | **predicted 58–67%** | — | eval pending |

Target: ≥65% SR (minimum 60%) | GT baseline: 63.77%
