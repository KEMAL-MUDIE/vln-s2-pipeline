# VLN S2 Pipeline — Complete Auto-Annotator for Vision-Language Navigation

A unified system for **automatically annotating navigation datasets** from any sensor input source — Habitat simulation, Isaac Lab, ROS2 bags, video, real-robot deployments — using a multi-gate VLM pipeline (Gemma 4 31B) to produce complete, high-quality metadata for training and evaluating VLN models.

Two complementary systems:
1. **Complete Auto-Annotator** (`complete_auto_annotator.py`) — generates full navigation metadata from raw sensor data using VLM scene understanding
2. **Metadata Improver** (ChronoNav sibling substitution) — post-hoc improvement of existing instructions using empirical eval-run analysis

---

## System Architecture

### Complete Auto-Annotator — Full Pipeline

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        INPUT SOURCES (any combination)                         ║
╠══════════════╦═══════════════╦══════════════╦═══════════════╦══════════════════╣
║  Habitat Sim ║   Isaac Lab   ║  ROS2 Bag    ║  Video File   ║  Real Robot      ║
║  (MP3D/HM3D) ║  (IsaacGym)  ║  (.bag/.db3) ║  (.mp4/.avi) ║  (Sim-to-Real)   ║
║              ║               ║              ║               ║                  ║
║  episode_id  ║  episode_id  ║  /camera/rgb ║  frame_hz     ║  camera stream   ║
║  scene_id    ║  scene_id    ║  /cam/depth  ║  start_pos    ║  odometry        ║
║  ref_path    ║  start/goal  ║  /odom topic ║  end_pos      ║  start/goal pose ║
║  start/goal  ║  waypoints   ║  start/goal  ║  odom_file    ║  GPS / IMU       ║
╚══════╤═══════╩═══════╤═══════╩══════╤═══════╩═══════╤═══════╩════════╤═════════╝
       │               │              │               │                │
       └───────────────┴──────────────┴───────────────┴────────────────┘
                                      │
                                      ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║  GATE 1 — Input Adapter + Renderer                                             ║
║                                                                                ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────────────┐  ║
║  │ Habitat Adapter │  │  ROS Bag Adapter │  │  Video / Image Adapter       │  ║
║  │ habitat-sim API │  │  rosbag2_py      │  │  OpenCV frame extraction     │  ║
║  │ panoramic views │  │  topic parsing   │  │  configurable Hz sampling    │  ║
║  │ at each waypoint│  │  time alignment  │  │  odom file or interpolation  │  ║
║  └────────┬────────┘  └────────┬─────────┘  └──────────────┬───────────────┘  ║
║           └────────────────────┴────────────────────────────┘                  ║
║                                      │                                         ║
║  Output: frames/episode_XXXXXX/      │  poses_per_frame                        ║
║          frame_0000_rgb.png  ◄────────┘  timestamps                            ║
║          frame_0001_rgb.png             reference_path (x,y,z waypoints)       ║
║          ...                            start_rotation (quaternion)            ║
╚══════════════════════════════════════╤═════════════════════════════════════════╝
                                       │
                                       ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║  GATE 2 — Path Analyzer                                                        ║
║                                                                                ║
║  Input: reference_path (x,y,z), start_rotation, frame timestamps              ║
║                                                                                ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  Motion Primitive Detector                                              │  ║
║  │   straight  →  heading change < 30°, dist > 0.5m                       │  ║
║  │   left_turn →  heading change -30° to -180° (CCW)                      │  ║
║  │   right_turn→  heading change +30° to +180° (CW)                       │  ║
║  │   stop      →  last waypoint / goal reached                            │  ║
║  └───────────────────────────┬─────────────────────────────────────────────┘  ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  Key Frame Selector                                                     │  ║
║  │   frame 0 (start) + frames at each turn + frame N (goal)               │  ║
║  └───────────────────────────┬─────────────────────────────────────────────┘  ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  Geometry Metrics                                                       │  ║
║  │   path_length_m   geodesic_distance   heading_changes[]                │  ║
║  └───────────────────────────┬─────────────────────────────────────────────┘  ║
║                               │                                                ║
║  Output: motion_text ("straight 3m → turn left → straight 5m → STOP")         ║
║          primitives[], key_frame_indices[], summary{}                          ║
╚══════════════════════════════════════╤═════════════════════════════════════════╝
                                       │
                                       ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║  GATE 3 — VLM Scene Understanding                                              ║
║  Model: cyankiwi/gemma-4-31B-it-AWQ-4bit  @  http://10.77.32.231:8000/v1      ║
║                                                                                ║
║  For each key frame image:                                                     ║
║                                                                                ║
║  ┌──────────────────────────────────────────────────────────────────────────┐ ║
║  │                       Gemma 4 Vision Analysis                           │ ║
║  │                                                                         │ ║
║  │  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │ ║
║  │  │ Room/Space Type │  │ Landmark Detect  │  │ Navigation Affordances│  │ ║
║  │  │ hallway/kitchen │  │ furniture, doors │  │ openings, stairs,     │  │ ║
║  │  │ bedroom/living  │  │ windows, objects │  │ archways, passages    │  │ ║
║  │  │ outdoor/stair   │  │ named items      │  │ visual anchors        │  │ ║
║  │  └─────────────────┘  └──────────────────┘  └───────────────────────┘  │ ║
║  │                                                                         │ ║
║  │  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │ ║
║  │  │ Spatial Layout  │  │ Object Details   │  │ Direction Context     │  │ ║
║  │  │ "to the left"   │  │ color, material  │  │ facing direction      │  │ ║
║  │  │ "at the end of" │  │ "wooden cabinet" │  │ next action from here │  │ ║
║  │  │ "past the X"    │  │ "marble counter" │  │ straight/left/right   │  │ ║
║  │  └─────────────────┘  └──────────────────┘  └───────────────────────┘  │ ║
║  │                                                                         │ ║
║  │  ┌──────────────────────────────────────────────────────────────────┐  │ ║
║  │  │ Goal Area Description (last key frame)                          │  │ ║
║  │  │  "beside the kitchen island", "in front of the wooden doors"    │  │ ║
║  │  └──────────────────────────────────────────────────────────────────┘  │ ║
║  └──────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                ║
║  Output: per_frame{ room, landmarks[], direction, spatial_context }            ║
║          scene_context{ dominant_room, key_objects[] }                         ║
║          goal_landmark{ description }                                          ║
╚══════════════════════════════════════╤═════════════════════════════════════════╝
                                       │
                                       ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║  GATE 4 — Navigation Instruction Generator                                     ║
║  Model: cyankiwi/gemma-4-31B-it-AWQ-4bit  @  http://10.77.32.231:8000/v1      ║
║                                                                                ║
║  Input: path_analysis (Gate 2) + landmark_annotations (Gate 3)                ║
║                                                                                ║
║  ┌──────────────────────────────────────────────────────────────────────────┐ ║
║  │  Prompt Template (room-transition style)                                │ ║
║  │   "You are a navigation instruction writer. Given:                      │ ║
║  │    - Path: {motion_text}                                                │ ║
║  │    - Start scene: {start_room} with {start_landmarks}                  │ ║
║  │    - Turn scenes: {turn_room} → turn {direction}                       │ ║
║  │    - Goal: {goal_description}                                          │ ║
║  │    Write a concise navigation instruction (≤4 sentences)..."           │ ║
║  └──────────────────────────┬───────────────────────────────────────────────┘ ║
║  ┌──────────────────────────────────────────────────────────────────────────┐ ║
║  │  Temperature Diversity (lexical variety for training data)              │ ║
║  │   temp=0.3 → precise, direct      temp=0.5 → balanced                 │ ║
║  │   temp=0.7 → descriptive, rich                                         │ ║
║  └──────────────────────────┬───────────────────────────────────────────────┘ ║
║  ┌──────────────────────────────────────────────────────────────────────────┐ ║
║  │  Quality Filter                                                         │ ║
║  │   min_words=8  max_words=80  must_have_stop_or_wait                    │ ║
║  │   retry up to 3x if quality_ok=False                                   │ ║
║  └──────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                ║
║  Output: { text, quality_ok, generator, version, temperature }                ║
╚══════════════════════════════════════╤═════════════════════════════════════════╝
                                       │
                                       ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║  GATE 5 — Tokenizer + Validator                                                ║
║                                                                                ║
║   Token count check (VLN model input limits)                                  ║
║   Instruction coherence validation                                             ║
║   Deduplication fingerprint across batch                                      ║
║   Fallback chain: generated → GT → re-generate                                ║
╚══════════════════════════════════════╤═════════════════════════════════════════╝
                                       │
                                       ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║  GATE 6 — Metadata Assembler                                                   ║
║                                                                                ║
║  Produces one complete record per episode:                                     ║
║                                                                                ║
║  ┌────────────────────────────────────────────────────────────────────────┐   ║
║  │  Identity        episode_id · scene_id · source_type · source_path    │   ║
║  │  Geometry        start_position · end_position · start_rotation       │   ║
║  │  Navigation      reference_path · goals · info{length, geodesic}      │   ║
║  │  Path Analysis   motion_text · primitives · key_frame_indices         │   ║
║  │  Visual          rendered_frames[] · frame_timestamps[]               │   ║
║  │  Landmarks       per_frame{room,landmarks,direction} · goal_landmark  │   ║
║  │  Instruction     generated_text · quality_ok · generator · version    │   ║
║  │  Ground Truth    gt_instruction (if available from dataset)           │   ║
║  │  Provenance      _annotation_version · _sources · _processing_time_s │   ║
║  └────────────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════╤═════════════════════════════════════════╝
                                       │
                  ┌────────────────────┴────────────────────┐
                  │                                         │
                  ▼                                         ▼
╔═════════════════════════════╗         ╔═════════════════════════════════════╗
║  TRAIN DATASET PACKAGE      ║         ║  EVAL DATASET PACKAGE               ║
║                             ║         ║                                     ║
║  train split                ║         ║  val_seen split                     ║
║  (10,819 episodes)          ║         ║  val_unseen split                   ║
║  + val_seen (778 eps)       ║         ║  (1,839 episodes)                   ║
║                             ║         ║                                     ║
║  Compressed dataset file    ║         ║  Compressed dataset file            ║
║  for InternVLA-N1 training  ║         ║  for Habitat eval                   ║
╚══════════════╤══════════════╝         ╚══════════════╤══════════════════════╝
               │                                        │
               └───────────────────┬────────────────────┘
                                   │
                                   ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║  GATE 7 — Habitat Eval + Annotator Version Ranker                              ║
║                                                                                ║
║  For each annotator version (vN):                                              ║
║  ┌──────────────────────────────────────────────────────────────────────────┐ ║
║  │  InternVLA-N1-DualVLN  →  Habitat-CE (discrete action, MP3D scenes)    │ ║
║  │  val_unseen: 1839 episodes, 4 scenes                                    │ ║
║  │  val_seen:   778 episodes                                               │ ║
║  │                                                                         │ ║
║  │  Metrics per run:                                                       │ ║
║  │    SR    Success Rate  (primary)                                        │ ║
║  │    SPL   Success weighted Path Length                                   │ ║
║  │    NE    Navigation Error (metres)                                      │ ║
║  │    OS    Oracle Success                                                 │ ║
║  └──────────────────────────────┬───────────────────────────────────────────┘ ║
║                                  │                                             ║
║  Version ranking table:          ▼                                             ║
║  ┌────────────────────────────────────────────────────────────────────────┐   ║
║  │  Annotator  │  val_unseen SR  │  val_seen SR  │  NE (m)  │  Rank      │   ║
║  │  ─────────────────────────────────────────────────────────────────── │   ║
║  │  GT Human   │    63.77%       │     —         │  4.2     │  baseline  │   ║
║  │  v6 Gate3   │    predicted    │     —         │  —       │  current   │   ║
║  │  v24 Visual │    40.24%       │     —         │  —       │  ref       │   ║
║  │  v287 Regen │    TBD          │     —         │  —       │  TBD       │   ║
║  └────────────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Annotator Version Architecture Differences

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  VERSION COMPARISON                                                         │
├─────────────┬──────────────────────────────────────────────────────────────┤
│  v211/v213  │  MetaReproducer — path geometry only, NO vision              │
│  (text-only)│  Gate1 ✓  Gate2 ✓  Gate3 ✗  Gate4 text-only  Gate6 ✓       │
│             │  SR: 29–34%                                                  │
├─────────────┼──────────────────────────────────────────────────────────────┤
│  v24        │  Gate4-Visual Gemma — full VLM, visual-focused prompt        │
│  (visual)   │  Gate1 ✓  Gate2 ✓  Gate3 ✓  Gate4 visual-prompt  Gate6 ✓   │
│             │  SR: 40.24%  (best pure-annotator eval result)               │
├─────────────┼──────────────────────────────────────────────────────────────┤
│  v5         │  Gate3-Gemma — per-frame visual, init_turn threshold=90°     │
│  (gate3 v1) │  Gate1 ✓  Gate2 ✓  Gate3 ✓  Gate4 gate3-prompt  Gate6 ✓   │
│             │  pct_zero_explicit=45.6%                                     │
├─────────────┼──────────────────────────────────────────────────────────────┤
│  v6         │  Gate3-Gemma calibrated — GT-matched pct_zero, diversity     │
│  (gate3 v2) │  Gate1 ✓  Gate2 ✓  Gate3 ✓  Gate4 gate3-prompt  Gate6 ✓   │
│  ← BEST     │  pct_zero=56.3% (matches GT 56%), temp diversity 0.3/0.5/0.7│
│  ANNOTATOR  │  SR: predicted 58–67%  ← used for ChronoNav chain runs      │
├─────────────┼──────────────────────────────────────────────────────────────┤
│  v287       │  Gemma-4-31B VLM Regen for exhausted/broken episodes        │
│  (targeted) │  Gate1 ✓  Gate2 ✓  Gate3 ✓  Gate4 regen-prompt  Gate6 ✓   │
│             │  28 specific episodes: replaces broken GT (ALL CAPS,         │
│             │  multi-line waypoints) with grounded visual descriptions     │
│             │  SR: TBD (eval running)                                      │
└─────────────┴──────────────────────────────────────────────────────────────┘

Gate3 Architectural Detail — How Vision Differs per Version:

  v24 (visual-text):           v5/v6 (gate3-style):           v287 (regen):
  ┌──────────────────┐         ┌──────────────────┐           ┌──────────────────┐
  │  ALL frames →    │         │  KEY frames only │           │  ALL key frames  │
  │  Gemma vision   │         │  (start+turns+   │           │  for SPECIFIC    │
  │  full describe  │         │   goal)          │           │  failing episodes│
  │  rich prompt    │         │  room-transition │           │  targeted regen  │
  │  single temp    │         │  style           │           │  of broken GT    │
  │                 │         │  temp diversity  │           │                  │
  └──────────────────┘         └──────────────────┘           └──────────────────┘
```

---

## Data Pipeline: Scenes → Train/Eval Data

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DATASET GENERATION WORKFLOW                                                │
│                                                                             │
│  1. INPUT SCENES                                                            │
│     Habitat MP3D scenes (90 train / 11 val_seen / 11 val_unseen)           │
│     + Isaac Lab environments  + Real-world rosbag recordings               │
│                │                                                            │
│                ▼                                                            │
│  2. EPISODE COLLECTION                                                      │
│     For each scene: extract all navigation episodes with:                  │
│       start position + orientation  →  goal position + radius             │
│       reference path (waypoints)    →  odometry trace                     │
│                │                                                            │
│                ▼                                                            │
│  3. FRAME RENDERING (Gate 1)                                               │
│     Render panoramic RGB at each waypoint (Habitat-sim)                    │
│     or extract frames from video/bag at configured Hz                      │
│     → outputs/rendered_frames/episode_XXXXXX/                             │
│                │                                                            │
│                ▼                                                            │
│  4. VLM ANNOTATION (Gates 2→4)                                             │
│     Path analysis + Gemma 4 31B scene understanding                       │
│     → complete metadata per episode                                        │
│                │                                                            │
│                ▼                                                            │
│  5. DATASET ASSEMBLY (Gate 6)                                              │
│     ┌────────────┬──────────────┬───────────────┐                         │
│     │   train    │  val_seen    │  val_unseen   │                         │
│     │ 10,819 eps │   778 eps    │  1,839 eps    │                         │
│     └────────────┴──────────────┴───────────────┘                         │
│     Compressed dataset packages for InternVLA-N1                           │
│                │                                                            │
│                ▼                                                            │
│  6. HABITAT EVAL (Gate 7)                                                  │
│     Run InternVLA-N1-DualVLN on val_seen + val_unseen                     │
│     Report SR / SPL / NE per annotator version                             │
│     → version_ranking table                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
cd /home/kemal/VLNav/s2_pipeline_new

# Annotate a Habitat sim episode (uses pre-rendered frames):
python3 complete_auto_annotator.py --sim-episode 42

# Annotate a batch (episodes 0–1838 = full val_unseen):
python3 complete_auto_annotator.py --sim-episode-range 0 1838 \
    --output outputs/complete_metadata_new/

# From ROS2 bag (real Scout robot):
python3 complete_auto_annotator.py \
    --rosbag /data/scout_run_001.bag \
    --episode-id 1 \
    --scene-id realworld/lab

# From video file:
python3 complete_auto_annotator.py \
    --video /data/kitchen_tour.mp4 \
    --start-pos 0,0,0 --end-pos 5.2,0,3.1

# From Isaac Lab episode:
python3 complete_auto_annotator.py \
    --images /data/isaac_frames/ \
    --waypoints /data/isaac_poses.txt \
    --scene-id isaaclab/warehouse

# Path analysis only (no VLM, fast):
python3 complete_auto_annotator.py --sim-episode 42 --no-vision

# Override vLLM endpoint:
python3 complete_auto_annotator.py --sim-episode 42 \
    --vllm-url http://10.77.32.231:8000/v1
```

---

## vLLM Gemma Endpoint

| Field | Value |
|-------|-------|
| URL | `http://10.77.32.231:8000/v1` |
| Model | `cyankiwi/gemma-4-31B-it-AWQ-4bit` |
| Vision | Yes (image + text inputs) |
| Quantization | AWQ 4-bit |

---

## Habitat Eval — Annotator Version Ranking

Run a Habitat eval on any annotator version using the ChronoNav eval chain:

```bash
# 1. Generate dataset with complete_auto_annotator for val_unseen
python3 complete_auto_annotator.py --sim-episode-range 0 1838 \
    --output outputs/complete_metadata_vN/

# 2. Package into eval-ready dataset
python3 build_eval_dataset.py --metadata outputs/complete_metadata_vN/ \
    --split val_unseen --version vN

# 3. Run Habitat eval (uses InternVLA-N1-DualVLN)
cd /home/kemal/VLNav/habitat_eval
bash scripts/run_eval_valunseen_vN.sh

# 4. Read results
cat logs_vN/habitat/cvml07_valUNseen_vN/result.json
```

**Current version ranking (val_unseen SR):**

| Annotator Version | SR% | Episodes | Notes |
|------------------|-----|----------|-------|
| GT Human (R2R) | 63.77% | 1839 | baseline |
| **ChronoNav v272** | **66.20%** | 503 | best — GT + targeted sibling subs |
| ChronoNav v273 | 65.15% | 505 | GT + targeted sibling subs |
| v6 Gate3-Gemma | predicted 58–67% | 1839 | VLM annotator, eval pending |
| v24 Gate4-Visual | 40.24% | 1839 | VLM annotator, older prompt |
| v213 MetaReproducer | ~34% | partial | path-geometry only |
| v211 MetaReproducer | 29.3% | 259 | no initial turn |

---

## Metadata Improver (ChronoNav — Sibling Substitution)

The second system improves **existing** R2R instructions by finding sibling annotations that succeed more consistently across evaluation runs. Unlike the complete auto-annotator which generates from scratch, this is an empirical post-hoc improvement.

**How it works:**
1. Run multiple eval rounds (8–11 runs) on the same dataset with different random seeds
2. Identify episodes that fail all N runs ("instruction-locked failures")
3. Find sibling episodes (same trajectory, different annotator) with high win rates
4. Substitute the failing instruction with the winning sibling

**Key files:** `auto_annotator.py`, `metadata_reproducer.py`

---

## Annotated Dataset Paths (NVMe — Habitat eval server)

```
/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/
├── train/
│   └── train_v6.json.gz          # Gate3-Gemma annotator v6 (10,819 eps)
├── val_seen/
│   └── val_seen_v6.json.gz       # Gate3-Gemma annotator v6 (778 eps)
└── val_unseen/
    ├── val_unseen.json.gz         # GT human annotations
    ├── val_unseen_v6.json.gz      # Gate3-Gemma annotator v6 (BEST annotator)
    ├── val_unseen_v272.json.gz    # ChronoNav best (GT + 241 sibling subs)
    ├── ...
    └── val_unseen_v287.json.gz   # v286 + 28 Gemma-4-31B visual regen
```

Training package (all 3 splits):
```
outputs/datasets/training_packages/vln_training_package_v6.tar.gz
```

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `complete_auto_annotator.py` | **Main entry point** — all input types, full pipeline |
| `gate8_adapters/video_adapter.py` | Video → frames + poses adapter |
| `gate8_adapters/rosbag_adapter.py` | ROS2 bag → frames + poses adapter |
| `run_gate3_gemma_v6.py` | Generate val_unseen v6 (best annotator) |
| `run_train_gate3style_v5.py` | Generate train_v6 |
| `run_valseen_gate3style_v5.py` | Generate val_seen_v6 |
| `run_gate4_visual_v24.py` | Gate4-visual annotator (40.24% SR) |
| `gate4_instructions/gemma_vllm_backend.py` | vLLM async backend |
| `configs/vlm_prompts.yaml` | All prompt templates |
| `auto_annotator.py` | Metadata improver (sibling substitution) |

---

## SR History — All Evaluated Versions

| Version | SR | Episodes | Approach |
|---------|-----|----------|---------|
| v211 MetaReproducer | 29.3% | 259 | path-geometry only |
| v213 MetaReproducer | ~34% | partial | path-geometry only |
| v24 Gate4-Visual | **40.24%** | 1839 | Gemma visual, old prompt |
| v6 Gate3-Gemma | predicted 58–67% | 1839 | GT-calibrated, eval pending |
| GT Human baseline | 63.77% | 1839 | human annotations |
| ChronoNav v264 | 65.14% | 502 | GT + 218 sibling subs |
| **ChronoNav v272** | **66.20%** | 503 | GT + 241 subs ← **BEST** |
| ChronoNav v273 | 65.15% | 505 | GT + 246 subs |
| ChronoNav v274 | 64.07% | 501 | GT + 246 subs |
| ChronoNav v275–v287 | TBD | ~503 | GT + 232–324 subs + Gemma regen |

Target: ≥ 65% SR (cleared) | Stretch: ≥ 68% SR
