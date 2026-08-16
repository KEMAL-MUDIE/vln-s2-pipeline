# S2 Pipeline — Auto-Annotation of VLN Episodes from Images + Paths

## Core Goal

Build a pipeline that takes **only** scene images + trajectory/pose data as input and produces
GT-quality VLN metadata + navigation instructions — with **no access to existing instructions**.

This pipeline will then be used to annotate new scenes (rosbags, real-world captures, sim-to-real
transfers) so they can be used for training InternNav models.

## Why the Old Approach Was Wrong

The previous "S2 pipeline" took existing GT instructions and paraphrased them with Gemma.
That tests instruction robustness, not annotation capability. The correct task is:

```
INPUT:  scene_images[]  +  reference_path[]  +  scene_id  +  goal_position
OUTPUT: instruction_text  +  instruction_tokens  +  full episode JSON (GT format)
```

No GT instruction is ever shown to the model. The VLM must reason about the visual scene
and path geometry to produce the instruction independently.

---

## Pipeline Architecture

```
 [Scene .glb / Rosbag / Video]
         │
         ▼
 ┌──────────────────┐
 │ Gate 1: Renderer │  →  RGB frames at each pose along path
 └──────────────────┘
         │
         ├────────────────────────────────────┐
         ▼                                    ▼
 ┌──────────────────┐              ┌──────────────────────┐
 │ Gate 2: Path     │              │ Gate 3: Landmark     │
 │ Analyzer         │              │ Detector (VLM)       │
 │ turns/distances  │              │ rooms/objects/refs   │
 └──────────────────┘              └──────────────────────┘
         │                                    │
         └──────────────┬─────────────────────┘
                        ▼
               ┌────────────────────────┐
               │ Gate 4: Instruction   │
               │ Generator (VLM)       │
               │ motion + landmarks    │
               │ → GT-style text       │
               └────────────────────────┘
                        │
                        ▼
               ┌────────────────────────┐
               │ Gate 5: Tokenizer     │
               │ text → tokens (GT     │
               │ vocabulary)           │
               └────────────────────────┘
                        │
                        ▼
               ┌────────────────────────┐
               │ Gate 6: Assembler     │
               │ → VLN episode JSON    │
               │ → .json.gz batch      │
               └────────────────────────┘
                        │
                        ▼
               ┌────────────────────────┐
               │ Gate 7: Eval          │
               │ Habitat + Isaac Lab   │
               │ on generated episodes │
               └────────────────────────┘
```

---

## Gate-by-Gate Plan

### Gate 0 — Foundation & GT Analysis
**Goal**: Understand GT episode format in full detail; establish quality metrics.

Steps:
1. Parse GT VLN-CE val_unseen (1839 episodes) and extract statistics:
   - Instruction length distribution (words, sentences)
   - Path waypoint count distribution
   - Turn direction distribution (left/right/straight proportions)
   - Landmark category distribution from GT text
   - Instruction style patterns (how GT describes rooms, objects, turns)
2. Build a qualitative rubric for "GT-like" instructions (used in Gate 7 eval)
3. Output: `gate0_analysis/gt_statistics.json`, `gate0_analysis/gt_style_examples.txt`

Status: **TODO** (script: `gate0_analysis/analyze_gt.py`)

---

### Gate 1 — Scene Renderer
**Goal**: Render RGB frames along a reference_path using Habitat-Sim.

Inputs: `scene_id` (e.g. `mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb`), `reference_path` (3D waypoints),
        `start_rotation`

Outputs per episode: `{episode_id}/frame_{i:04d}.png` + `poses.json` with full 6-DOF pose

Key design decisions:
- Render at 640×480 (match GT eval resolution)
- Camera: 90° HFOV (Habitat default for R2R)
- Frame density: one frame per waypoint + 4 intermediate frames per segment
- Also capture: heading-aligned frame at each waypoint (what agent sees facing goal direction)

Adapters for other sources (Gate 8 uses these):
- `adapters/habitat_adapter.py`: .glb + path → frames (this gate)
- `adapters/rosbag_adapter.py`: ROS2 bag → frames + poses
- `adapters/video_adapter.py`: MP4 + GPS/IMU → frames + poses
- `adapters/isaac_adapter.py`: IsaacSim → frames + poses

Status: **TODO** (script: `gate1_renderer/renderer.py`)

---

### Gate 2 — Path Analyzer
**Goal**: Convert 3D waypoints into human-readable motion primitives.

Inputs: `reference_path` (list of [x,y,z] waypoints), `start_rotation` (quaternion)

Algorithm:
1. Compute heading direction at each waypoint from consecutive positions
2. Detect heading changes > 30° → classify as left/right turn
3. Compute segment distances between waypoints
4. Detect elevation changes (up/down stairs)
5. Output sequence:
   ```
   [{"type":"straight", "distance_m": 2.3},
    {"type":"left_turn", "angle_deg": 87},
    {"type":"straight", "distance_m": 1.5},
    {"type":"stop"}]
   ```

Additional analysis:
- Is path mostly linear or winding?
- How many decision points?
- Geodesic distance (sum of segment lengths along path)

Status: **TODO** (script: `gate2_path/path_analyzer.py`)

---

### Gate 3 — Landmark Detector
**Goal**: Identify salient visual landmarks at key frames along the path.

Inputs: frames from Gate 1, motion primitives from Gate 2

Key frame selection:
- Frame 0 (start — what agent sees first)
- Frame at each significant turn (the decision point image)
- Frame approaching the goal (last 2 waypoints)

VLM prompt strategy:
```
Given this indoor navigation image, identify:
1. The room type (kitchen, bedroom, hallway, living room, etc.)
2. Salient objects that could serve as landmarks for navigation
   (furniture, doorways, stairs, rugs, columns, plants, etc.)
3. The approximate direction of the main path (left, right, straight ahead)

Be concise. Format: room_type | landmark1, landmark2, landmark3
```

Backends:
- **Gemma 4 31B AWQ** (local, fast): primary
- **GPT-4o** (API): secondary / quality comparison

Output per frame:
```json
{"frame": 0, "room": "bedroom", "landmarks": ["gray couch", "doorway", "rug"]}
```

Status: **TODO** (script: `gate3_landmarks/landmark_detector.py`)

---

### Gate 4 — Instruction Generator
**Goal**: Produce GT-quality navigation instruction text from motion + landmarks.

This is the core task. The VLM receives:
- Images at key decision points (embedded in prompt)
- Motion primitive sequence from Gate 2
- Landmarks at each key frame from Gate 3

Prompt template:
```
You are a navigation instruction writer for a VLN (Vision-Language Navigation) dataset.
Your task: write a natural language navigation instruction for the path shown.

PATH DESCRIPTION:
- Start: facing [direction] in a [room_type]
- Segment 1: go straight ~2.3m (landmark: gray couch ahead)
- Decision point: turn left 87° (landmark: doorway on left)
- Segment 2: go straight ~1.5m
- Goal: stop near [goal_landmark]

KEY IMAGES: [image_start], [image_at_turn], [image_at_goal]

STYLE REQUIREMENTS:
- 1-4 concise sentences, 15-40 words total
- Reference visible landmarks, not abstract coordinates
- Use natural turn language: "turn left", "make a left", "go left at"
- Specify stop condition: "stop at/near/by [landmark]"
- Do NOT mention distances in meters; use natural language instead

Write the navigation instruction:
```

Quality filters:
- Length check: 10-60 words
- Must contain a stop condition
- Must reference at least one landmark
- Retry up to 3x if quality check fails

Status: **TODO** (script: `gate4_instructions/instruction_generator.py`)

---

### Gate 5 — Tokenizer
**Goal**: Convert generated instruction text to GT-compatible token array.

The GT uses a fixed vocabulary (`instruction_vocab.word2idx_dict`, 2711 words).
Tokens are padded to 200 positions with 0 (PAD_INDEX).

Steps:
1. Lowercase + tokenize text (same tokenization as GT dataset)
2. Map words to indices using word2idx_dict (unknown words → 1 = UNK_INDEX)
3. Add `</s>` end token
4. Pad to 200 with 0

Note: Modern InternVLA-N1 models use their own tokenizer and don't strictly need
`instruction_tokens` — but including it maintains 100% format compatibility.

Status: **TODO** (script: `gate5_tokenizer/tokenizer.py`)

---

### Gate 6 — Episode Assembler
**Goal**: Combine all components into valid VLN episode JSON, matching GT format exactly.

Input from all previous gates + original episode structural metadata
(start_position, start_rotation, reference_path, goals, scene_id — these come from the path data).

Output:
```json
{
  "episodes": [
    {
      "episode_id": 1,
      "trajectory_id": 15,
      "scene_id": "mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb",
      "start_position": [...],
      "start_rotation": [...],
      "info": {"geodesic_distance": 7.96},
      "goals": [{"position": [...], "radius": 3.0}],
      "instruction": {
        "instruction_text": "<GENERATED>",
        "instruction_tokens": [...]
      },
      "reference_path": [...]
    }
  ],
  "instruction_vocab": { ... }
}
```

Batch processing: process all 1839 val_unseen episodes, output `val_unseen_generated.json.gz`

Status: **TODO** (script: `gate6_assembler/assembler.py`)

---

### Gate 7 — Evaluation Harness
**Goal**: Measure how well the pipeline's generated instructions support navigation.

Evaluation method:
1. Run the pipeline on ALL VLN-CE val_unseen episodes → `val_unseen_generated.json.gz`
   (using ONLY images + paths as input — no GT instructions shown)
2. Run Habitat eval with generated dataset (same InternVLA-N1-DualVLN model as GT baseline)
3. Compare:

| Dataset | SR | SPL | NE |
|---------|-----|-----|-----|
| GT instructions | 63.89% | 58.55% | 4.027m |
| Generated instructions (this pipeline) | TBD | TBD | TBD |

Success criterion: SR within ±3pp of GT → pipeline is ready for new data annotation.

Also run Isaac Lab eval for VLN-PE comparison.

Stretch: compute semantic similarity between generated and GT instructions
(BERTScore, BLEU-4, CIDEr) as an additional quality signal.

Status: **TODO** (uses existing `habitat_eval/` infrastructure)

---

### Gate 8 — New Data Adapters
**Goal**: Enable the pipeline to annotate new data from any source.

Adapters to build:
1. **Rosbag adapter** (`gate8_adapters/rosbag_adapter.py`):
   - Input: ROS2 .bag file with `/camera/color/image_raw` + `/gdq/msg/gdq_odom`
   - Extract: frames at ~2 Hz + 6-DOF poses
   - Output: `frames/` + `poses.json` (same format as Gate 1 output)

2. **Video+pose adapter** (`gate8_adapters/video_adapter.py`):
   - Input: MP4 + CSV of [timestamp, x, y, z, qw, qx, qy, qz]
   - Output: same format as Gate 1

3. **IsaacSim adapter** (`gate8_adapters/isaac_adapter.py`):
   - Render frames from Isaac Sim using the H1 robot camera
   - Output: same format as Gate 1

All adapters output to the same interface → Gate 2 onwards is unchanged.

Status: **TODO** (rosbag adapter is highest priority for real-world use)

---

### Gate 9 — Training
**Goal**: Train InternNav models on combined data.

Training data:
- VLN-CE original GT (1839 train episodes × N augmentations)
- Auto-annotated new scenes (from Gate 8 → Gate 6)

Training configs: extend existing `internnav/` training pipeline.

Status: **FUTURE** (after Gates 0-7 validated)

---

### Gate 10 — Visually Impaired Fine-tuning
**Goal**: Adapt instruction style for visually impaired users.

VI instruction rules (to define):
- More precise landmark descriptions ("the door is immediately to your left")
- Distance cues in steps not meters
- Texture/sound/tactile cues where relevant
- Hazard warnings ("step down", "narrow passage")
- Confirmation points ("you should now feel carpet underfoot")

Fine-tune the Gate 4 instruction generator with VI-style rules injected into prompt.

Status: **FUTURE** (after Gate 9)

---

## Implementation Order (Time-Efficient)

```
Week 1:  Gate 0 (1d) → Gate 2 (1d) → Gate 5 (0.5d) → Gate 6 skeleton (0.5d)
Week 2:  Gate 1 (2d) → Gate 3 (2d) → Gate 4 (1d)
Week 3:  Gate 6 integration (1d) → Gate 7 eval (2d) → iterate on Gate 4 prompt
Week 4:  Gate 8 rosbag adapter (2d) → annotate first new scenes → Gate 7 on new data
Later:   Gate 9 training → Gate 10 VI fine-tuning
```

Start with Gates 0, 2, 5 — these have no dependencies and establish the scaffolding.
Gate 1 (rendering) requires Habitat container. Gate 3/4 require Gemma or GPT API access.

---

## Quality Targets

| Gate | Metric | Target |
|------|--------|--------|
| Gate 4 | Instruction length | 15–40 words (GT mean=26.8) |
| Gate 4 | Landmark mention rate | ≥1 per instruction (GT: ~100%) |
| Gate 4 | Stop condition present | 100% |
| Gate 7 | Habitat SR vs GT | within ±3pp (≥60.9%) |
| Gate 7 | BERTScore vs GT | ≥0.85 |
| Gate 7 | BLEU-4 vs GT | ≥0.15 |

---

## File Structure

```
s2_pipeline_new/
├── PLAN.md                          ← this file
├── configs/
│   ├── pipeline_config.yaml         ← paths, model choices, thresholds
│   └── vlm_prompts.yaml             ← prompt templates (editable without code changes)
├── gate0_analysis/
│   ├── analyze_gt.py                ← GT statistics analysis
│   └── gt_statistics.json           ← output (after running)
├── gate1_renderer/
│   ├── renderer.py                  ← Habitat scene renderer
│   └── render_batch.sh              ← batch render all val_unseen episodes
├── gate2_path/
│   ├── path_analyzer.py             ← motion primitive extractor
│   └── test_path_analyzer.py        ← unit tests
├── gate3_landmarks/
│   ├── landmark_detector.py         ← VLM landmark detection
│   └── test_landmarks.py            ← test on sample frames
├── gate4_instructions/
│   ├── instruction_generator.py     ← main instruction generator
│   ├── gemma_backend.py             ← Gemma 4 31B AWQ backend
│   ├── gpt_backend.py               ← GPT-4o backend
│   └── quality_filter.py            ← post-generation quality check
├── gate5_tokenizer/
│   └── tokenizer.py                 ← GT-vocabulary tokenizer
├── gate6_assembler/
│   ├── assembler.py                 ← episode JSON assembler
│   └── batch_assemble.py            ← batch process all episodes
├── gate7_eval/
│   ├── run_habitat_eval.sh          ← launch eval with generated dataset
│   ├── run_isaac_eval.sh            ← launch Isaac Lab eval
│   └── compare_results.py           ← GT vs generated metrics comparison
├── gate8_adapters/
│   ├── rosbag_adapter.py            ← ROS2 bag → frames + poses
│   ├── video_adapter.py             ← MP4 + CSV → frames + poses
│   └── isaac_adapter.py             ← IsaacSim → frames + poses
├── pipeline.py                      ← end-to-end runner (gates 1-6)
└── outputs/                         ← generated datasets, intermediate outputs
```
