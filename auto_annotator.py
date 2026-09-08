#!/usr/bin/env python3
"""
Auto-Annotator: Metadata Reproducer for VLN episodes.

Takes images + path data → rich navigation metadata + R2R-style instructions.
Works with:
  - Existing VLN-CE episode directories (with poses.json)
  - New/real-world paths (images + waypoints JSON)
  - Robot camera recordings (sequential images + odometry)

Phases:
  Phase A: Path analysis — extract motion primitives (turns, distances) from waypoints
  Phase B: Key frame selection — identify start, turn, goal frames
  Phase C: Vision description — LLM describes key frames (color, material, shape)
  Phase D: Instruction generation — LLM writes R2R-style navigation instruction
  Phase E: Metadata assembly — combine into complete per-episode JSON

Usage:
  # From existing VLN-CE episode directory:
  python auto_annotator.py --episode-dir outputs/rendered_frames/episode_000001/

  # From image directory + waypoints file:
  python auto_annotator.py --images /path/to/images/ --waypoints path_info.json

  # Batch: re-annotate all 1839 VLN-CE episodes with new vision prompts:
  python auto_annotator.py --batch --rendered-frames outputs/rendered_frames/ --output outputs/auto_metadata/

  # Single image dir, output to specific file:
  python auto_annotator.py --images /path/to/imgs/ --waypoints wp.json --output my_episode.json

  # Show quality metrics vs GT (if --gt-json provided):
  python auto_annotator.py --episode-dir ep_dir/ --gt-json val_unseen_patched.json.gz
"""

import argparse
import asyncio
import base64
import gzip
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Pipeline configuration ───────────────────────────────────────────────────

VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL    = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
VLLM_API_KEY  = "token-abc123"

PIPELINE_ROOT   = Path(__file__).parent
DEFAULT_OUT_DIR = PIPELINE_ROOT / "outputs" / "auto_metadata"

ANNOTATION_VERSION = "2.0-auto"


# ── Gate 2 imports (path analysis) ───────────────────────────────────────────

sys.path.insert(0, str(PIPELINE_ROOT))
try:
    from gate2_path.path_analyzer import analyze_path, primitives_to_text
except ImportError:
    print("[auto_annotator] WARNING: gate2_path not found — path analysis disabled.")
    def analyze_path(path, rot=None, **kw):
        return {"primitives": [{"type": "stop"}], "summary": {}, "key_frame_indices": [0], "segment_headings": []}
    def primitives_to_text(prims):
        return "Navigate to the destination."


# ── Vision prompts (Phase C) ──────────────────────────────────────────────────

VISION_PROMPT_START = (
    "A robot starts navigation from this indoor location. "
    "In 1-2 sentences: describe the most distinctive features of this starting area — "
    "what room type it is, and the 1-2 most prominent objects or architectural features visible. "
    "Examples: 'Starting in a bright hallway with brown wooden double doors on the right. "
    "The polished stone floor leads forward.' or 'A modern bathroom with white tile walls "
    "and a bathrobe hanging by the door. Straight ahead is a long corridor.' "
    "Reply with ONLY these 1-2 sentences, nothing else."
)

VISION_PROMPT_TURN = (
    "A robot is navigating indoors and is about to turn at this location. "
    "In 1-2 sentences: (1) describe the most prominent landmark at this turning point "
    "(color, material, shape), and (2) briefly note what's visible in the direction the "
    "robot will go after turning. Focus on navigation-useful details a person would remember. "
    "Examples: 'Turn at the white rectangular dining table with dark chairs. Ahead, a sunlit "
    "living room with hardwood floors opens up.' or 'The grey stone pillar marks the corner. "
    "A hallway with wooden panels continues to the left.' "
    "Reply with ONLY these 1-2 sentences, nothing else."
)

VISION_PROMPT_GOAL = (
    "A robot has arrived at its navigation destination. "
    "In 1-2 sentences, describe the stopping location: (1) the specific object that marks where "
    "to stop (color, material, shape), and (2) any distinctive context around it (window, wall, "
    "adjacent furniture). Do NOT use 'Stop' or 'Wait' — just describe what you see. "
    "Examples: 'The grey fabric lounge chair positioned against the window wall. Warm afternoon "
    "light falls across the wooden floor nearby.' or 'A white marble fireplace with a dark wooden "
    "mantel. It faces a seating area with couches on both sides.' "
    "Reply with ONLY the description, nothing else."
)

_VISION_PROMPTS = {"start": VISION_PROMPT_START, "goal": VISION_PROMPT_GOAL}


def get_vision_prompt(label: str) -> str:
    return _VISION_PROMPTS.get(label, VISION_PROMPT_TURN)


# ── Phase 2 instruction prompt builder ───────────────────────────────────────

INSTRUCTION_SYSTEM = (
    "You write concise R2R navigation instructions for an indoor robot. "
    "Style: Natural and brief, like a human R2R annotator. Match how real humans describe indoor navigation."
)

def build_instruction_prompt(path_analysis: Dict, vision_descs: Dict[str, str]) -> str:
    """Build Phase D LLM prompt from path analysis + vision descriptions.

    Vision-first approach: let visual landmarks anchor the instruction,
    not mechanical motion primitive descriptions that cause 'walk forward' overuse.
    """
    primitives = path_analysis.get("primitives", [])
    summary    = path_analysis.get("summary", {})

    # Summarize route structure (turns only — avoid listing every straight segment)
    n_left  = summary.get("n_left_turns", 0)
    n_right = summary.get("n_right_turns", 0)
    n_turns = n_left + n_right
    total_dist = summary.get("total_distance_m", 0)

    # Elevation
    has_elevation = any(p["type"] == "elevation" for p in primitives)
    elev_dir = next((p.get("direction", "up") for p in primitives if p["type"] == "elevation"), None)

    # Build turn list (direction + sharpness)
    turns = []
    for p in primitives:
        if p["type"] in ("left_turn", "right_turn"):
            d = "left" if p["type"] == "left_turn" else "right"
            s = "sharp " if p.get("sharp") else ""
            turns.append(f"{s}turn {d}")

    route_summary = []
    if has_elevation and elev_dir:
        route_summary.append(f"go {elev_dir} stairs/ramp")
    for t in turns:
        route_summary.append(t)
    route_summary.append("stop at destination")

    route_text = " → ".join(route_summary) if route_summary else "walk to destination → stop"

    # Build vision context — this is the PRIMARY guide for the instruction
    vision_lines = []
    start_desc = vision_descs.get("start")
    goal_desc  = vision_descs.get("goal")
    turn_descs = {k: v for k, v in vision_descs.items() if k.startswith("turn_") and v}

    if start_desc:
        vision_lines.append(f"Start: {start_desc}")
    for turn_key in sorted(turn_descs.keys()):
        turn_label = turn_key.replace("_", " ")
        vision_lines.append(f"{turn_label}: {turn_descs[turn_key]}")
    if goal_desc:
        vision_lines.append(f"Goal: {goal_desc}")

    vision_context = "\n".join(f"  {l}" for l in vision_lines) if vision_lines else "  (no visual context)"

    prompt = f"""Route: {route_text}
Total: {total_dist:.0f}m, {n_turns} turn{'s' if n_turns != 1 else ''}

Visual context:
{vision_context}

Write a concise R2R navigation instruction (2-3 sentences max).
Critical rules:
- Use vision context as your PRIMARY guide — mention specific visual landmarks
- AVOID 'walk forward' — instead describe WHERE to go ("walk through the hallway", "walk to the [object]")
- For straight segments between turns: just say "walk straight" OR describe what you pass
- For turns: use the visual landmark at the turn point ("turn left at the [object]")
- Use 'continue' at MOST once per instruction
- End: "stop at/near/by the [goal object]"
- 2-3 sentences only. Do NOT list every step mechanically.

Instruction:"""
    return prompt


# ── Image utilities ───────────────────────────────────────────────────────────

def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_vision_desc(raw: str) -> Optional[str]:
    """Post-process raw vision LLM output. Returns None if unusable."""
    raw = raw.strip().strip('"\'').strip()
    for prefix in ["Description:", "The landmark is", "I see", "I can see",
                   "In this image", "This image shows", "The image shows"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].strip(" :-").strip()
    sents = re.split(r'(?<=[.!?])\s+', raw)
    raw = " ".join(sents[:2]).strip()
    raw = re.sub(r"\s+", " ", raw).strip()
    words = raw.split()
    if len(words) < 3 or len(words) > 60:
        return None
    generic = {"a room", "an area", "furniture", "the room", "indoor", "a space"}
    if raw.lower() in generic:
        return None
    return raw


# ── Phase A: Path analysis ────────────────────────────────────────────────────

def analyze_episode_path(waypoints: List[List[float]], start_rotation: Optional[List[float]] = None) -> Dict:
    """Run gate2 path analysis on waypoints."""
    return analyze_path(waypoints, start_rotation)


# ── Phase B: Key frame selection ─────────────────────────────────────────────

def select_key_frames(
    frames: List[Dict],
    path_analysis: Dict,
) -> List[Dict]:
    """
    Select frames that are most navigationally informative.
    Returns frames annotated with label: start / turn_N / goal.
    """
    if not frames:
        return []

    # If frames already have labels (poses.json format), use them directly
    if all("label" in f for f in frames):
        return frames

    # Otherwise: assign labels based on path analysis key_frame_indices
    key_indices = set(path_analysis.get("key_frame_indices", [0, len(frames) - 1]))
    key_indices.add(0)
    key_indices.add(len(frames) - 1)

    labeled = []
    turn_count = 0
    for i, f in enumerate(frames):
        if i not in key_indices:
            continue
        frame = dict(f)
        frame["waypoint_idx"] = i
        if i == 0:
            frame["label"] = "start"
        elif i == len(frames) - 1:
            frame["label"] = "goal"
        else:
            turn_count += 1
            frame["label"] = f"turn_{i}"
        labeled.append(frame)

    return labeled


# ── Phase C: Vision description (async) ──────────────────────────────────────

async def describe_frame_async(
    client,
    frame: Dict,
    sem: asyncio.Semaphore,
    done: List[int],
    total: int,
    t0: float,
) -> Dict:
    """Async LLM vision call for one key frame."""
    image_path = Path(frame.get("image_path", frame.get("path", "")))
    label = frame.get("label", "turn")

    async with sem:
        try:
            b64 = image_to_base64(image_path)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text",      "text": get_vision_prompt(label)},
                ],
            }]
            resp = await client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                max_tokens=80,
                temperature=0.20,
            )
            raw  = resp.choices[0].message.content.strip()
            desc = clean_vision_desc(raw)
            result = {"label": label, "desc": desc, "raw": raw, "ok": True}
        except Exception as e:
            result = {"label": label, "desc": None, "raw": "", "ok": False, "error": str(e)}

    done[0] += 1
    if done[0] % 10 == 0 or done[0] == total:
        elapsed = time.time() - t0
        rate = done[0] / elapsed if elapsed > 0 else 0.001
        eta = (total - done[0]) / rate if rate > 0 else 0
        print(f"  [Vision {done[0]}/{total}] {rate:.1f}/s  ETA={eta/60:.1f}m", flush=True)

    return result


async def run_phase_c(key_frames: List[Dict], concurrency: int = 8) -> Dict[str, Optional[str]]:
    """
    Phase C: Vision description for all key frames.
    Returns {label: description_or_None}.
    """
    from openai import AsyncOpenAI

    print(f"\n=== Phase C: Vision Description ===")
    print(f"  Frames: {len(key_frames)}  Concurrency: {concurrency}")

    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    sem  = asyncio.Semaphore(concurrency)
    done = [0]
    t0   = time.time()

    coros = [describe_frame_async(client, f, sem, done, len(key_frames), t0) for f in key_frames]
    results = await asyncio.gather(*coros)

    descs: Dict[str, Optional[str]] = {}
    for r in results:
        descs[r["label"]] = r["desc"]

    ok   = sum(1 for d in descs.values() if d)
    fail = sum(1 for d in descs.values() if not d)
    print(f"\nPhase C done: {ok} descriptions, {fail} failed")
    return descs


# ── Phase D: Instruction generation ──────────────────────────────────────────

async def run_phase_d(path_analysis: Dict, vision_descs: Dict[str, Optional[str]]) -> str:
    """Phase D: Generate R2R instruction from metadata."""
    from openai import AsyncOpenAI

    print("\n=== Phase D: Instruction Generation ===")

    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    prompt = build_instruction_prompt(path_analysis, {k: v for k, v in vision_descs.items() if v})

    try:
        resp = await client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": INSTRUCTION_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=120,
            temperature=0.30,
        )
        instruction = resp.choices[0].message.content.strip()
        instruction = re.sub(r"^Instruction:\s*", "", instruction, flags=re.IGNORECASE)
        instruction = instruction.strip()
        print(f"  Generated: {instruction[:100]}...")
        return instruction
    except Exception as e:
        print(f"  Phase D ERROR: {e}")
        return primitives_to_text(path_analysis.get("primitives", []))


# ── Input loading helpers ─────────────────────────────────────────────────────

def load_poses(episode_dir: Path) -> Optional[Dict]:
    """Load poses.json from a VLN-CE episode directory."""
    p = episode_dir / "poses.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_images_from_dir(image_dir: Path) -> List[Path]:
    """Load image files from directory, sorted by name."""
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])


def load_waypoints_json(path: Path) -> Tuple[List[List[float]], Optional[List[float]]]:
    """
    Load waypoints from JSON file.
    Accepts:
      {"waypoints": [[x,y,z], ...], "start_rotation": [qx,qy,qz,qw]}  # recommended
      {"reference_path": [[x,y,z], ...], "start_rotation": [...]}      # VLN-CE format
      [[x,y,z], ...]                                                    # bare list
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data, None

    waypoints = data.get("waypoints") or data.get("reference_path") or []
    rot = data.get("start_rotation")
    return waypoints, rot


def frames_from_image_list(image_paths: List[Path]) -> List[Dict]:
    """Convert image path list into frame dicts (no labels — assigned later)."""
    return [
        {"frame_idx": i, "image_path": str(p), "path": str(p)}
        for i, p in enumerate(image_paths)
    ]


def frames_from_poses(episode_dir: Path, poses: Dict) -> List[Dict]:
    """Convert poses.json frames into frame dicts with absolute image paths."""
    result = []
    for f in poses.get("frames", []):
        frame = dict(f)
        rel_path = frame.get("path", "")
        abs_path = episode_dir / rel_path
        frame["image_path"] = str(abs_path)
        result.append(frame)
    return result


# ── Phase E: Metadata assembly ────────────────────────────────────────────────

def assemble_metadata(
    episode_id: int,
    scene_id: str,
    reference_path: List[List[float]],
    start_rotation: Optional[List[float]],
    key_frames: List[Dict],
    path_analysis: Dict,
    vision_descs: Dict[str, Optional[str]],
    instruction: str,
    gt_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble complete metadata JSON matching complete_metadata format."""
    return {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "start_position": reference_path[0] if reference_path else None,
        "start_rotation": start_rotation,
        "reference_path": reference_path,
        "path_analysis": path_analysis,
        "rendered_frames": key_frames,
        "n_frames": len(key_frames),
        "vision_descriptions": vision_descs,
        "generated_instruction": {
            "text": instruction,
            "generator": VLLM_MODEL.split("/")[-1],
            "version": ANNOTATION_VERSION,
        },
        "gt_instruction": {"text": gt_instruction} if gt_instruction else None,
        "_annotation_version": ANNOTATION_VERSION,
        "_annotation_sources": {
            "path_analysis": "gate2_path.path_analyzer",
            "vision": f"vllm:{VLLM_MODEL}",
            "instruction": f"vllm:{VLLM_MODEL}",
        },
    }


# ── VLN-CE dataset export ─────────────────────────────────────────────────────

def export_to_vlnce_dataset(
    metadata_list: List[Dict],
    output_path: Path,
    gt_path: Optional[Path] = None,
) -> None:
    """
    Export auto-annotated metadata to VLN-CE dataset format (.json.gz).
    Output is ready for direct use with Habitat eval (val_unseen_auto_v*.json.gz).

    Reuses instruction_vocab from GT dataset to ensure token compatibility.
    """
    # Load tokenizer and GT vocab
    try:
        from gate5_tokenizer.tokenizer import VLNTokenizer
        vocab_source = str(gt_path) if gt_path else None
        tokenizer = VLNTokenizer(vocab_source or PIPELINE_ROOT / "gate5_tokenizer" / "tokenizer.py")
    except Exception as e:
        print(f"[export] Tokenizer unavailable ({e}), using empty tokens")
        tokenizer = None

    # Load GT data for structural fields (info, goals, trajectory_id)
    gt_map: Dict[int, Dict] = {}
    gt_vocab = {}
    if gt_path and gt_path.exists():
        with gzip.open(gt_path, "rt") as f:
            gt_data = json.load(f)
        gt_map = {ep["episode_id"]: ep for ep in gt_data["episodes"]}
        gt_vocab = gt_data.get("instruction_vocab", {})

    episodes_out = []
    for meta in metadata_list:
        ep_id = meta["episode_id"]
        gt_ep = gt_map.get(ep_id, {})

        instruction_text = meta.get("generated_instruction", {}).get("text", "")
        tokens = tokenizer.encode(instruction_text) if tokenizer else ([0] * 200)

        episode = {
            "episode_id": ep_id,
            "trajectory_id": gt_ep.get("trajectory_id", ep_id),
            "scene_id": meta.get("scene_id", gt_ep.get("scene_id", "unknown")),
            "start_position": meta.get("start_position") or gt_ep.get("start_position"),
            "start_rotation": meta.get("start_rotation") or gt_ep.get("start_rotation"),
            "info": gt_ep.get("info", {}),
            "goals": gt_ep.get("goals", []),
            "instruction": {
                "instruction_text": instruction_text,
                "instruction_tokens": tokens,
            },
            "reference_path": meta.get("reference_path") or gt_ep.get("reference_path", []),
        }
        episodes_out.append(episode)

    dataset = {
        "episodes": episodes_out,
        "instruction_vocab": gt_vocab,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt") as f:
        json.dump(dataset, f)

    print(f"[export] VLN-CE dataset saved: {output_path} ({len(episodes_out)} episodes)")

    # Coverage check
    if tokenizer:
        coverages = [tokenizer.coverage(m.get("generated_instruction", {}).get("text", ""))
                     for m in metadata_list]
        avg_cov = sum(coverages) / len(coverages) if coverages else 0
        print(f"[export] Avg vocab coverage: {avg_cov:.3f} (1.0 = all words in VLN vocab)")


# ── Quality metrics (optional) ────────────────────────────────────────────────

def compute_quality_metrics(generated: str, gt: str) -> Dict[str, float]:
    """Compute text similarity metrics between generated and GT instruction."""
    metrics: Dict[str, float] = {}
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        import nltk
        gen_tokens = generated.lower().split()
        gt_tokens  = gt.lower().split()
        sf = SmoothingFunction().method1
        metrics["bleu_1"] = sentence_bleu([gt_tokens], gen_tokens, weights=(1,0,0,0), smoothing_function=sf)
        metrics["bleu_2"] = sentence_bleu([gt_tokens], gen_tokens, weights=(0.5,0.5,0,0), smoothing_function=sf)
    except Exception:
        pass
    try:
        from rouge_score import rouge_scorer as rs_lib
        scorer = rs_lib.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(gt, generated)
        metrics["rouge_l"] = scores["rougeL"].fmeasure
    except Exception:
        pass

    # Word overlap (simple)
    gen_words = set(re.findall(r'\b\w+\b', generated.lower()))
    gt_words  = set(re.findall(r'\b\w+\b', gt.lower()))
    if gt_words:
        metrics["word_recall"]    = len(gen_words & gt_words) / len(gt_words)
        metrics["word_precision"] = len(gen_words & gt_words) / len(gen_words) if gen_words else 0.0
        p, r = metrics["word_precision"], metrics["word_recall"]
        metrics["word_f1"] = 2*p*r/(p+r) if (p+r) > 0 else 0.0

    return metrics


# ── Main annotation pipeline ──────────────────────────────────────────────────

async def annotate_episode(
    episode_id: int,
    reference_path: List[List[float]],
    start_rotation: Optional[List[float]],
    frames: List[Dict],
    scene_id: str = "unknown",
    gt_instruction: Optional[str] = None,
    concurrency: int = 8,
    skip_phase_c: bool = False,
    skip_phase_d: bool = False,
) -> Dict[str, Any]:
    """Full annotation pipeline for one episode."""

    print(f"\n{'='*60}")
    print(f"Episode {episode_id} | scene={scene_id} | {len(reference_path)} waypoints | {len(frames)} key frames")

    # Phase A: Path analysis
    print("\n=== Phase A: Path Analysis ===")
    path_analysis = analyze_episode_path(reference_path, start_rotation)
    prims = path_analysis.get("primitives", [])
    summ  = path_analysis.get("summary", {})
    print(f"  Primitives: {[p['type'] for p in prims]}")
    print(f"  Distance: {summ.get('total_distance_m', 0):.1f}m  "
          f"Turns: L={summ.get('n_left_turns',0)} R={summ.get('n_right_turns',0)}")

    # Phase B: Key frame selection / label assignment
    print("\n=== Phase B: Key Frame Selection ===")
    key_frames = select_key_frames(frames, path_analysis)
    print(f"  {len(key_frames)} key frames: {[f['label'] for f in key_frames]}")

    # Phase C: Vision description
    vision_descs: Dict[str, Optional[str]] = {}
    if not skip_phase_c and key_frames:
        vision_descs = await run_phase_c(key_frames, concurrency=concurrency)
        for label, desc in vision_descs.items():
            print(f"  [{label}] {desc or 'N/A'}")
    else:
        print("  Phase C: skipped")

    # Phase D: Instruction generation
    instruction = ""
    if not skip_phase_d:
        instruction = await run_phase_d(path_analysis, vision_descs)
    else:
        print("  Phase D: skipped — using motion_text fallback")
        instruction = primitives_to_text(prims)

    # Phase E: Metadata assembly
    metadata = assemble_metadata(
        episode_id, scene_id, reference_path, start_rotation,
        key_frames, path_analysis, vision_descs, instruction, gt_instruction
    )

    # Quality metrics (if GT available)
    if gt_instruction:
        metrics = compute_quality_metrics(instruction, gt_instruction)
        metadata["quality_metrics"] = metrics
        print(f"\n  Quality vs GT:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.3f}")

    return metadata


# ── Batch processing ──────────────────────────────────────────────────────────

async def batch_annotate(
    rendered_frames_dir: Path,
    output_dir: Path,
    gt_path: Optional[Path] = None,
    episode_ids: Optional[List[int]] = None,
    concurrency: int = 8,
    ep_parallelism: int = 8,
    overwrite: bool = False,
    max_episodes: Optional[int] = None,
) -> None:
    """
    Batch annotation of all episodes in rendered_frames_dir.

    ep_parallelism: Number of episodes processed concurrently (default: 8).
    concurrency: Number of concurrent LLM calls within each episode (Phase C).
    Total concurrent LLM calls = ep_parallelism × avg_frames_per_ep × concurrency.
    Recommended: ep_parallelism=8, concurrency=4 for ~32 concurrent calls (safe for vLLM).
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load GT if provided
    gt_map: Dict[int, Dict] = {}
    if gt_path and gt_path.exists():
        print(f"Loading GT from {gt_path}...")
        with gzip.open(gt_path, "rt") as f:
            gt_data = json.load(f)
        gt_map = {ep["episode_id"]: ep for ep in gt_data["episodes"]}
        print(f"  Loaded {len(gt_map)} GT episodes")

    # Discover episodes
    ep_dirs = sorted(rendered_frames_dir.glob("episode_*"))
    if not ep_dirs:
        print(f"ERROR: No episode directories found in {rendered_frames_dir}")
        return

    if episode_ids:
        ep_dirs = [d for d in ep_dirs if int(d.name.split("_")[1]) in set(episode_ids)]

    if max_episodes:
        ep_dirs = ep_dirs[:max_episodes]

    print(f"\nBatch annotating {len(ep_dirs)} episodes (ep_parallelism={ep_parallelism}, concurrency={concurrency})...")

    results = {"processed": 0, "skipped": 0, "failed": 0}
    quality_agg: Dict[str, List[float]] = {}
    done_total = [0]
    t_start = time.time()

    async def process_one(ep_dir: Path) -> Optional[Dict]:
        """Annotate a single episode and save output. Returns metadata or None."""
        ep_id = int(ep_dir.name.split("_")[1])
        out_path = output_dir / f"episode_{ep_id:06d}.json"

        if out_path.exists() and not overwrite:
            results["skipped"] += 1
            return None

        poses = load_poses(ep_dir)
        if poses is None:
            results["skipped"] += 1
            return None

        gt_ep = gt_map.get(ep_id, {})
        reference_path = gt_ep.get("reference_path", [])
        start_rotation = gt_ep.get("start_rotation")
        scene_id = gt_ep.get("scene_id", ep_dir.name)

        if not reference_path:
            reference_path = [f["position"] for f in poses.get("frames", []) if "position" in f]

        gt_instr = None
        gt_instructions = gt_ep.get("instruction", {})
        if isinstance(gt_instructions, list) and gt_instructions:
            gt_instr = gt_instructions[0].get("instruction_text") or gt_instructions[0].get("instruction", "")
        elif isinstance(gt_instructions, dict):
            gt_instr = gt_instructions.get("instruction_text") or gt_instructions.get("instruction", "")

        frames = frames_from_poses(ep_dir, poses)

        try:
            metadata = await annotate_episode(
                ep_id, reference_path, start_rotation, frames,
                scene_id=scene_id, gt_instruction=gt_instr, concurrency=concurrency
            )
            with open(out_path, "w") as f:
                json.dump(metadata, f, indent=2)
            results["processed"] += 1

            done_total[0] += 1
            elapsed = time.time() - t_start
            rate = done_total[0] / elapsed if elapsed > 0 else 0.001
            eta = (len(ep_dirs) - done_total[0]) / rate if rate > 0 else 0
            print(f"  [Batch {done_total[0]}/{len(ep_dirs)}] {rate:.2f} ep/s  ETA={eta/60:.1f}m", flush=True)

            return metadata
        except Exception as e:
            print(f"  ERROR episode {ep_id}: {e}")
            results["failed"] += 1
            return None

    # Process episodes in parallel batches of ep_parallelism
    ep_sem = asyncio.Semaphore(ep_parallelism)

    async def process_with_sem(ep_dir: Path) -> Optional[Dict]:
        async with ep_sem:
            return await process_one(ep_dir)

    all_results = await asyncio.gather(*[process_with_sem(d) for d in ep_dirs])
    all_meta_out = [m for m in all_results if m is not None and "generated_instruction" in m]

    print(f"\n{'='*60}")
    print(f"Batch complete: {results['processed']} processed, "
          f"{results['skipped']} skipped, {results['failed']} failed")

    # Aggregate quality metrics from completed results
    for m in all_meta_out:
        if "quality_metrics" in m:
            for k, v in m["quality_metrics"].items():
                quality_agg.setdefault(k, []).append(v)

    if quality_agg:
        print("\nAggregate quality metrics:")
        for k, vals in quality_agg.items():
            print(f"  {k}: {sum(vals)/len(vals):.3f} (n={len(vals)})")

    # Save batch summary
    summary = {
        "n_episodes": len(ep_dirs),
        "results": results,
        "quality_aggregate": {k: sum(v)/len(v) for k, v in quality_agg.items()} if quality_agg else {},
        "annotation_version": ANNOTATION_VERSION,
    }
    with open(output_dir / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir / 'batch_summary.json'}")

    # Export VLN-CE dataset from all available episode files
    if results["processed"] > 0:
        if not all_meta_out:
            # Fall back to reading files if in-memory list is empty
            for ep_dir in ep_dirs:
                ep_id = int(ep_dir.name.split("_")[1])
                meta_path = output_dir / f"episode_{ep_id:06d}.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        all_meta_out.append(json.load(f))
        if all_meta_out:
            dataset_path = output_dir / f"val_unseen_auto_{ANNOTATION_VERSION.replace('.', '_')}.json.gz"
            export_to_vlnce_dataset(all_meta_out, dataset_path, gt_path)
            print(f"\nVLN-CE dataset ready for evaluation: {dataset_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Auto-annotate VLN episodes from images + path data.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--episode-dir",    type=Path, help="VLN-CE episode dir with poses.json + images")
    mode.add_argument("--images",         type=Path, help="Directory of image files (sequential)")
    mode.add_argument("--batch",          action="store_true", help="Batch mode: process all episodes")

    # Single episode options
    ap.add_argument("--waypoints",      type=Path, help="JSON file with waypoints + start_rotation")
    ap.add_argument("--episode-id",     type=int,  default=0, help="Episode ID (for output naming)")
    ap.add_argument("--scene-id",       type=str,  default="unknown", help="Scene identifier")
    ap.add_argument("--gt-instruction", type=str,  default=None, help="Ground truth instruction for quality metrics")
    ap.add_argument("--gt-json",        type=Path, default=None, help="GT episodes JSON.gz (for GT lookup)")

    # Batch options
    ap.add_argument("--rendered-frames", type=Path,
                    default=PIPELINE_ROOT / "outputs" / "rendered_frames",
                    help="Batch: root dir of rendered_frames/episode_*/ directories")

    # Output
    ap.add_argument("--output",    type=Path, default=None, help="Output path (file for single, dir for batch)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    # Runtime
    ap.add_argument("--concurrency",    type=int, default=4,  help="Concurrent LLM calls within each episode (Phase C)")
    ap.add_argument("--ep-parallelism", type=int, default=10, help="Episodes processed concurrently (default: 10)")
    ap.add_argument("--skip-vision",    action="store_true",  help="Skip Phase C (vision description)")
    ap.add_argument("--skip-instr",     action="store_true",  help="Skip Phase D (instruction generation)")
    ap.add_argument("--max-episodes",   type=int, default=None, help="Batch: max episodes to process")
    ap.add_argument("--episodes",       type=int, nargs="+",  help="Batch: specific episode IDs")

    return ap.parse_args()


async def main():
    args = parse_args()

    if args.batch:
        # Batch mode
        gt_path = args.gt_json
        if gt_path is None:
            default_gt = PIPELINE_ROOT / "outputs" / "datasets" / "val_unseen_patched.json.gz"
            if not default_gt.exists():
                default_gt = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
            gt_path = default_gt if default_gt.exists() else None

        out_dir = args.output or DEFAULT_OUT_DIR
        await batch_annotate(
            rendered_frames_dir=args.rendered_frames,
            output_dir=out_dir,
            gt_path=gt_path,
            episode_ids=args.episodes,
            concurrency=args.concurrency,
            ep_parallelism=args.ep_parallelism,
            overwrite=args.overwrite,
            max_episodes=args.max_episodes,
        )
        return

    # Single episode mode
    reference_path: List[List[float]] = []
    start_rotation: Optional[List[float]] = None
    frames: List[Dict] = []
    scene_id = args.scene_id
    ep_id = args.episode_id

    if args.episode_dir:
        # VLN-CE episode directory
        ep_dir = args.episode_dir
        poses = load_poses(ep_dir)
        if poses is None:
            print(f"ERROR: No poses.json in {ep_dir}")
            sys.exit(1)
        frames = frames_from_poses(ep_dir, poses)
        ep_id = poses.get("episode_id", ep_id)
        scene_id = poses.get("scene_id", scene_id)

        # Try to load waypoints from GT if available
        if args.gt_json and args.gt_json.exists():
            with gzip.open(args.gt_json, "rt") as f:
                gt_data = json.load(f)
            gt_ep = next((ep for ep in gt_data["episodes"] if ep["episode_id"] == ep_id), None)
            if gt_ep:
                reference_path = gt_ep.get("reference_path", [])
                start_rotation = gt_ep.get("start_rotation")
                if not args.gt_instruction:
                    instrs = gt_ep.get("instruction", {})
                    if isinstance(instrs, list) and instrs:
                        args.gt_instruction = instrs[0].get("instruction_text") or instrs[0].get("instruction", "")
                    elif isinstance(instrs, dict):
                        args.gt_instruction = instrs.get("instruction_text") or instrs.get("instruction", "")

        if not reference_path:
            # Fall back to frame positions
            reference_path = [f["position"] for f in frames if "position" in f]

    elif args.images:
        # Arbitrary image directory
        image_files = load_images_from_dir(args.images)
        if not image_files:
            print(f"ERROR: No images found in {args.images}")
            sys.exit(1)
        frames = frames_from_image_list(image_files)
        print(f"Loaded {len(image_files)} images from {args.images}")

        if args.waypoints:
            reference_path, start_rotation = load_waypoints_json(args.waypoints)
            print(f"Loaded {len(reference_path)} waypoints from {args.waypoints}")
        else:
            print("WARNING: No waypoints provided. Path analysis will be minimal.")
            reference_path = [[i, 0, 0] for i in range(len(frames))]

    # GT instruction
    gt_instr = args.gt_instruction

    # Run annotation
    metadata = await annotate_episode(
        ep_id, reference_path, start_rotation, frames,
        scene_id=scene_id, gt_instruction=gt_instr,
        concurrency=args.concurrency,
        skip_phase_c=args.skip_vision,
        skip_phase_d=args.skip_instr,
    )

    # Output
    out_path = args.output
    if out_path is None:
        out_path = DEFAULT_OUT_DIR / f"episode_{ep_id:06d}.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {out_path}")

    # Print summary
    instr = metadata.get("generated_instruction", {})
    print(f"\n{'='*60}")
    print(f"Generated instruction: {instr.get('text', 'N/A')}")
    if gt_instr:
        print(f"GT instruction:        {gt_instr}")
    if "quality_metrics" in metadata:
        print(f"Quality metrics:       {metadata['quality_metrics']}")


if __name__ == "__main__":
    asyncio.run(main())
