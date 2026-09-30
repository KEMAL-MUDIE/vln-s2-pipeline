#!/usr/bin/env python3
"""
Batch Auto-Annotator — complete metadata for val_unseen / val_seen / train
===========================================================================
Uses the Gemma-4-31B VLM (complete_auto_annotator.py pipeline) to produce
rich complete metadata for every episode in a dataset split.

Output per split:
  outputs/complete_metadata_unseen_v272/   (1839 episodes)
  outputs/complete_metadata_seen_v272/     (778 episodes)
  outputs/complete_metadata_train_v272/    (10819 episodes, needs rendered frames)

Final packed datasets (assembled after annotation completes):
  outputs/datasets/val_unseen_complete_v272.json.gz
  outputs/datasets/val_seen_complete_v272.json.gz
  outputs/datasets/train_complete_v272.json.gz

Usage:
  python3 run_batch_annotator.py --split val_unseen [--workers 8] [--resume]
  python3 run_batch_annotator.py --split val_seen   [--workers 8] [--resume]
  python3 run_batch_annotator.py --split train      [--workers 4] [--resume]

GPU note: rendering train frames requires GPU2 to be free (not used by eval).
"""

import argparse
import gzip
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT))

# ── vLLM endpoint ─────────────────────────────────────────────────────────────
VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL    = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
VLLM_API_KEY  = "token-abc123"

ANNOTATION_VERSION = "3.0-complete-v272"

# ── Split configuration ───────────────────────────────────────────────────────
HABITAT_DATA = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")

SPLIT_CONFIG = {
    "val_unseen": {
        "gt_path":     HABITAT_DATA / "val_unseen/val_unseen.json.gz",
        "frames_dir":  PIPELINE_ROOT / "outputs/rendered_frames",
        "out_dir":     PIPELINE_ROOT / "outputs/complete_metadata_unseen_v272",
        "dataset_out": PIPELINE_ROOT / "outputs/datasets/val_unseen_complete_v272.json.gz",
    },
    "val_seen": {
        "gt_path":     HABITAT_DATA / "val_seen/val_seen.json.gz",
        "frames_dir":  PIPELINE_ROOT / "outputs/rendered_frames_val_seen",
        "out_dir":     PIPELINE_ROOT / "outputs/complete_metadata_seen_v272",
        "dataset_out": PIPELINE_ROOT / "outputs/datasets/val_seen_complete_v272.json.gz",
    },
    "train": {
        "gt_path":     HABITAT_DATA / "train/train.json.gz",
        "frames_dir":  PIPELINE_ROOT / "outputs/rendered_frames_train",
        "out_dir":     PIPELINE_ROOT / "outputs/complete_metadata_train_v272",
        "dataset_out": PIPELINE_ROOT / "outputs/datasets/train_complete_v272.json.gz",
    },
}

# ── Prompts (match complete_auto_annotator.py) ────────────────────────────────
SCENE_DESCRIPTION_PROMPT = """\
A robot is navigating through this indoor environment.
Describe what you see in 1-2 sentences focusing on:
1. The ROOM TYPE (bedroom, kitchen, hallway, living room, staircase, etc.)
2. The 2-3 most distinctive LANDMARKS visible (furniture, doorways, rugs, appliances, decor)
3. The apparent NAVIGATION DIRECTION (straight ahead, turning left/right, approaching stairs)

Format: ROOM: <type> | LANDMARKS: <item1>, <item2>, <item3> | DIRECTION: <straight|left|right|stairs|unclear>
"""

GOAL_LANDMARK_PROMPT = """\
This is the FINAL frame — the destination where the robot should stop.
Describe the stopping location in 1 sentence suitable for use as a navigation landmark.
Focus on the most distinctive object or feature at/near the stopping point.
Examples: "near the wooden chair by the window", "at the foot of the stairs", "beside the kitchen island"
Write ONLY the landmark description (no full sentence prefix):
"""

INSTRUCTION_GENERATION_PROMPT = """\
You are a navigation instruction writer for vision-language robot navigation.

Write a concise, natural navigation instruction for this path.

PATH MOTION SEQUENCE:
{motion_text}

SCENE CONTEXT AT KEY POINTS:
{scene_context}

GOAL/DESTINATION:
{goal_landmark}

REQUIREMENTS:
- 1-4 short sentences, 15-40 words total
- Reference specific visible landmarks (furniture, rooms, doorways, etc.)
- Use natural turn language: "turn left/right", "make a left", "go left at"
- MUST include a clear stop condition: "stop at/near/by [landmark]" or "wait at [landmark]"
- No distances in metres or numbers
- Concise, direct — like giving directions to a person

GT-STYLE EXAMPLES (do NOT copy — style reference only):
- "Exit the bedroom and turn left. Walk past the gray couch and stop near the rug."
- "Walk through the kitchen doorway, turn right at the dining table, and stop in the hallway."
- "Go straight through the foyer. Turn left at the stairs and wait at the bottom."

Write ONLY the instruction text:
"""


# ── vLLM client ───────────────────────────────────────────────────────────────
def _get_openai_client():
    from openai import OpenAI
    return OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


import base64
def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _describe_image(client, img_path: str, prompt: str, max_tokens: int = 120) -> str:
    b64 = _encode_image(img_path)
    resp = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def _generate_text(client, prompt: str, max_tokens: int = 150, temperature: float = 0.5) -> str:
    resp = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# ── Path analysis ─────────────────────────────────────────────────────────────
def _path_analysis(reference_path: List, start_rotation: List) -> Dict:
    try:
        from gate2_path.path_analyzer import analyze_path, primitives_to_text
        analysis = analyze_path(reference_path, start_rotation)
        analysis["motion_text"] = primitives_to_text(analysis["primitives"])
        return analysis
    except Exception as e:
        path_length = sum(
            math.sqrt(sum((b[i]-a[i])**2 for i in range(3)))
            for a, b in zip(reference_path[:-1], reference_path[1:])
        ) if len(reference_path) >= 2 else 0.0
        n = len(reference_path)
        key_frames = [0, n//2, n-1] if n >= 3 else list(range(n))
        return {
            "primitives": [{"type": "straight", "distance_m": round(path_length, 2)}, {"type": "stop"}],
            "summary": {"path_length_m": round(path_length, 2), "n_turns": 0},
            "motion_text": f"Walk forward {path_length:.1f}m and stop.",
            "key_frame_indices": key_frames,
            "segment_headings": [],
        }


def _parse_scene_description(raw: str) -> Dict:
    result = {"room": "unknown", "landmarks": [], "direction": "unclear", "raw": raw}
    for part in raw.split("|"):
        part = part.strip()
        if part.startswith("ROOM:"):
            result["room"] = part[5:].strip().lower()
        elif part.startswith("LANDMARKS:"):
            result["landmarks"] = [x.strip() for x in part[10:].split(",") if x.strip()]
        elif part.startswith("DIRECTION:"):
            result["direction"] = part[10:].strip().lower()
    return result


# ── Core annotation for one episode ──────────────────────────────────────────
def annotate_episode(ep_id: int, gt_ep: Dict, frames_dir: Path) -> Dict:
    """
    Full annotation pipeline for one episode.
    Returns complete metadata dict.
    Raises on hard failure (missing frames).
    """
    t0 = time.time()
    client = _get_openai_client()

    # Load rendered frames
    ep_dir = frames_dir / f"episode_{ep_id:06d}"
    frame_paths = []
    if ep_dir.exists():
        frame_paths = sorted([
            str(p) for p in ep_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

    if not frame_paths:
        raise FileNotFoundError(f"No frames for ep{ep_id} at {ep_dir}")

    # Load poses
    poses = None
    poses_path = ep_dir / "poses.json"
    if poses_path.exists():
        with open(poses_path) as f:
            poses = json.load(f)

    reference_path = gt_ep.get("reference_path", [])
    start_rotation = gt_ep.get("start_rotation", [0, 0, 0, 1])
    gt_instruction = gt_ep.get("instruction", {}).get("instruction_text", "")

    # Gate 2: path analysis
    path_analysis = _path_analysis(reference_path, start_rotation)
    key_frame_indices = path_analysis.get("key_frame_indices", list(range(len(frame_paths))))
    valid_key_frames = [i for i in key_frame_indices if i < len(frame_paths)]

    # Gate 3: vision annotation — per-frame scene descriptions
    per_frame = {}
    for fidx in valid_key_frames:
        img_path = frame_paths[fidx]
        if not Path(img_path).exists():
            continue
        try:
            raw = _describe_image(client, img_path, SCENE_DESCRIPTION_PROMPT, max_tokens=120)
            per_frame[str(fidx)] = _parse_scene_description(raw)
        except Exception as e:
            per_frame[str(fidx)] = {"room": "unknown", "landmarks": [], "direction": "unclear", "error": str(e)}

    # Goal landmark: last frame
    goal_landmark = {"description": "the destination", "raw": ""}
    if frame_paths and Path(frame_paths[-1]).exists():
        try:
            raw_goal = _describe_image(client, frame_paths[-1], GOAL_LANDMARK_PROMPT, max_tokens=80)
            goal_landmark = {"description": raw_goal, "raw": raw_goal}
        except Exception as e:
            goal_landmark["error"] = str(e)

    landmark_annotations = {
        "per_frame": per_frame,
        "scene_context": per_frame.get(str(valid_key_frames[0]), {}) if valid_key_frames else {},
        "goal_landmark": goal_landmark,
        "n_frames_annotated": len(per_frame),
    }

    # Gate 4: instruction generation
    motion_text = path_analysis.get("motion_text", "Navigate to the destination.")
    scene_context_text = "\n".join(
        f"  [Frame {k}] {v.get('room','?').title()} — {', '.join(v.get('landmarks',[]))}, dir: {v.get('direction','?')}"
        for k, v in sorted(per_frame.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0)
    ) or "  Indoor environment."
    goal_desc = goal_landmark.get("description", "the destination")

    prompt = INSTRUCTION_GENERATION_PROMPT.format(
        motion_text=motion_text,
        scene_context=scene_context_text,
        goal_landmark=goal_desc,
    )

    generated_text = ""
    quality_ok = False
    issues = []
    for attempt in range(3):
        try:
            generated_text = _generate_text(client, prompt, max_tokens=120, temperature=0.4 + attempt * 0.1)
            words = generated_text.split()
            issues = []
            if len(words) < 8:
                issues.append(f"too_short({len(words)}w)")
            if not any(w in generated_text.lower() for w in ["stop","wait","stand","remain","halt"]):
                issues.append("no_stop")
            quality_ok = len(issues) == 0
            if quality_ok:
                break
        except Exception as e:
            issues = [str(e)]

    elapsed = time.time() - t0
    path_len = sum(
        math.sqrt(sum((b[i]-a[i])**2 for i in range(3)))
        for a, b in zip(reference_path[:-1], reference_path[1:])
    ) if len(reference_path) >= 2 else 0.0

    return {
        "episode_id": ep_id,
        "trajectory_id": gt_ep.get("trajectory_id"),
        "scene_id": gt_ep.get("scene_id", ""),
        "start_position": gt_ep.get("start_position", []),
        "start_rotation": start_rotation,
        "reference_path": reference_path,
        "goals": gt_ep.get("goals", []),
        "info": {
            "geodesic_distance": gt_ep.get("info", {}).get("geodesic_distance", 0.0),
            "path_length_m": round(path_len, 3),
        },
        "path_analysis": path_analysis,
        "rendered_frames": frame_paths,
        "n_frames": len(frame_paths),
        "landmark_annotations": landmark_annotations,
        "generated_instruction": {
            "text": generated_text,
            "generator": VLLM_MODEL,
            "version": ANNOTATION_VERSION,
            "quality_ok": quality_ok,
            "quality_issues": issues,
        },
        "gt_instruction": gt_instruction,
        "_annotation_version": ANNOTATION_VERSION,
        "_annotation_sources": ["gate1_renderer", "gate2_path", "gate3_gemma4_vision", "gate4_instruction"],
        "_processing_time_s": round(elapsed, 2),
        "_poses": poses,
    }


# ── Dataset assembler ─────────────────────────────────────────────────────────
def assemble_dataset(split: str, gt_data: Dict, out_dir: Path, dataset_out: Path):
    """Pack completed metadata files into a Habitat-compatible JSON.gz dataset."""
    with gzip.open(SPLIT_CONFIG[split]["gt_path"], "rt") as f:
        original = json.load(f)

    updated_episodes = []
    missing = 0
    for ep in original["episodes"]:
        ep_id = ep["episode_id"]
        meta_file = out_dir / f"episode_{ep_id:06d}.json"
        if not meta_file.exists():
            updated_episodes.append(ep)  # keep GT as fallback
            missing += 1
            continue
        meta = json.load(open(meta_file))
        gen = meta.get("generated_instruction", {})
        new_instr = gen.get("text", "") if isinstance(gen, dict) else str(gen)
        if new_instr:
            ep = dict(ep)
            ep["instruction"] = {"instruction_text": new_instr}
            ep["_complete_metadata"] = {
                "landmark_annotations": meta.get("landmark_annotations", {}),
                "path_analysis": {
                    "summary": meta.get("path_analysis", {}).get("summary", {}),
                    "motion_text": meta.get("path_analysis", {}).get("motion_text", ""),
                },
                "annotation_version": ANNOTATION_VERSION,
            }
        updated_episodes.append(ep)

    result = dict(original)
    result["episodes"] = updated_episodes
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dataset_out, "wt") as f:
        json.dump(result, f)
    print(f"Packed {len(updated_episodes)} episodes → {dataset_out}  (missing={missing})")


# ── Main batch runner ─────────────────────────────────────────────────────────
def run_batch(split: str, workers: int, resume: bool, limit: Optional[int] = None):
    cfg = SPLIT_CONFIG[split]
    out_dir: Path = cfg["out_dir"]
    frames_dir: Path = cfg["frames_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (PIPELINE_ROOT / "outputs/datasets").mkdir(parents=True, exist_ok=True)

    if not frames_dir.exists():
        print(f"[ERROR] Frames directory does not exist: {frames_dir}")
        print(f"  For '{split}', render frames first:")
        if split == "train":
            print(f"  bash gate1_renderer/render_train.sh")
        sys.exit(1)

    # Load GT dataset
    print(f"Loading {cfg['gt_path']} ...")
    with gzip.open(cfg["gt_path"], "rt") as f:
        gt_raw = json.load(f)
    gt_data = {e["episode_id"]: e for e in gt_raw["episodes"]}
    print(f"  {len(gt_data)} episodes in {split}")

    # Find which episodes to process
    ep_ids = sorted(gt_data.keys())
    if limit:
        ep_ids = ep_ids[:limit]

    if resume:
        done = {int(f.stem.replace("episode_", "")) for f in out_dir.glob("episode_*.json")}
        ep_ids = [e for e in ep_ids if e not in done]
        print(f"  Resuming: {len(done)} done, {len(ep_ids)} remaining")
    else:
        print(f"  Processing: {len(ep_ids)} episodes")

    if not ep_ids:
        print("  All done!")
        return

    # Verify vLLM
    from openai import OpenAI
    client_test = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    models = client_test.models.list()
    print(f"  vLLM OK: {models.data[0].id}")

    # Batch processing with progress tracking
    t_start = time.time()
    done_count = 0
    error_count = 0
    total = len(ep_ids)

    def process_one(ep_id: int):
        out_file = out_dir / f"episode_{ep_id:06d}.json"
        if out_file.exists():
            return ep_id, "skip", None
        gt_ep = gt_data[ep_id]
        try:
            meta = annotate_episode(ep_id, gt_ep, frames_dir)
            with open(out_file, "w") as f:
                json.dump(meta, f)
            return ep_id, "ok", meta["_processing_time_s"]
        except FileNotFoundError as e:
            return ep_id, "no_frames", str(e)
        except Exception as e:
            return ep_id, "error", str(e)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, ep_id): ep_id for ep_id in ep_ids}
        for future in as_completed(futures):
            ep_id, status, info = future.result()
            done_count += 1
            if status == "error":
                error_count += 1
                print(f"[WARN] ep{ep_id} error: {info}")
            elif status == "no_frames":
                error_count += 1

            # Progress every 50 episodes
            if done_count % 50 == 0 or done_count == total:
                elapsed = time.time() - t_start
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (total - done_count) / rate if rate > 0 else 0
                print(f"  [{done_count}/{total}] {rate:.1f} eps/s  ETA={eta/60:.1f}m  errors={error_count}")

    elapsed_total = time.time() - t_start
    completed = total - error_count
    print(f"\nDone: {completed}/{total} annotated in {elapsed_total/60:.1f} min ({error_count} errors)")

    # Assemble final dataset
    print(f"\nAssembling dataset → {cfg['dataset_out']}")
    assemble_dataset(split, gt_data, out_dir, cfg["dataset_out"])


def main():
    parser = argparse.ArgumentParser(description="Batch complete metadata annotator")
    parser.add_argument("--split", required=True, choices=["val_unseen", "val_seen", "train"],
                        help="Dataset split to annotate")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers (default: 8)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed episodes")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N episodes (for testing)")
    parser.add_argument("--assemble-only", action="store_true",
                        help="Skip annotation, just pack existing metadata into dataset")
    args = parser.parse_args()

    if args.assemble_only:
        cfg = SPLIT_CONFIG[args.split]
        with gzip.open(cfg["gt_path"], "rt") as f:
            gt_data = {e["episode_id"]: e for e in json.load(f)["episodes"]}
        assemble_dataset(args.split, gt_data, cfg["out_dir"], cfg["dataset_out"])
        return

    run_batch(args.split, args.workers, args.resume, args.limit)


if __name__ == "__main__":
    main()
