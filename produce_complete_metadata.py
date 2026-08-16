#!/usr/bin/env python3
"""
Metadata Producer — assembles all S2 pipeline outputs into complete per-episode metadata.

For each of the 1839 val-unseen VLN-CE episodes, merges:
  - GT structural data (reference_path, start_position, goals, scene_id, geodesic_distance)
  - Rendered frames (Gate 1: image_paths + waypoint poses)
  - Path analysis (Gate 2: motion primitives, turn sequence, distance breakdown)
  - Landmark annotations (Gate 3: scene_context + goal_landmark from Gemma vision)
  - Generated instruction (Gate 4 v2: Gemma 4 31B natural language instruction)
  - GT instruction (for quality comparison in compare_metadata_quality.py)

Output: outputs/complete_metadata/episode_XXXXXX.json (1839 files)
        outputs/complete_metadata/metadata_index.json (summary across all episodes)

Usage:
  python3 produce_complete_metadata.py                 # all 1839 episodes
  python3 produce_complete_metadata.py --episodes 1 2 3   # specific IDs
  python3 produce_complete_metadata.py --overwrite        # re-run all
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from gate2_path.path_analyzer import analyze_path, primitives_to_text

PIPELINE_ROOT = Path(__file__).parent
GT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
RENDERED_FRAMES_DIR = PIPELINE_ROOT / "outputs" / "rendered_frames"
LANDMARKS_ALL_PATH = PIPELINE_ROOT / "outputs" / "gate3_landmarks" / "all_landmarks.json"
GATE4_CHECKPOINT = PIPELINE_ROOT / "outputs" / "gate4_visual_v2_checkpoint.json"
OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "complete_metadata"
ANNOTATION_VERSION = "1.0"


def load_gt_episodes() -> Dict[int, Dict]:
    with gzip.open(GT_PATH, "rt") as f:
        data = json.load(f)
    return {ep["episode_id"]: ep for ep in data["episodes"]}


def load_landmarks_by_episode() -> Dict[int, Dict]:
    with open(LANDMARKS_ALL_PATH) as f:
        data = json.load(f)
    return {r["episode_id"]: r for r in data["results"]}


def load_gate4_instructions() -> Dict[int, str]:
    with open(GATE4_CHECKPOINT) as f:
        raw = json.load(f)
    # Keys are stored as strings (JSON limitation)
    return {int(k): v for k, v in raw.items()}


def load_poses(episode_id: int) -> Optional[Dict]:
    ep_dir = RENDERED_FRAMES_DIR / f"episode_{episode_id:06d}"
    poses_path = ep_dir / "poses.json"
    if not poses_path.exists():
        return None
    with open(poses_path) as f:
        return json.load(f)


def produce_episode_metadata(
    episode_id: int,
    gt_ep: Dict,
    landmarks: Optional[Dict],
    generated_instruction: Optional[str],
    poses: Optional[Dict],
) -> Dict[str, Any]:
    reference_path = gt_ep.get("reference_path", [])
    start_rotation = gt_ep.get("start_rotation", None)
    path_analysis = analyze_path(reference_path, start_rotation)
    motion_text = primitives_to_text(path_analysis["primitives"])

    # Rendered frame info with absolute image paths
    frames_info: List[Dict] = []
    if poses:
        ep_dir = RENDERED_FRAMES_DIR / f"episode_{episode_id:06d}"
        for frame in poses.get("frames", []):
            frames_info.append({
                "frame_idx": frame["frame_idx"],
                "waypoint_idx": frame["waypoint_idx"],
                "label": frame["label"],
                "position": frame["position"],
                "rotation": frame["rotation"],
                "image_path": str(ep_dir / frame["path"]),
                "image_filename": frame["path"],
            })

    # GT instruction
    gt_instr = gt_ep.get("instruction", {})
    if isinstance(gt_instr, dict):
        gt_text = gt_instr.get("instruction_text", "")
        gt_tokens = gt_instr.get("instruction_tokens", [])
    else:
        gt_text = str(gt_instr)
        gt_tokens = []

    # Landmark annotations
    lm_scene_context = landmarks.get("scene_context", {}) if landmarks else {}
    lm_goal_landmark = landmarks.get("goal_landmark", {}) if landmarks else {}

    return {
        # ── Structural (from GT — scene + path geometry) ─────────────────────
        "episode_id": episode_id,
        "trajectory_id": gt_ep.get("trajectory_id"),
        "scene_id": gt_ep.get("scene_id"),
        "start_position": gt_ep.get("start_position"),
        "start_rotation": gt_ep.get("start_rotation"),
        "reference_path": reference_path,
        "goals": gt_ep.get("goals", []),
        "info": gt_ep.get("info", {}),
        # ── Path analysis (Gate 2 — from reference_path geometry) ────────────
        "path_analysis": {
            "primitives": path_analysis["primitives"],
            "summary": path_analysis["summary"],
            "motion_text": motion_text,
            "key_frame_indices": path_analysis["key_frame_indices"],
            "segment_headings": path_analysis["segment_headings"],
        },
        # ── Rendered frames (Gate 1 — rendered RGB from Habitat) ─────────────
        "rendered_frames": frames_info,
        "n_frames": len(frames_info),
        # ── Landmark annotations (Gate 3 — Gemma vision on start+goal frames) ─
        "landmark_annotations": {
            "scene_context": lm_scene_context,
            "goal_landmark": lm_goal_landmark,
            "n_frames_annotated": landmarks.get("n_frames", 0) if landmarks else 0,
        },
        # ── Generated instruction (Gate 4 v2 — Gemma 4 31B via vLLM) ─────────
        "generated_instruction": {
            "text": generated_instruction or "",
            "generator": "gemma-4-31b-awq",
            "version": "v2",
        },
        # ── GT instruction (for quality comparison) ───────────────────────────
        "gt_instruction": {
            "text": gt_text,
            "tokens": gt_tokens,
        },
        # ── Provenance ────────────────────────────────────────────────────────
        "_annotation_version": ANNOTATION_VERSION,
        "_annotation_sources": [
            "gate1_renderer",
            "gate2_path_analyzer",
            "gate3_landmarks_gemma4",
            "gate4_visual_v2_gemma4_31b",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce complete per-episode metadata for VLN-CE val-unseen")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Episode IDs to process (default: all 1839)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-produce even if output file already exists")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ChronoNav Metadata Producer")
    print("=" * 60)
    print(f"Output: {args.output_dir}")

    print("\nLoading GT episodes...", end=" ", flush=True)
    gt_episodes = load_gt_episodes()
    print(f"{len(gt_episodes)} episodes")

    print("Loading landmark annotations...", end=" ", flush=True)
    landmarks_by_ep = load_landmarks_by_episode()
    print(f"{len(landmarks_by_ep)} episodes")

    print("Loading Gate 4 v2 instructions...", end=" ", flush=True)
    gate4_instrs = load_gate4_instructions()
    print(f"{len(gate4_instrs)} instructions")

    episode_ids = args.episodes if args.episodes else sorted(gt_episodes.keys())
    total = len(episode_ids)
    print(f"\nProducing metadata for {total} episodes...\n")

    stats = {
        "produced": 0, "skipped": 0,
        "missing_frames": 0, "missing_landmarks": 0, "missing_instruction": 0,
    }
    index: List[Dict] = []
    t0 = time.time()

    for i, ep_id in enumerate(episode_ids):
        out_path = args.output_dir / f"episode_{ep_id:06d}.json"

        if out_path.exists() and not args.overwrite:
            stats["skipped"] += 1
            with open(out_path) as f:
                meta = json.load(f)
        else:
            gt_ep = gt_episodes.get(ep_id)
            if not gt_ep:
                continue

            poses = load_poses(ep_id)
            ep_landmarks = landmarks_by_ep.get(ep_id)
            gen_instr = gate4_instrs.get(ep_id)

            if not poses:
                stats["missing_frames"] += 1
            if not ep_landmarks:
                stats["missing_landmarks"] += 1
            if not gen_instr:
                stats["missing_instruction"] += 1

            meta = produce_episode_metadata(ep_id, gt_ep, ep_landmarks, gen_instr, poses)

            with open(out_path, "w") as f:
                json.dump(meta, f, indent=2)
            stats["produced"] += 1

        # Build index entry (lightweight — no large arrays)
        lm = meta.get("landmark_annotations", {})
        index.append({
            "episode_id": ep_id,
            "scene_id": meta.get("scene_id", ""),
            "geodesic_distance_m": meta.get("info", {}).get("geodesic_distance"),
            "n_frames": meta.get("n_frames", 0),
            "path_type": meta.get("path_analysis", {}).get("summary", {}).get("path_type"),
            "n_waypoints": meta.get("path_analysis", {}).get("summary", {}).get("n_waypoints"),
            "has_landmarks": bool(lm.get("scene_context")),
            "has_generated_instruction": bool(meta.get("generated_instruction", {}).get("text")),
            "generated_instruction_length": len(meta.get("generated_instruction", {}).get("text", "")),
            "gt_instruction_length": len(meta.get("gt_instruction", {}).get("text", "")),
        })

        if (i + 1) % 200 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            pct = (i + 1) / total * 100
            print(f"  [{i+1:4d}/{total}] {pct:.1f}% | produced={stats['produced']} "
                  f"skipped={stats['skipped']} "
                  f"miss_frames={stats['missing_frames']} "
                  f"miss_lm={stats['missing_landmarks']} "
                  f"miss_instr={stats['missing_instruction']} "
                  f"| {rate:.1f} eps/s ETA {eta:.0f}s")

    # Save master index
    index_path = args.output_dir / "metadata_index.json"
    with open(index_path, "w") as f:
        json.dump({
            "n_episodes": len(index),
            "annotation_version": ANNOTATION_VERSION,
            "sources": [
                "gate1_renderer",
                "gate2_path_analyzer",
                "gate3_landmarks_gemma4",
                "gate4_visual_v2_gemma4_31b",
            ],
            "episodes": index,
        }, f, indent=2)

    elapsed = time.time() - t0
    has_instr = sum(1 for e in index if e["has_generated_instruction"])
    has_lm = sum(1 for e in index if e["has_landmarks"])
    has_fr = sum(1 for e in index if e["n_frames"] > 0)

    print(f"\n{'=' * 60}")
    print(f"Complete — {elapsed:.1f}s total")
    print(f"  Produced:  {stats['produced']} new, {stats['skipped']} already existed")
    print(f"  Coverage:  frames={has_fr}/{len(index)} "
          f"landmarks={has_lm}/{len(index)} "
          f"instruction={has_instr}/{len(index)}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Index:     {index_path}")
    print(f"\nNext step: python3 compare_metadata_quality.py")


if __name__ == "__main__":
    main()
