#!/usr/bin/env python3
"""
Config-driven batch annotator.
Reads a version-specific YAML config from configs/annotator_v{N}.yaml
and runs the Gate1-4 annotation pipeline for one split.

Usage:
  python3 run_annotator_configured.py --config configs/annotator_v278.yaml --split val_seen
  python3 run_annotator_configured.py --config configs/annotator_v272.yaml --split train --resume
  python3 run_annotator_configured.py --config configs/annotator_v278.yaml --split val_unseen --resume

Splits:
  val_unseen  — 1839 episodes (4 eval-cap scenes)
  val_seen    — 778 episodes
  train       — 10819 episodes (requires rendered frames in outputs/rendered_frames_train/)

Output structure (from config output.base_dir):
  outputs/annotated_datasets_top5/
    seen_v278.json.gz          ← final packed dataset
    seen_v278_meta/            ← per-episode metadata JSON files
    unseen_v278.json.gz
    unseen_v278_meta/
    train_v278.json.gz         (pending GPU2)
    train_v278_meta/
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

import yaml

PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT))


def load_config(config_path: str) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_openai_client(cfg: Dict):
    from openai import OpenAI
    return OpenAI(
        base_url=cfg["vllm"]["base_url"],
        api_key=cfg["vllm"]["api_key"],
    )


def _encode_image(path: str) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _describe_image(client, img_path: str, prompt: str, max_tokens: int, temperature: float) -> str:
    ext = Path(img_path).suffix.lower()
    mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
    b64 = _encode_image(img_path)
    resp = client.chat.completions.create(
        model=client._client.base_url.__str__().split("//")[1].split(":")[0],  # not used
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def _describe_image_vllm(client, img_path: str, prompt: str, cfg: Dict, stage: str) -> str:
    """Call vLLM with image. Uses model from config."""
    import base64
    ext = Path(img_path).suffix.lower()
    mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
    with open(img_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("utf-8")
    stage_cfg = cfg.get(stage, cfg.get("gate3", {}))
    resp = client.chat.completions.create(
        model=cfg["vllm"]["model"],
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=stage_cfg.get("max_tokens", 120),
        temperature=stage_cfg.get("temperature", 0.4),
    )
    return resp.choices[0].message.content.strip()


def _generate_text(client, prompt: str, cfg: Dict, stage: str = "gate4",
                   temperature_override: Optional[float] = None) -> str:
    stage_cfg = cfg.get(stage, cfg.get("gate4", {}))
    temp = temperature_override if temperature_override is not None else stage_cfg.get("temperature", 0.4)
    resp = client.chat.completions.create(
        model=cfg["vllm"]["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=stage_cfg.get("max_tokens", 120),
        temperature=temp,
    )
    return resp.choices[0].message.content.strip()


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
Write ONLY the landmark description:
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

Write ONLY the instruction text:
"""


def _path_analysis(reference_path: List, start_rotation: List) -> Dict:
    try:
        from gate2_path.path_analyzer import analyze_path, primitives_to_text
        analysis = analyze_path(reference_path, start_rotation)
        analysis["motion_text"] = primitives_to_text(analysis["primitives"])
        return analysis
    except Exception:
        path_length = sum(
            math.sqrt(sum((b[i]-a[i])**2 for i in range(3)))
            for a, b in zip(reference_path[:-1], reference_path[1:])
        ) if len(reference_path) >= 2 else 0.0
        n = len(reference_path)
        key_frames = [0, n//2, n-1] if n >= 3 else list(range(n))
        return {
            "primitives": [{"type": "straight", "distance_m": round(path_length, 2)}, {"type": "stop"}],
            "summary": {"path_length_m": round(path_length, 2), "n_turns": 0},
            "motion_text": f"Walk forward and stop.",
            "key_frame_indices": key_frames,
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


def _quality_check(text: str, cfg: Dict) -> List[str]:
    """Returns list of quality issues (empty = OK)."""
    q = cfg.get("quality", {})
    words = text.split()
    issues = []
    if len(words) < q.get("min_words", 8):
        issues.append(f"too_short({len(words)}w)")
    if len(words) > q.get("max_words", 80):
        issues.append(f"too_long({len(words)}w)")
    if q.get("require_stop_condition", True):
        if not any(w in text.lower() for w in ["stop", "wait", "stand", "remain", "halt"]):
            issues.append("no_stop")
    if q.get("require_landmark", False):
        common_nouns = {"table", "chair", "couch", "sofa", "desk", "bed", "stairs", "staircase",
                        "doorway", "door", "window", "kitchen", "bedroom", "hallway", "living",
                        "dining", "bathroom", "cabinet", "shelf", "rug", "counter", "fireplace"}
        if not any(n in text.lower() for n in common_nouns):
            issues.append("no_landmark")
    return issues


def annotate_episode(ep_id: int, gt_ep: Dict, frames_dir: Path, cfg: Dict) -> Dict:
    t0 = time.time()
    from openai import OpenAI
    client = OpenAI(base_url=cfg["vllm"]["base_url"], api_key=cfg["vllm"]["api_key"])

    ep_dir = frames_dir / f"episode_{ep_id:06d}"
    frame_paths = []
    if ep_dir.exists():
        frame_paths = sorted([
            str(p) for p in ep_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])
    if not frame_paths:
        raise FileNotFoundError(f"No frames for ep{ep_id} at {ep_dir}")

    poses = None
    poses_path = ep_dir / "poses.json"
    if poses_path.exists():
        with open(poses_path) as f:
            poses = json.load(f)

    reference_path = gt_ep.get("reference_path", [])
    start_rotation = gt_ep.get("start_rotation", [0, 0, 0, 1])
    gt_instruction = gt_ep.get("instruction", {}).get("instruction_text", "")

    path_analysis = _path_analysis(reference_path, start_rotation)
    key_frame_indices = path_analysis.get("key_frame_indices", list(range(len(frame_paths))))
    valid_key_frames = [i for i in key_frame_indices if i < len(frame_paths)]

    per_frame = {}
    gate3_cfg = cfg.get("gate3", {})
    for fidx in valid_key_frames:
        img_path = frame_paths[fidx]
        if not Path(img_path).exists():
            continue
        try:
            import base64
            ext = Path(img_path).suffix.lower()
            mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
            with open(img_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("utf-8")
            resp = client.chat.completions.create(
                model=cfg["vllm"]["model"],
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": SCENE_DESCRIPTION_PROMPT},
                ]}],
                max_tokens=gate3_cfg.get("max_tokens", 120),
                temperature=gate3_cfg.get("temperature", 0.4),
            )
            raw = resp.choices[0].message.content.strip()
            per_frame[str(fidx)] = _parse_scene_description(raw)
        except Exception as e:
            per_frame[str(fidx)] = {"room": "unknown", "landmarks": [], "direction": "unclear", "error": str(e)}

    goal_landmark = {"description": "the destination", "raw": ""}
    if frame_paths and Path(frame_paths[-1]).exists():
        try:
            import base64
            ext = Path(frame_paths[-1]).suffix.lower()
            mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
            with open(frame_paths[-1], "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("utf-8")
            resp = client.chat.completions.create(
                model=cfg["vllm"]["model"],
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": GOAL_LANDMARK_PROMPT},
                ]}],
                max_tokens=80,
                temperature=gate3_cfg.get("temperature", 0.4),
            )
            raw_goal = resp.choices[0].message.content.strip()
            goal_landmark = {"description": raw_goal, "raw": raw_goal}
        except Exception as e:
            goal_landmark["error"] = str(e)

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

    gate4_cfg = cfg.get("gate4", {})
    base_temp = gate4_cfg.get("temperature", 0.4)
    max_retries = gate4_cfg.get("max_retries", 3)

    generated_text = ""
    quality_ok = False
    issues = []
    for attempt in range(max_retries):
        try:
            temp = base_temp + attempt * 0.1
            resp = client.chat.completions.create(
                model=cfg["vllm"]["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=gate4_cfg.get("max_tokens", 120),
                temperature=temp,
            )
            generated_text = resp.choices[0].message.content.strip()
            issues = _quality_check(generated_text, cfg)
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
        "info": {"geodesic_distance": gt_ep.get("info", {}).get("geodesic_distance", 0.0),
                 "path_length_m": round(path_len, 3)},
        "path_analysis": path_analysis,
        "rendered_frames": frame_paths,
        "n_frames": len(frame_paths),
        "landmark_annotations": {
            "per_frame": per_frame,
            "goal_landmark": goal_landmark,
            "n_frames_annotated": len(per_frame),
        },
        "generated_instruction": {
            "text": generated_text,
            "generator": cfg["vllm"]["model"],
            "version": cfg.get("version", "unknown"),
            "quality_ok": quality_ok,
            "quality_issues": issues,
        },
        "gt_instruction": gt_instruction,
        "_annotation_version": cfg.get("version", "unknown"),
        "_annotation_sources": ["gate1_renderer", "gate2_path", "gate3_gemma4_vision", "gate4_instruction"],
        "_processing_time_s": round(elapsed, 2),
        "_poses": poses,
    }


def assemble_dataset(split: str, gt_data: Dict, out_dir: Path, dataset_out: Path,
                     source_dataset: Optional[str] = None):
    """Assemble final .json.gz from per-episode metadata."""
    meta_files = sorted(out_dir.glob("ep_*.json"))
    print(f"Assembling {len(meta_files)} episodes → {dataset_out}")

    # Load source dataset (ChronoNav) or GT as instruction source
    if source_dataset and Path(source_dataset).exists():
        with gzip.open(source_dataset, "rt") as f:
            src = json.load(f)
        src_eps = {ep["episode_id"]: ep for ep in src["episodes"]}
    else:
        src_eps = {}

    episodes = []
    for mf in meta_files:
        with open(mf) as f:
            meta = json.load(f)
        ep_id = meta["episode_id"]

        # Use ChronoNav instruction if available, else generated
        if ep_id in src_eps:
            instr_text = src_eps[ep_id].get("instruction", {}).get("instruction_text",
                         meta["generated_instruction"]["text"])
        else:
            instr_text = meta["generated_instruction"]["text"]

        episodes.append({
            "episode_id": ep_id,
            "trajectory_id": meta.get("trajectory_id"),
            "scene_id": meta["scene_id"],
            "start_position": meta["start_position"],
            "start_rotation": meta["start_rotation"],
            "goals": meta["goals"],
            "reference_path": meta["reference_path"],
            "info": meta.get("info", {}),
            "instruction": {
                "instruction_text": instr_text,
                "instruction_tokens": [],
            },
            "_annotation_version": meta.get("_annotation_version"),
            "_generated_instruction": meta["generated_instruction"],
            "_landmark_annotations": meta.get("landmark_annotations"),
        })

    # Preserve vocab from GT
    output = {
        "episodes": episodes,
        "instruction_vocab": gt_data.get("instruction_vocab", {}),
    }
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dataset_out, "wt") as f:
        json.dump(output, f)
    print(f"  Wrote {len(episodes)} episodes → {dataset_out}")


def run_batch(cfg: Dict, split: str, workers: int, resume: bool, limit: Optional[int] = None):
    split_cfg = cfg.get("output", {}).get(split, {})
    if not split_cfg:
        # Map split names
        key = {"val_seen": "val_seen", "val_unseen": "val_unseen", "train": "train"}.get(split, split)
        split_cfg = cfg.get("output", {}).get(key, {})

    gt_path = Path(split_cfg.get("source_dataset", ""))
    frames_dir = Path(PIPELINE_ROOT / split_cfg.get("frames_dir", ""))
    out_dir = Path(PIPELINE_ROOT / split_cfg.get("metadata_dir", f"outputs/annotated_datasets_top5/{split}_meta"))
    dataset_out = Path(PIPELINE_ROOT / split_cfg.get("dataset_out", f"outputs/annotated_datasets_top5/{split}.json.gz"))

    if not gt_path.exists():
        print(f"ERROR: source dataset not found: {gt_path}")
        sys.exit(1)
    if not frames_dir.exists():
        print(f"ERROR: frames dir not found: {frames_dir}")
        print("       Run gate1_renderer/render_train.sh first for train split.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(gt_path, "rt") as f:
        gt_data = json.load(f)
    episodes = gt_data["episodes"]
    if limit:
        episodes = episodes[:limit]

    if resume:
        done = {int(p.stem.split("_")[1]) for p in out_dir.glob("ep_*.json")}
        episodes = [e for e in episodes if e["episode_id"] not in done]
        print(f"Resuming: {len(done)} done, {len(episodes)} remaining")

    total = len(episodes)
    print(f"Annotating {total} episodes for {split} with config version={cfg.get('version')}")
    print(f"  Frames: {frames_dir}  Out: {out_dir}")

    completed = 0
    failed = 0

    def process_one(ep):
        ep_id = ep["episode_id"]
        out_file = out_dir / f"ep_{ep_id:06d}.json"
        try:
            meta = annotate_episode(ep_id, ep, frames_dir, cfg)
            with open(out_file, "w") as f:
                json.dump(meta, f)
            return ep_id, None
        except Exception as e:
            return ep_id, str(e)

    w = min(workers, cfg.get("workers", 8))
    with ThreadPoolExecutor(max_workers=w) as pool:
        futures = {pool.submit(process_one, ep): ep["episode_id"] for ep in episodes}
        for fut in as_completed(futures):
            ep_id, err = fut.result()
            completed += 1
            if err:
                failed += 1
                print(f"  [{completed}/{total}] ep{ep_id} FAIL: {err}")
            elif completed % 50 == 0:
                print(f"  [{completed}/{total}] ok={completed-failed} fail={failed}")

    print(f"Done: {completed-failed}/{total} ok, {failed} failed")
    assemble_dataset(split, gt_data, out_dir, dataset_out, str(gt_path))


def main():
    parser = argparse.ArgumentParser(description="Config-driven batch annotator")
    parser.add_argument("--config", required=True, help="Path to annotator YAML config (e.g. configs/annotator_v278.yaml)")
    parser.add_argument("--split", required=True, choices=["val_unseen", "val_seen", "train"])
    parser.add_argument("--workers", type=int, default=None, help="Override workers from config")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    workers = args.workers or cfg.get("workers", 8)
    run_batch(cfg, args.split, workers, args.resume, args.limit)


if __name__ == "__main__":
    main()
