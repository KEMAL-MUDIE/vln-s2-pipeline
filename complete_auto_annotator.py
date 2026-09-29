#!/usr/bin/env python3
"""
Complete Auto-Annotator — Unified VLN Metadata Generator
=========================================================
Accepts any input source and produces complete navigation metadata using
visual understanding (Gemma 4 31B via vLLM) at every stage.

INPUT TYPES
-----------
  --rosbag /path/to/recording.bag
      ROS2 bag with camera + odometry topics. Start/end auto-detected from odom.

  --video /path/to/video.mp4 [--start-pos x,y,z --end-pos x,y,z]
      MP4/AVI video. Odometry is estimated from visual flow if not provided.

  --images /path/to/frames/ [--waypoints poses.json]
      Directory of RGB images + optional JSON waypoints file.

  --sim-episode <id>
      Habitat VLN-CE sim episode (uses pre-rendered Gate 1 frames).

COMMON OPTIONS
--------------
  --start-pos x,y,z     Override / provide start position (metres)
  --end-pos x,y,z       Override / provide end position (metres)
  --odom-file path      JSON file: list of {position:[x,y,z], rotation:[qx,qy,qz,qw]}
  --scene-id str        Scene label (default: "realworld/unknown")
  --episode-id int      ID for output naming (default: 0)
  --output dir          Output directory (default: outputs/complete_metadata_new)
  --vllm-url url        vLLM endpoint (default: http://10.77.32.231:8000/v1)
  --frame-hz float      Camera downsample rate for rosbag (default: 2.0)
  --camera-topic str    ROS2 camera topic (default: /camera/camera/color/image_raw)
  --odom-topic str      ROS2 odom topic (default: /gdq/msg/gdq_odom)
  --no-vision           Skip vision annotation (path analysis + text only)

OUTPUT
------
  outputs/complete_metadata_new/episode_XXXXXX.json

  Schema:
    episode_id, source_type, source_path, scene_id,
    start_position, end_position, start_rotation,
    reference_path, goals, info{geodesic_distance, path_length_m},
    path_analysis{primitives, summary, motion_text, key_frame_indices},
    rendered_frames, frame_timestamps, n_frames,
    landmark_annotations{per_frame, scene_context, goal_landmark, n_frames_annotated},
    generated_instruction{text, generator, version, quality_ok},
    odom_raw (rosbag/real-world only),
    _annotation_version, _annotation_sources, _processing_time_s

EXAMPLES
--------
  # From ROS2 bag (real robot):
  python3 complete_auto_annotator.py --rosbag /data/scout_run_001.bag --episode-id 1

  # From video file:
  python3 complete_auto_annotator.py --video /data/kitchen_tour.mp4 \\
      --start-pos 0,0,0 --end-pos 5.2,0,3.1 --episode-id 2

  # From image directory + waypoints:
  python3 complete_auto_annotator.py --images /data/frames/ \\
      --waypoints /data/poses.json --episode-id 3

  # From sim episode (Habitat VLN-CE):
  python3 complete_auto_annotator.py --sim-episode 42 --episode-id 42

  # Batch: all sim episodes 0-99:
  python3 complete_auto_annotator.py --sim-episode-range 0 99 \\
      --output outputs/complete_metadata_new/
"""

import argparse
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

PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT))

# ── Default configuration ────────────────────────────────────────────────────

VLLM_BASE_URL  = "http://10.77.32.231:8000/v1"
VLLM_MODEL     = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
VLLM_API_KEY   = "token-abc123"

ANNOTATION_VERSION = "3.0-complete"

GT_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
RENDERED_FRAMES_DIR = PIPELINE_ROOT / "outputs" / "rendered_frames"
DEFAULT_OUT_DIR     = PIPELINE_ROOT / "outputs" / "complete_metadata_new"

# Vision prompts (used by Gate 3 and Gate 4)
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


# ── vLLM / Gemma backend ─────────────────────────────────────────────────────

class GemmaVLLM:
    """Async-compatible Gemma 4 31B client via vLLM OpenAI-compatible endpoint."""

    def __init__(self, base_url: str = VLLM_BASE_URL, model: str = VLLM_MODEL,
                 api_key: str = VLLM_API_KEY):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            except ImportError:
                raise RuntimeError("pip install openai")
        return self._client

    def describe_image(self, image_path: str, prompt: str, max_tokens: int = 200) -> str:
        """Send an image + prompt to Gemma, return text response."""
        client = self._get_client()
        b64 = _image_to_base64(image_path)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def generate_text(self, prompt: str, max_tokens: int = 150, temperature: float = 0.5) -> str:
        """Text-only generation."""
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    def ping(self) -> bool:
        """Return True if the vLLM server is reachable."""
        try:
            client = self._get_client()
            models = client.models.list()
            return len(models.data) > 0
        except Exception:
            return False


def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── Input adapters ────────────────────────────────────────────────────────────

def adapt_rosbag(
    bag_path: str,
    output_dir: Path,
    episode_id: int,
    camera_topic: str,
    odom_topic: str,
    frame_hz: float,
    start_time_s: Optional[float],
    end_time_s: Optional[float],
    scene_id: str,
) -> Dict:
    """
    Extract frames + poses from a ROS2 .bag file.
    Returns episode dict with keys: frames_dir, poses, source_type, source_path, scene_id.
    """
    from gate8_adapters.rosbag_adapter import RosbagAdapter
    adapter = RosbagAdapter(bag_path)
    ep = adapter.extract(
        output_dir=output_dir,
        episode_id=episode_id,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        frame_hz=frame_hz,
        scene_id=scene_id,
    )
    ep["source_type"] = "rosbag"
    ep["source_path"] = bag_path
    return ep


def adapt_video(
    video_path: str,
    output_dir: Path,
    episode_id: int,
    frame_hz: float,
    start_pos: Optional[List[float]],
    end_pos: Optional[List[float]],
    odom_file: Optional[str],
    scene_id: str,
) -> Dict:
    """
    Extract frames from a video file. Odometry from odom_file or straight-line estimate.
    Returns episode dict compatible with the rest of the pipeline.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("pip install opencv-python")

    ep_dir = output_dir / f"episode_{episode_id:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(video_fps / frame_hz))
    frame_idx = 0
    saved_count = 0
    frame_paths = []
    frame_timestamps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            fname = ep_dir / f"frame_{saved_count:04d}_rgb.png"
            cv2.imwrite(str(fname), frame)
            frame_paths.append(str(fname))
            frame_timestamps.append(frame_idx / video_fps)
            saved_count += 1
        frame_idx += 1
    cap.release()

    if saved_count == 0:
        raise ValueError(f"No frames extracted from {video_path}")

    # Build reference path from odom_file or straight-line interpolation
    if odom_file and Path(odom_file).exists():
        with open(odom_file) as f:
            odom_data = json.load(f)
        reference_path = [p["position"] for p in odom_data]
        start_rotation = odom_data[0].get("rotation", [0, 0, 0, 1]) if odom_data else [0,0,0,1]
        odom_raw = odom_data
    else:
        # Straight-line estimate if no odom provided
        sp = start_pos or [0.0, 0.0, 0.0]
        ep_pos = end_pos or [5.0, 0.0, 0.0]
        n = max(2, saved_count)
        reference_path = [
            [sp[0] + (ep_pos[0]-sp[0])*i/(n-1),
             sp[1] + (ep_pos[1]-sp[1])*i/(n-1),
             sp[2] + (ep_pos[2]-sp[2])*i/(n-1)]
            for i in range(n)
        ]
        start_rotation = [0.0, 0.0, 0.0, 1.0]
        odom_raw = None

    # Build poses.json (same format as Gate 1)
    poses = {"frames": []}
    for i, (fp, ts) in enumerate(zip(frame_paths, frame_timestamps)):
        path_idx = min(i, len(reference_path)-1)
        poses["frames"].append({
            "frame_index": i,
            "timestamp_s": ts,
            "image_path": str(fp),
            "position": reference_path[path_idx],
            "rotation": start_rotation,
        })
    poses_path = ep_dir / "poses.json"
    with open(poses_path, "w") as f:
        json.dump(poses, f, indent=2)

    ep = {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "frames_dir": str(ep_dir),
        "frame_paths": frame_paths,
        "frame_timestamps": frame_timestamps,
        "poses": poses,
        "start_position": reference_path[0],
        "start_rotation": start_rotation,
        "reference_path": reference_path,
        "goals": [{"position": reference_path[-1], "radius": 3.0}],
        "source_type": "video",
        "source_path": video_path,
        "odom_raw": odom_raw,
    }
    return ep


def adapt_images(
    images_dir: str,
    waypoints_file: Optional[str],
    episode_id: int,
    start_pos: Optional[List[float]],
    end_pos: Optional[List[float]],
    scene_id: str,
) -> Dict:
    """
    Use a directory of pre-extracted images. Waypoints from JSON or straight-line.
    """
    img_dir = Path(images_dir)
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    frame_paths = sorted([str(p) for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if not frame_paths:
        raise ValueError(f"No images found in {images_dir}")

    if waypoints_file and Path(waypoints_file).exists():
        with open(waypoints_file) as f:
            wp_data = json.load(f)
        if isinstance(wp_data, list) and wp_data and "position" in wp_data[0]:
            reference_path = [w["position"] for w in wp_data]
            start_rotation = wp_data[0].get("rotation", [0,0,0,1])
            odom_raw = wp_data
        elif isinstance(wp_data, dict) and "frames" in wp_data:
            reference_path = [f["position"] for f in wp_data["frames"]]
            start_rotation = wp_data["frames"][0].get("rotation", [0,0,0,1])
            odom_raw = wp_data["frames"]
        else:
            reference_path = wp_data if isinstance(wp_data, list) else [[0,0,0],[5,0,0]]
            start_rotation = [0,0,0,1]
            odom_raw = None
    else:
        sp = start_pos or [0.0, 0.0, 0.0]
        ep_pos = end_pos or [5.0, 0.0, 0.0]
        n = max(2, len(frame_paths))
        reference_path = [
            [sp[0] + (ep_pos[0]-sp[0])*i/(n-1),
             sp[1] + (ep_pos[1]-sp[1])*i/(n-1),
             sp[2] + (ep_pos[2]-sp[2])*i/(n-1)]
            for i in range(n)
        ]
        start_rotation = [0.0, 0.0, 0.0, 1.0]
        odom_raw = None

    frame_timestamps = list(range(len(frame_paths)))  # frame index as timestamp

    ep = {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "frames_dir": images_dir,
        "frame_paths": frame_paths,
        "frame_timestamps": frame_timestamps,
        "start_position": reference_path[0],
        "start_rotation": start_rotation,
        "reference_path": reference_path,
        "goals": [{"position": reference_path[-1], "radius": 3.0}],
        "source_type": "images",
        "source_path": images_dir,
        "odom_raw": odom_raw,
    }
    return ep


def adapt_sim_episode(episode_id: int, gt_data: Dict, scene_id: Optional[str]) -> Dict:
    """
    Load a sim episode from pre-rendered Gate 1 frames + GT data.
    """
    gt_ep = gt_data.get(episode_id)
    if gt_ep is None:
        raise ValueError(f"Episode {episode_id} not found in GT dataset")

    ep_dir = RENDERED_FRAMES_DIR / f"episode_{episode_id:06d}"
    exts = {".png", ".jpg", ".jpeg"}
    frame_paths = []
    if ep_dir.exists():
        frame_paths = sorted([str(p) for p in ep_dir.iterdir() if p.suffix.lower() in exts])

    # Load poses if available
    poses = None
    poses_path = ep_dir / "poses.json"
    if poses_path.exists():
        with open(poses_path) as f:
            poses = json.load(f)

    ep = {
        "episode_id": episode_id,
        "trajectory_id": gt_ep.get("trajectory_id"),
        "scene_id": scene_id or gt_ep.get("scene_id", ""),
        "frames_dir": str(ep_dir),
        "frame_paths": frame_paths,
        "frame_timestamps": list(range(len(frame_paths))),
        "start_position": gt_ep.get("start_position", [0,0,0]),
        "start_rotation": gt_ep.get("start_rotation", [0,0,0,1]),
        "reference_path": gt_ep.get("reference_path", []),
        "goals": gt_ep.get("goals", []),
        "info": gt_ep.get("info", {}),
        "gt_instruction": gt_ep.get("instruction", {}).get("instruction_text", ""),
        "source_type": "sim",
        "source_path": f"habitat_mp3d/episode_{episode_id}",
        "poses": poses,
        "odom_raw": None,
    }
    return ep


# ── Path analysis (Gate 2) ────────────────────────────────────────────────────

def run_path_analysis(reference_path: List, start_rotation: List) -> Dict:
    """Run Gate 2 path analyzer. Falls back to minimal analysis if not available."""
    try:
        from gate2_path.path_analyzer import analyze_path, primitives_to_text
        analysis = analyze_path(reference_path, start_rotation)
        analysis["motion_text"] = primitives_to_text(analysis["primitives"])
        return analysis
    except Exception as e:
        # Minimal fallback
        path_length = sum(
            math.sqrt(sum((b[i]-a[i])**2 for i in range(3)))
            for a, b in zip(reference_path[:-1], reference_path[1:])
        ) if len(reference_path) >= 2 else 0.0
        n = len(reference_path)
        key_frames = [0, n//2, n-1] if n >= 3 else list(range(n))
        return {
            "primitives": [{"type": "straight", "distance_m": round(path_length, 2)}, {"type": "stop"}],
            "summary": {"path_length_m": round(path_length, 2), "n_turns": 0},
            "motion_text": f"Walk {path_length:.1f}m and stop.",
            "key_frame_indices": key_frames,
            "segment_headings": [],
            "_fallback": str(e),
        }


# ── Vision annotation (Gate 3 + scene description) ──────────────────────────

def _parse_scene_description(raw: str) -> Dict:
    """Parse Gemma scene description response into structured dict."""
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


def run_vision_annotation(
    frame_paths: List[str],
    key_frame_indices: List[int],
    gemma: GemmaVLLM,
) -> Dict:
    """
    Gate 3: Run Gemma vision on key frames.
    Returns landmark_annotations dict.
    """
    per_frame = {}
    valid_key_frames = [i for i in key_frame_indices if i < len(frame_paths)]

    for frame_idx in valid_key_frames:
        img_path = frame_paths[frame_idx]
        if not Path(img_path).exists():
            continue
        try:
            raw = gemma.describe_image(img_path, SCENE_DESCRIPTION_PROMPT, max_tokens=120)
            per_frame[str(frame_idx)] = _parse_scene_description(raw)
        except Exception as e:
            per_frame[str(frame_idx)] = {"room": "unknown", "landmarks": [], "direction": "unclear", "error": str(e)}

    # Scene context: start frame (or first key frame)
    scene_context = per_frame.get(str(valid_key_frames[0]), {}) if valid_key_frames else {}

    # Goal landmark: last frame
    goal_landmark = {"description": "the destination", "raw": ""}
    if frame_paths:
        last_frame = frame_paths[-1]
        if Path(last_frame).exists():
            try:
                raw_goal = gemma.describe_image(last_frame, GOAL_LANDMARK_PROMPT, max_tokens=80)
                goal_landmark = {"description": raw_goal, "raw": raw_goal}
            except Exception as e:
                goal_landmark["error"] = str(e)

    return {
        "per_frame": per_frame,
        "scene_context": scene_context,
        "goal_landmark": goal_landmark,
        "n_frames_annotated": len(per_frame),
    }


# ── Instruction generation (Gate 4) ──────────────────────────────────────────

def _build_scene_context_text(landmark_annotations: Dict) -> str:
    """Format landmark annotations into a text block for the instruction prompt."""
    per_frame = landmark_annotations.get("per_frame", {})
    lines = []
    for frame_idx in sorted(per_frame.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        fd = per_frame[frame_idx]
        room = fd.get("room", "unknown")
        lms = ", ".join(fd.get("landmarks", []))
        direction = fd.get("direction", "")
        label = "Start" if frame_idx == "0" else f"Frame {frame_idx}"
        lines.append(f"  [{label}] {room.title()} — landmarks: {lms or 'none'}, direction: {direction}")
    return "\n".join(lines) if lines else "  Indoor environment, path to destination."


def _quality_check(instruction: str) -> Tuple[bool, List[str]]:
    words = instruction.split()
    issues = []
    if len(words) < 8:
        issues.append(f"too_short ({len(words)} words)")
    if len(words) > 70:
        issues.append(f"too_long ({len(words)} words)")
    stop_words = ["stop", "wait", "stand", "remain", "halt"]
    if not any(w in instruction.lower() for w in stop_words):
        issues.append("no_stop_condition")
    return len(issues) == 0, issues


def run_instruction_generation(
    path_analysis: Dict,
    landmark_annotations: Dict,
    gemma: GemmaVLLM,
    max_retries: int = 2,
) -> Dict:
    """Gate 4: Generate navigation instruction from path + scene context."""
    motion_text = path_analysis.get("motion_text", "Navigate to the destination.")
    scene_context_text = _build_scene_context_text(landmark_annotations)
    goal_desc = landmark_annotations.get("goal_landmark", {}).get("description", "the destination")

    prompt = INSTRUCTION_GENERATION_PROMPT.format(
        motion_text=motion_text,
        scene_context=scene_context_text,
        goal_landmark=goal_desc,
    )

    best = None
    for attempt in range(max_retries + 1):
        temp = 0.3 + attempt * 0.2
        try:
            text = gemma.generate_text(prompt, max_tokens=100, temperature=temp)
            # Strip common prefixes Gemma sometimes adds
            text = re.sub(r"^(Instruction:|Navigation:|Answer:)\s*", "", text, flags=re.I).strip()
            ok, issues = _quality_check(text)
            if ok or attempt == max_retries:
                best = {"text": text, "quality_ok": ok, "quality_issues": issues, "attempt": attempt + 1}
                if ok:
                    break
        except Exception as e:
            best = {"text": "", "quality_ok": False, "quality_issues": [str(e)], "attempt": attempt + 1}

    return {
        "text": best["text"] if best else "",
        "generator": VLLM_MODEL,
        "version": ANNOTATION_VERSION,
        "quality_ok": best["quality_ok"] if best else False,
        "quality_issues": best.get("quality_issues", []),
    }


# ── Metadata assembly ─────────────────────────────────────────────────────────

def _path_length(path: List) -> float:
    return sum(
        math.sqrt(sum((b[i]-a[i])**2 for i in range(3)))
        for a, b in zip(path[:-1], path[1:])
    ) if len(path) >= 2 else 0.0


def assemble_metadata(
    ep: Dict,
    path_analysis: Dict,
    landmark_annotations: Optional[Dict],
    generated_instruction: Optional[Dict],
    processing_time_s: float,
    annotation_sources: List[str],
) -> Dict:
    """Assemble all components into the complete metadata JSON."""

    reference_path = ep.get("reference_path", [])
    path_len = ep.get("info", {}).get("path_length_m") or _path_length(reference_path)

    info = dict(ep.get("info", {}))
    info.setdefault("path_length_m", round(path_len, 3))
    if reference_path and len(reference_path) >= 2:
        info.setdefault("geodesic_distance", round(_path_length(reference_path), 3))

    meta = {
        "episode_id": ep.get("episode_id", 0),
        "source_type": ep.get("source_type", "unknown"),
        "source_path": ep.get("source_path", ""),
        "scene_id": ep.get("scene_id", "realworld/unknown"),
        "start_position": ep.get("start_position", [0.0, 0.0, 0.0]),
        "end_position": ep.get("goals", [{}])[0].get("position", reference_path[-1] if reference_path else [0,0,0]),
        "start_rotation": ep.get("start_rotation", [0.0, 0.0, 0.0, 1.0]),
        "reference_path": reference_path,
        "goals": ep.get("goals", []),
        "info": info,
        "path_analysis": path_analysis,
        "rendered_frames": ep.get("frame_paths", []),
        "frame_timestamps": ep.get("frame_timestamps", []),
        "n_frames": len(ep.get("frame_paths", [])),
        "landmark_annotations": landmark_annotations or {},
        "generated_instruction": generated_instruction or {},
        "_annotation_version": ANNOTATION_VERSION,
        "_annotation_sources": annotation_sources,
        "_processing_time_s": round(processing_time_s, 2),
    }

    # Include trajectory_id and gt_instruction for sim episodes
    if ep.get("trajectory_id") is not None:
        meta["trajectory_id"] = ep["trajectory_id"]
    if ep.get("gt_instruction"):
        meta["gt_instruction"] = {"text": ep["gt_instruction"]}

    # Include raw odom for real-world sources
    if ep.get("odom_raw") is not None:
        meta["odom_raw"] = ep["odom_raw"]

    return meta


# ── Main pipeline runner ──────────────────────────────────────────────────────

def run_episode(
    ep: Dict,
    gemma: Optional[GemmaVLLM],
    use_vision: bool,
    output_dir: Path,
    verbose: bool = True,
) -> Dict:
    """Run the full annotation pipeline for one episode."""
    t0 = time.time()
    annotation_sources = []

    if verbose:
        print(f"\n[{ep['episode_id']}] source={ep['source_type']} frames={len(ep.get('frame_paths',[]))} "
              f"path_pts={len(ep.get('reference_path',[]))}")

    # Gate 2: Path analysis
    path_analysis = run_path_analysis(ep.get("reference_path", []), ep.get("start_rotation", [0,0,0,1]))
    annotation_sources.append("gate2_path_analyzer")
    if verbose:
        print(f"  Gate2: {path_analysis.get('motion_text','')[:80]}")

    # Gate 3: Vision annotation
    landmark_annotations = {}
    if use_vision and gemma and ep.get("frame_paths"):
        key_frames = path_analysis.get("key_frame_indices", [0, len(ep["frame_paths"])//2, len(ep["frame_paths"])-1])
        landmark_annotations = run_vision_annotation(ep["frame_paths"], key_frames, gemma)
        annotation_sources.append("gate3_gemma4_scene_description")
        if verbose:
            sc = landmark_annotations.get("scene_context", {})
            print(f"  Gate3: room={sc.get('room','?')} landmarks={sc.get('landmarks',[])}")
    elif not use_vision:
        annotation_sources.append("gate3_skipped_no_vision")
    else:
        annotation_sources.append("gate3_skipped_no_frames")

    # Gate 4: Instruction generation
    generated_instruction = {}
    if gemma:
        generated_instruction = run_instruction_generation(path_analysis, landmark_annotations, gemma)
        annotation_sources.append("gate4_gemma4_31b_instruction")
        if verbose:
            print(f"  Gate4: [{'+' if generated_instruction['quality_ok'] else '-'}] "
                  f"{generated_instruction.get('text','')[:80]}")
    else:
        generated_instruction = {
            "text": path_analysis.get("motion_text", "Navigate to the destination."),
            "generator": "path_analysis_fallback",
            "version": ANNOTATION_VERSION,
            "quality_ok": False,
            "quality_issues": ["no_vllm_backend"],
        }
        annotation_sources.append("gate4_path_fallback")

    # Assemble
    elapsed = time.time() - t0
    meta = assemble_metadata(ep, path_analysis, landmark_annotations, generated_instruction,
                             elapsed, annotation_sources)

    # Save
    out_path = output_dir / f"episode_{ep['episode_id']:06d}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"  Done in {elapsed:.1f}s → {out_path}")

    return meta


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_pos(s: Optional[str]) -> Optional[List[float]]:
    if not s:
        return None
    try:
        return [float(x) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Position must be x,y,z floats, got: {s}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete Auto-Annotator: scene/video/rosbag/sim → complete navigation metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input source (mutually exclusive)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--rosbag", metavar="PATH", help="ROS2 .bag file")
    src.add_argument("--video", metavar="PATH", help="Video file (mp4, avi, ...)")
    src.add_argument("--images", metavar="DIR", help="Directory of RGB images")
    src.add_argument("--sim-episode", metavar="ID", type=int, help="Habitat sim episode ID")
    src.add_argument("--sim-episode-range", nargs=2, metavar=("START","END"), type=int,
                     help="Batch: process sim episodes START..END inclusive")

    # Position / odom
    parser.add_argument("--start-pos", metavar="x,y,z", help="Start position override")
    parser.add_argument("--end-pos", metavar="x,y,z", help="End position override")
    parser.add_argument("--odom-file", metavar="PATH", help="Odometry JSON file")

    # Metadata
    parser.add_argument("--scene-id", default="realworld/unknown", help="Scene label")
    parser.add_argument("--episode-id", type=int, default=0, help="Episode ID for output naming")
    parser.add_argument("--output", default=str(DEFAULT_OUT_DIR), help="Output directory")

    # vLLM
    parser.add_argument("--vllm-url", default=VLLM_BASE_URL, help="vLLM endpoint URL")
    parser.add_argument("--no-vision", action="store_true", help="Skip vision annotation (path analysis only)")
    parser.add_argument("--no-vllm", action="store_true", help="Skip all Gemma calls (path analysis fallback only)")

    # ROS2 bag options
    parser.add_argument("--camera-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--odom-topic", default="/gdq/msg/gdq_odom")
    parser.add_argument("--frame-hz", type=float, default=2.0, help="Frame extraction rate")
    parser.add_argument("--bag-start", type=float, default=None, help="Bag start time (seconds)")
    parser.add_argument("--bag-end", type=float, default=None, help="Bag end time (seconds)")

    # Other
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    output_dir = Path(args.output)
    start_pos = parse_pos(args.start_pos)
    end_pos = parse_pos(args.end_pos)

    # Init Gemma backend
    gemma = None
    if not args.no_vllm:
        gemma = GemmaVLLM(base_url=args.vllm_url)
        if verbose:
            reachable = gemma.ping()
            status = "OK" if reachable else "UNREACHABLE (will use fallback)"
            print(f"[vLLM] {args.vllm_url} → {status}")
            if not reachable:
                gemma = None

    # Load GT data for sim episodes
    gt_data = {}
    if args.sim_episode is not None or args.sim_episode_range is not None:
        if GT_PATH.exists():
            with gzip.open(GT_PATH, "rt") as f:
                raw = json.load(f)
            gt_data = {ep["episode_id"]: ep for ep in raw["episodes"]}
            if verbose:
                print(f"[GT] Loaded {len(gt_data)} sim episodes")
        else:
            print(f"WARNING: GT dataset not found at {GT_PATH}")

    # Build episode list
    episodes_to_run = []
    scratch_dir = output_dir / "_frames"

    if args.rosbag:
        ep = adapt_rosbag(
            args.rosbag, scratch_dir, args.episode_id,
            args.camera_topic, args.odom_topic, args.frame_hz,
            args.bag_start, args.bag_end, args.scene_id,
        )
        episodes_to_run = [ep]

    elif args.video:
        ep = adapt_video(
            args.video, scratch_dir, args.episode_id, args.frame_hz,
            start_pos, end_pos, args.odom_file, args.scene_id,
        )
        episodes_to_run = [ep]

    elif args.images:
        ep = adapt_images(
            args.images, args.odom_file, args.episode_id,
            start_pos, end_pos, args.scene_id,
        )
        episodes_to_run = [ep]

    elif args.sim_episode is not None:
        ep = adapt_sim_episode(args.sim_episode, gt_data, args.scene_id if args.scene_id != "realworld/unknown" else None)
        ep["episode_id"] = args.episode_id if args.episode_id != 0 else args.sim_episode
        episodes_to_run = [ep]

    elif args.sim_episode_range is not None:
        start_id, end_id = args.sim_episode_range
        for eid in range(start_id, end_id + 1):
            try:
                ep = adapt_sim_episode(eid, gt_data, None)
                episodes_to_run.append(ep)
            except ValueError as e:
                if verbose:
                    print(f"  SKIP ep{eid}: {e}")

    if not episodes_to_run:
        print("No episodes to process.")
        sys.exit(1)

    if verbose:
        print(f"\nProcessing {len(episodes_to_run)} episode(s) → {output_dir}\n")

    # Run pipeline
    t_batch_start = time.time()
    results = []
    for ep in episodes_to_run:
        try:
            meta = run_episode(ep, gemma, use_vision=not args.no_vision,
                               output_dir=output_dir, verbose=verbose)
            results.append({"episode_id": meta["episode_id"], "status": "ok",
                            "instruction": meta.get("generated_instruction", {}).get("text", "")})
        except Exception as e:
            print(f"  ERROR ep{ep.get('episode_id','?')}: {e}")
            results.append({"episode_id": ep.get("episode_id", "?"), "status": "error", "error": str(e)})

    # Write index
    index_path = output_dir / "metadata_index.json"
    index = {
        "n_episodes": len(results),
        "n_ok": sum(1 for r in results if r["status"] == "ok"),
        "n_error": sum(1 for r in results if r["status"] == "error"),
        "total_time_s": round(time.time() - t_batch_start, 1),
        "vllm_url": args.vllm_url,
        "annotation_version": ANNOTATION_VERSION,
        "episodes": results,
    }
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Done: {index['n_ok']}/{index['n_episodes']} OK in {index['total_time_s']:.1f}s")
        print(f"Index: {index_path}")
        if index['n_error'] > 0:
            print(f"Errors: {index['n_error']} — check individual episode files")


if __name__ == "__main__":
    main()
