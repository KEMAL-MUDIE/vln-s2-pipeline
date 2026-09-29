#!/usr/bin/env python3
"""
Gate 8: Video Adapter
Converts a video file (mp4, avi, mkv) into the same frames + poses format
produced by Gate 1 (Habitat renderer) and Gate 8 (rosbag adapter).

After this adapter, data flows through Gates 2-6 unchanged.

Odometry sources (in priority order):
  1. --odom-file: JSON list of {position:[x,y,z], rotation:[qx,qy,qz,qw], timestamp_s:float}
  2. --start-pos / --end-pos: straight-line interpolation along path
  3. Default: 0,0,0 → 5,0,0 (fallback for unknown scenes)

Output format (same as Gate 1 and Gate 8 rosbag):
  frames/episode_{id:06d}/frame_{i:04d}_rgb.png
  frames/episode_{id:06d}/poses.json
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional


class VideoAdapter:
    """
    Extracts frames from a video file at a configurable rate.

    Usage:
        adapter = VideoAdapter("/path/to/video.mp4")
        episode = adapter.extract(
            output_dir=Path("outputs/rendered_frames"),
            episode_id=0,
            frame_hz=2.0,
            start_pos=[0, 0, 0],     # optional
            end_pos=[5, 0, 3],       # optional
            odom_file="poses.json",  # optional
            scene_id="realworld/unknown",
        )
        # episode now has same format as Gate 1 / Gate 8 output
    """

    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        self._check_cv2()

    @staticmethod
    def _check_cv2():
        try:
            import cv2
        except ImportError:
            raise RuntimeError("pip install opencv-python")

    def extract(
        self,
        output_dir: Path,
        episode_id: int = 0,
        frame_hz: float = 2.0,
        start_pos: Optional[List[float]] = None,
        end_pos: Optional[List[float]] = None,
        odom_file: Optional[str] = None,
        scene_id: str = "realworld/unknown",
        start_time_s: Optional[float] = None,
        end_time_s: Optional[float] = None,
    ) -> Dict:
        """
        Extract frames from video and build episode dict.

        Returns:
            Dict with keys matching Gate 1 / Gate 8 output:
              episode_id, scene_id, frames_dir, frame_paths, frame_timestamps,
              start_position, start_rotation, reference_path, goals,
              source_type, source_path, odom_raw, poses
        """
        import cv2

        ep_dir = output_dir / f"episode_{episode_id:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / video_fps if video_fps > 0 else 0.0

        # Time window filtering
        start_frame = int((start_time_s or 0.0) * video_fps)
        end_frame = int((end_time_s * video_fps) if end_time_s else total_frames)

        step = max(1, int(video_fps / frame_hz))
        frame_idx = 0
        saved_count = 0
        frame_paths = []
        frame_timestamps = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx < start_frame:
                frame_idx += 1
                continue
            if frame_idx > end_frame:
                break
            if (frame_idx - start_frame) % step == 0:
                fname = ep_dir / f"frame_{saved_count:04d}_rgb.png"
                cv2.imwrite(str(fname), frame)
                frame_paths.append(str(fname))
                frame_timestamps.append(frame_idx / video_fps)
                saved_count += 1
            frame_idx += 1
        cap.release()

        if saved_count == 0:
            raise ValueError(f"No frames extracted from {self.video_path} "
                             f"(fps={video_fps:.1f}, duration={duration_s:.1f}s)")

        # Build reference path from odom or interpolation
        odom_raw = None
        start_rotation = [0.0, 0.0, 0.0, 1.0]

        if odom_file and Path(odom_file).exists():
            with open(odom_file) as f:
                odom_data = json.load(f)
            if isinstance(odom_data, list) and odom_data:
                if "position" in odom_data[0]:
                    reference_path = [p["position"] for p in odom_data]
                    start_rotation = odom_data[0].get("rotation", [0, 0, 0, 1])
                elif isinstance(odom_data[0], list) and len(odom_data[0]) == 3:
                    reference_path = odom_data  # plain list of [x,y,z]
                else:
                    reference_path = _interpolate_path(start_pos, end_pos, saved_count)
            elif isinstance(odom_data, dict) and "frames" in odom_data:
                reference_path = [f["position"] for f in odom_data["frames"]]
                start_rotation = odom_data["frames"][0].get("rotation", [0, 0, 0, 1])
                odom_data = odom_data["frames"]
            else:
                reference_path = _interpolate_path(start_pos, end_pos, saved_count)
            odom_raw = odom_data
        else:
            reference_path = _interpolate_path(start_pos, end_pos, saved_count)

        if not reference_path:
            reference_path = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]

        # Build poses.json (one entry per extracted frame)
        n_path = len(reference_path)
        poses_frames = []
        for i, (fp, ts) in enumerate(zip(frame_paths, frame_timestamps)):
            path_idx = min(int(i * n_path / max(1, saved_count)), n_path - 1)
            poses_frames.append({
                "frame_index": i,
                "timestamp_s": round(ts, 4),
                "image_path": str(fp),
                "position": reference_path[path_idx],
                "rotation": start_rotation,
            })

        poses = {"frames": poses_frames, "n_frames": saved_count, "source": "video_adapter"}
        poses_path = ep_dir / "poses.json"
        with open(poses_path, "w") as f:
            json.dump(poses, f, indent=2)

        path_len = _path_length(reference_path)

        return {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "frames_dir": str(ep_dir),
            "frame_paths": frame_paths,
            "frame_timestamps": frame_timestamps,
            "start_position": reference_path[0],
            "start_rotation": start_rotation,
            "reference_path": reference_path,
            "goals": [{"position": reference_path[-1], "radius": 3.0}],
            "info": {
                "path_length_m": round(path_len, 3),
                "video_fps": round(video_fps, 2),
                "video_duration_s": round(duration_s, 2),
                "frames_extracted": saved_count,
            },
            "source_type": "video",
            "source_path": self.video_path,
            "poses": poses,
            "odom_raw": odom_raw,
        }


def _interpolate_path(
    start_pos: Optional[List[float]],
    end_pos: Optional[List[float]],
    n_points: int,
) -> List[List[float]]:
    """Linearly interpolate a reference path between start and end positions."""
    sp = start_pos or [0.0, 0.0, 0.0]
    ep = end_pos or [sp[0] + 5.0, sp[1], sp[2]]
    n = max(2, n_points)
    return [
        [sp[j] + (ep[j] - sp[j]) * i / (n - 1) for j in range(3)]
        for i in range(n)
    ]


def _path_length(path: List[List[float]]) -> float:
    return sum(
        math.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3)))
        for a, b in zip(path[:-1], path[1:])
    ) if len(path) >= 2 else 0.0
