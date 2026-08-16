#!/usr/bin/env python3
"""
Gate 8: ROS2 Bag Adapter
Converts ROS2 .bag files (real robot captures) into the same
frames + poses format produced by Gate 1 (Habitat renderer).

After this adapter, the data flows through Gates 2-6 unchanged.

Expected ROS2 topics:
  /camera/camera/color/image_raw   (sensor_msgs/Image, ~40 Hz)
  /gdq/msg/gdq_odom                (nav_msgs/Odometry, ~60 Hz)

Output format (same as Gate 1):
  frames/episode_{id:06d}/frame_{i:04d}_rgb.png
  frames/episode_{id:06d}/poses.json
"""
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def quaternion_to_list(q) -> List[float]:
    """ROS geometry_msgs/Quaternion → [qx, qy, qz, qw]."""
    return [q.x, q.y, q.z, q.w]


def point_to_list(p) -> List[float]:
    """ROS geometry_msgs/Point → [x, y, z]."""
    return [p.x, p.y, p.z]


class RosbagAdapter:
    """
    Extracts frames and poses from a ROS2 bag file.

    Usage:
        adapter = RosbagAdapter("/path/to/recording.bag")
        episode = adapter.extract(
            output_dir=Path("outputs/rendered_frames"),
            episode_id=0,
            start_time=None,    # None = from bag start
            end_time=None,      # None = to bag end
            frame_hz=2.0,       # downsample camera to this rate
        )
        # episode now has same format as Gate 1 output
        # → feed directly into gate2/path_analyzer.py
    """

    def __init__(self, bag_path: str):
        self.bag_path = bag_path
        self._check_rclpy()

    @staticmethod
    def _check_rclpy():
        try:
            import rclpy
            from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        except ImportError:
            raise RuntimeError(
                "ROS2 bag reading requires rclpy + rosbag2_py.\n"
                "Source ROS2 Jazzy: source /opt/ros/jazzy/setup.bash"
            )

    def extract(
        self,
        output_dir: Path,
        episode_id: int = 0,
        start_time_s: Optional[float] = None,
        end_time_s: Optional[float] = None,
        frame_hz: float = 2.0,
        scene_id: str = "realworld/unknown",
    ) -> Dict:
        """
        Extract frames + poses from the bag and save to output_dir.
        Returns episode metadata dict (same format as Gate 1).
        """
        import rclpy
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Image
        from nav_msgs.msg import Odometry
        import cv2
        import numpy as np

        ep_dir = output_dir / f"episode_{episode_id:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        reader = SequentialReader()
        storage_opts = StorageOptions(uri=self.bag_path, storage_id="sqlite3")
        conv_opts = ConverterOptions("", "")
        reader.open(storage_opts, conv_opts)

        frame_interval_ns = int(1e9 / frame_hz)
        last_frame_time_ns = None
        frame_count = 0

        poses_list = []        # all odom poses (high rate)
        frame_metadata = []    # downsampled frames

        cam_topic = "/camera/camera/color/image_raw"
        odom_topic = "/gdq/msg/gdq_odom"

        latest_odom = None     # most recent odom reading

        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()

            # Filter by time window
            t_s = timestamp_ns * 1e-9
            if start_time_s is not None and t_s < start_time_s:
                continue
            if end_time_s is not None and t_s > end_time_s:
                break

            if topic == odom_topic:
                msg = deserialize_message(data, Odometry)
                pos = point_to_list(msg.pose.pose.position)
                rot = quaternion_to_list(msg.pose.pose.orientation)
                latest_odom = {"position": pos, "rotation": rot, "timestamp_s": t_s}
                poses_list.append(latest_odom.copy())

            elif topic == cam_topic:
                if last_frame_time_ns is not None and (timestamp_ns - last_frame_time_ns) < frame_interval_ns:
                    continue
                last_frame_time_ns = timestamp_ns

                msg = deserialize_message(data, Image)
                # Convert ROS Image to numpy
                if msg.encoding == "bgr8":
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif msg.encoding == "rgb8":
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                else:
                    continue  # unsupported encoding

                # Save frame
                from PIL import Image as PILImage
                frame_path = ep_dir / f"frame_{frame_count:04d}_rgb.png"
                PILImage.fromarray(img).save(frame_path)

                frame_meta = {
                    "frame_idx": frame_count,
                    "rgb_path": str(frame_path.relative_to(output_dir)),
                    "timestamp_s": t_s,
                    "position": latest_odom["position"] if latest_odom else [0, 0, 0],
                    "rotation": latest_odom["rotation"] if latest_odom else [0, 0, 0, 1],
                    "waypoint_idx": frame_count,
                    "frame_type": "real_capture",
                }
                frame_metadata.append(frame_meta)
                frame_count += 1

        # Build reference_path from downsampled camera poses
        reference_path = [f["position"] for f in frame_metadata]

        result = {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "n_frames": frame_count,
            "source": "rosbag",
            "bag_path": self.bag_path,
            "camera_config": {"width": 640, "height": 480, "frame_hz": frame_hz},
            "frames": frame_metadata,
            # Fields needed by downstream gates
            "reference_path": reference_path,
            "start_position": reference_path[0] if reference_path else [0, 0, 0],
            "start_rotation": frame_metadata[0]["rotation"] if frame_metadata else [0, 0, 0, 1],
            "goals": [{"position": reference_path[-1] if reference_path else [0, 0, 0], "radius": 3.0}],
            "info": {},  # geodesic_distance will be computed by Gate 6
        }

        poses_path = ep_dir / "poses.json"
        with open(poses_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Extracted {frame_count} frames, {len(poses_list)} odom poses → {ep_dir}")
        return result


class VideoAdapter:
    """
    Gate 8 alternative: extract frames from MP4 + CSV pose file.

    CSV format: timestamp_s, x, y, z, qx, qy, qz, qw
    """

    def extract(
        self,
        video_path: str,
        pose_csv: str,
        output_dir: Path,
        episode_id: int = 0,
        frame_hz: float = 2.0,
        scene_id: str = "realworld/unknown",
    ) -> Dict:
        import csv
        import cv2
        from PIL import Image as PILImage

        ep_dir = output_dir / f"episode_{episode_id:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Load poses
        poses = []
        with open(pose_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                poses.append({
                    "timestamp_s": float(row["timestamp_s"]),
                    "position": [float(row["x"]), float(row["y"]), float(row["z"])],
                    "rotation": [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])],
                })

        # Extract video frames
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / frame_hz)

        frame_count = 0
        video_frame_idx = 0
        frame_metadata = []

        while cap.isOpened():
            ret, bgr = cap.read()
            if not ret:
                break
            if video_frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                t = video_frame_idx / video_fps

                # Find closest pose by timestamp
                closest_pose = min(poses, key=lambda p: abs(p["timestamp_s"] - t)) if poses else None
                pos = closest_pose["position"] if closest_pose else [0, 0, 0]
                rot = closest_pose["rotation"] if closest_pose else [0, 0, 0, 1]

                frame_path = ep_dir / f"frame_{frame_count:04d}_rgb.png"
                PILImage.fromarray(rgb).save(frame_path)
                frame_metadata.append({
                    "frame_idx": frame_count,
                    "rgb_path": str(frame_path.relative_to(output_dir)),
                    "timestamp_s": t,
                    "position": pos,
                    "rotation": rot,
                    "waypoint_idx": frame_count,
                    "frame_type": "video_capture",
                })
                frame_count += 1
            video_frame_idx += 1

        cap.release()
        reference_path = [f["position"] for f in frame_metadata]

        result = {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "n_frames": frame_count,
            "source": "video",
            "video_path": video_path,
            "frames": frame_metadata,
            "reference_path": reference_path,
            "start_position": reference_path[0] if reference_path else [0, 0, 0],
            "start_rotation": frame_metadata[0]["rotation"] if frame_metadata else [0, 0, 0, 1],
            "goals": [{"position": reference_path[-1] if reference_path else [0, 0, 0], "radius": 3.0}],
            "info": {},
        }

        poses_path = ep_dir / "poses.json"
        with open(poses_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Extracted {frame_count} frames from video → {ep_dir}")
        return result
