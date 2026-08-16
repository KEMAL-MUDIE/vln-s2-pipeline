#!/usr/bin/env python3
"""
Gate 1b: Universal Frame Extractor  (drop-in replacement for Habitat renderer)

Accepts any real or sim-captured data and produces IDENTICAL output to Gate 1
(renderer.py / run_renderer.py), so all downstream gates (2/3/4) work unchanged.

Supported input sources
-----------------------
  --source ros2    : ROS2 rosbag (.db3 file or directory)
  --source ros1    : ROS1 rosbag (.bag file)
  --source video   : MP4 / AVI / MOV / MKV video file
  --source imgdir  : directory of pre-extracted JPEGs/PNGs (sorted by name)
  --source isaac   : Isaac Sim USD-recorder export (images + json pose log)
  --source auto    : detect source type from path extension  [DEFAULT]

Key principle
-------------
The gate solves one problem: "given a sequence of real/sim frames and a path of
3D waypoints, extract the frames that best represent start / turns / goal."

Two alignment strategies are chosen automatically:
  ODOM (preferred) — rosbag/isaac carry odometry or pose messages;
                     each waypoint is matched to the nearest odom pose in time,
                     then the RGB frame at that timestamp is extracted.
  TEMPORAL (fallback) — no odometry available; waypoints are uniformly mapped
                        to the recording timeline, frames sampled proportionally.

Output (identical to Gate 1 renderer)
--------------------------------------
  <output_dir>/
    episode_<N>/
      frame_0000_rgb.jpg   (start)
      frame_0001_rgb.jpg   (turn 1, if present)
      ...
      frame_XXXX_rgb.jpg   (goal)
      poses.json           (camera config + per-frame metadata)

Usage examples
--------------
  # ROS2 bag + odometry topic + path waypoints JSON
  python3 gate1_renderer/rosbag_extractor.py \\
      --source ros2 \\
      --bag-path /mnt/nvme0/rosbags/scout_run_001.db3 \\
      --image-topic /camera/camera/color/image_raw \\
      --odom-topic /gdq/msg/gdq_odom \\
      --waypoints-json /mnt/data/episode_001_path.json \\
      --output-dir s2_pipeline_new/outputs/rendered_frames \\
      --episode-id 1001

  # Video file only (no odometry — temporal alignment)
  python3 gate1_renderer/rosbag_extractor.py \\
      --source video \\
      --video-path /mnt/data/robot_run.mp4 \\
      --waypoints-json /mnt/data/episode_001_path.json \\
      --output-dir s2_pipeline_new/outputs/rendered_frames \\
      --episode-id 1001

  # Batch: process a directory of bags, each paired with a waypoints JSON
  python3 gate1_renderer/rosbag_extractor.py \\
      --source ros2 \\
      --batch-manifest /mnt/data/bag_manifest.json \\
      --output-dir s2_pipeline_new/outputs/rendered_frames

  # Isaac Sim export directory
  python3 gate1_renderer/rosbag_extractor.py \\
      --source isaac \\
      --bag-path /mnt/data/isaac_export/episode_001 \\
      --output-dir s2_pipeline_new/outputs/rendered_frames \\
      --episode-id 1001
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Output format (must match Gate 1 exactly) ────────────────────────────────

CAMERA_CFG = {
    "width": 640,
    "height": 480,
    "hfov": 90.0,
    "sensor_height": 1.25,
}
JPEG_QUALITY = 88
TURN_THRESHOLD_DEG = 25.0   # same as run_renderer.py


# ── Geometry helpers (same as run_renderer.py) ────────────────────────────────

def heading_xz(p1: List[float], p2: List[float]) -> float:
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2]
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, -dz))


def angle_diff(a: float, b: float) -> float:
    return (b - a + 180.0) % 360.0 - 180.0


def heading_to_quat(heading_deg: float) -> List[float]:
    half = math.radians(heading_deg) / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def quat_from_yaw(yaw_rad: float) -> List[float]:
    half = yaw_rad / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def dist3d(p1: List[float], p2: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def dist2d_xz(p1: List[float], p2: List[float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2)


def select_key_waypoints(path: List[List[float]],
                          start_rotation: Optional[List[float]] = None
                          ) -> List[Dict]:
    """Identify start, turn, and goal waypoints (same logic as run_renderer.py)."""
    n = len(path)
    kws = []

    rot = start_rotation or [0.0, 0.0, 0.0, 1.0]
    kws.append({"waypoint_idx": 0, "position": path[0],
                 "rotation": rot, "label": "start"})

    prev_h = None
    for i in range(n - 1):
        h = heading_xz(path[i], path[i + 1])
        rot = heading_to_quat(h)
        if prev_h is not None and abs(angle_diff(prev_h, h)) >= TURN_THRESHOLD_DEG:
            kws.append({"waypoint_idx": i, "position": path[i],
                         "rotation": rot, "label": f"turn_{i}"})
        prev_h = h

    if n >= 2:
        last_h = heading_xz(path[-2], path[-1])
        kws.append({"waypoint_idx": n - 1, "position": path[-1],
                     "rotation": heading_to_quat(last_h), "label": "goal"})

    # Deduplicate
    seen, unique = set(), []
    for kw in kws:
        key = tuple(round(x, 2) for x in kw["position"])
        if key not in seen:
            seen.add(key)
            unique.append(kw)
    return unique


# ── Frame-source base ─────────────────────────────────────────────────────────

class FrameSource:
    """
    Abstract frame source.  Subclasses implement open() / close() / frames().
    frames() → iterator of (timestamp_sec: float, rgb_bgr: np.ndarray, pose: dict|None)
    pose is optional: {"x": m, "y": m, "z": m, "yaw_rad": rad}  — set None if unavailable.
    """

    def open(self):
        pass

    def close(self):
        pass

    def frames(self):
        """Yield (timestamp_sec, rgb_hwc_uint8, pose_or_None)."""
        raise NotImplementedError

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()


# ── ROS2 bag source ──────────────────────────────────────────────────────────

class ROS2BagSource(FrameSource):
    """
    Reads a ROS2 rosbag (.db3 or directory containing *.db3 + metadata.yaml).
    Requires: rosbag2_py  (pip install rosbag2-py  or sourcing ROS2 jazzy)
    Image topic: sensor_msgs/Image or sensor_msgs/CompressedImage
    Odometry topic: nav_msgs/Odometry  (optional)
    """

    def __init__(self, bag_path: str, image_topic: str, odom_topic: Optional[str] = None):
        self.bag_path = bag_path
        self.image_topic = image_topic
        self.odom_topic = odom_topic
        self._reader = None
        self._odom_cache: List[Tuple[float, dict]] = []  # [(t_sec, pose)]

    def open(self):
        try:
            import rosbag2_py
        except ImportError:
            raise RuntimeError(
                "rosbag2_py not available. Source ROS2 Jazzy:\n"
                "  source /opt/ros/jazzy/setup.bash\n"
                "  pip install rosbag2-py"
            )
        storage_opts = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id="sqlite3")
        conv_opts = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"
        )
        self._reader = rosbag2_py.SequentialReader()
        self._reader.open(storage_opts, conv_opts)

        # Pre-load odometry
        if self.odom_topic:
            self._preload_odom()

    def _preload_odom(self):
        """Read all odometry messages into memory for pose alignment."""
        try:
            from rclpy.serialization import deserialize_message
            from nav_msgs.msg import Odometry
        except ImportError:
            print("  [WARN] rclpy not available — skipping odometry alignment")
            return

        storage_opts = __import__("rosbag2_py").StorageOptions(uri=self.bag_path, storage_id="sqlite3")
        conv_opts = __import__("rosbag2_py").ConverterOptions("cdr", "cdr")
        reader2 = __import__("rosbag2_py").SequentialReader()
        reader2.open(storage_opts, conv_opts)

        while reader2.has_next():
            topic, data, ts_ns = reader2.read_next()
            if topic != self.odom_topic:
                continue
            try:
                msg = deserialize_message(data, Odometry)
                t = ts_ns * 1e-9
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y ** 2 + q.z ** 2))
                self.odom_cache.append((t, {"x": p.x, "y": p.y, "z": p.z, "yaw_rad": yaw}))
            except Exception:
                pass
        print(f"  Loaded {len(self._odom_cache)} odometry messages")

    def frames(self):
        try:
            from rclpy.serialization import deserialize_message
            from sensor_msgs.msg import Image, CompressedImage
            import numpy as np
        except ImportError:
            raise RuntimeError("rclpy not installed — source /opt/ros/jazzy/setup.bash")

        # Find out if topic is compressed
        topic_types = {t.name: t.type for t in self._reader.get_all_topics_and_types()}
        msg_type_str = topic_types.get(self.image_topic, "sensor_msgs/msg/Image")
        is_compressed = "Compressed" in msg_type_str

        while self._reader.has_next():
            topic, data, ts_ns = self._reader.read_next()
            if topic != self.image_topic:
                continue

            t_sec = ts_ns * 1e-9
            try:
                if is_compressed:
                    import cv2
                    msg = deserialize_message(data, CompressedImage)
                    buf = np.frombuffer(msg.data, dtype=np.uint8)
                    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                else:
                    msg = deserialize_message(data, Image)
                    arr = np.frombuffer(msg.data, dtype=np.uint8)
                    h, w = msg.height, msg.width
                    if msg.encoding in ("bgr8", "bgr"):
                        import cv2
                        bgr = arr.reshape((h, w, 3))
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    elif msg.encoding in ("rgb8", "rgb"):
                        rgb = arr.reshape((h, w, 3))
                    elif msg.encoding in ("rgba8", "rgba"):
                        rgb = arr.reshape((h, w, 4))[:, :, :3]
                    else:
                        import cv2
                        rgb = arr.reshape((h, w, -1))[:, :, :3]
            except Exception as e:
                continue

            # Nearest odom pose by time
            pose = self._nearest_odom(t_sec) if self._odom_cache else None
            yield t_sec, rgb, pose

    def _nearest_odom(self, t: float) -> Optional[dict]:
        if not self._odom_cache:
            return None
        # Binary search
        lo, hi = 0, len(self._odom_cache) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._odom_cache[mid][0] < t:
                lo = mid + 1
            else:
                hi = mid
        best_idx = lo
        if lo > 0 and abs(self._odom_cache[lo - 1][0] - t) < abs(self._odom_cache[lo][0] - t):
            best_idx = lo - 1
        return self._odom_cache[best_idx][1]

    def close(self):
        self._reader = None


# ── ROS1 bag source ──────────────────────────────────────────────────────────

class ROS1BagSource(FrameSource):
    """
    Reads a ROS1 .bag file.
    Requires: rosbag  (pip install rosbag  or ROS Noetic)
    """

    def __init__(self, bag_path: str, image_topic: str, odom_topic: Optional[str] = None):
        self.bag_path = bag_path
        self.image_topic = image_topic
        self.odom_topic = odom_topic
        self._bag = None
        self._odom_cache: List[Tuple[float, dict]] = []

    def open(self):
        try:
            import rosbag
        except ImportError:
            raise RuntimeError("rosbag not installed (ROS1 Noetic required)")
        self._bag = rosbag.Bag(self.bag_path, "r")
        if self.odom_topic:
            self._preload_odom()

    def _preload_odom(self):
        for topic, msg, t in self._bag.read_messages(topics=[self.odom_topic]):
            ts = t.to_sec()
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y ** 2 + q.z ** 2))
            self._odom_cache.append((ts, {"x": p.x, "y": p.y, "z": p.z, "yaw_rad": yaw}))
        print(f"  Loaded {len(self._odom_cache)} odometry messages")

    def frames(self):
        import numpy as np
        import cv2
        for topic, msg, t in self._bag.read_messages(topics=[self.image_topic]):
            ts = t.to_sec()
            try:
                if hasattr(msg, "format"):  # CompressedImage
                    buf = np.frombuffer(msg.data, dtype=np.uint8)
                    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                else:  # Image
                    arr = np.frombuffer(msg.data, dtype=np.uint8)
                    h, w = msg.height, msg.width
                    bgr = arr.reshape((h, w, 3))
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if "bgr" in msg.encoding else bgr
                pose = self._nearest_odom(ts) if self._odom_cache else None
                yield ts, rgb, pose
            except Exception:
                continue

    def _nearest_odom(self, t: float) -> Optional[dict]:
        if not self._odom_cache:
            return None
        lo, hi = 0, len(self._odom_cache) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._odom_cache[mid][0] < t:
                lo = mid + 1
            else:
                hi = mid
        best = lo
        if lo > 0 and abs(self._odom_cache[lo - 1][0] - t) < abs(self._odom_cache[lo][0] - t):
            best = lo - 1
        return self._odom_cache[best][1]

    def close(self):
        if self._bag:
            self._bag.close()


# ── Video source ──────────────────────────────────────────────────────────────

class VideoSource(FrameSource):
    """
    Reads frames from a video file (MP4 / AVI / MOV / MKV).
    No odometry — temporal alignment only.
    Requires: opencv-python
    """

    def __init__(self, video_path: str, sample_fps: float = 2.0):
        self.video_path = video_path
        self.sample_fps = sample_fps  # sample at this rate (not every frame)
        self._cap = None

    def open(self):
        import cv2
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

    def frames(self):
        import cv2
        cap = self._cap
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(video_fps / self.sample_fps))
        frame_idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                import numpy as np
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ts = frame_idx / video_fps
                yield ts, rgb, None  # no odometry
            frame_idx += 1

    def close(self):
        if self._cap:
            self._cap.release()


# ── Image directory source ────────────────────────────────────────────────────

class ImageDirSource(FrameSource):
    """
    Reads a directory of pre-extracted images (JPEGs / PNGs), sorted by filename.
    Optionally reads poses from poses.json or odom.json in the same directory.
    No opencv dependency — uses PIL.
    """

    def __init__(self, dir_path: str):
        self.dir_path = Path(dir_path)
        self._images: List[Path] = []
        self._poses: List[Optional[dict]] = []

    def open(self):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self._images = sorted(
            p for p in self.dir_path.iterdir()
            if p.suffix.lower() in exts
        )
        if not self._images:
            raise RuntimeError(f"No images found in {self.dir_path}")

        # Try to load poses from odom.json or poses.json
        for fname in ("odom.json", "poses.json", "pose.json"):
            pfile = self.dir_path / fname
            if pfile.exists():
                data = json.load(open(pfile))
                if isinstance(data, list):
                    self._poses = data
                elif isinstance(data, dict) and "frames" in data:
                    # Gate 1 poses.json format
                    self._poses = [
                        {"x": f["position"][0], "y": f["position"][1],
                         "z": f["position"][2], "yaw_rad": 0.0}
                        for f in data["frames"]
                    ]
                break

    def frames(self):
        from PIL import Image
        import numpy as np
        n = len(self._images)
        for i, img_path in enumerate(self._images):
            try:
                img = Image.open(img_path).convert("RGB")
                rgb = np.array(img)
                pose = self._poses[i] if i < len(self._poses) else None
                yield float(i), rgb, pose
            except Exception as e:
                print(f"  [WARN] Could not read {img_path}: {e}")
                continue


# ── Isaac Sim source ──────────────────────────────────────────────────────────

class IsaacSimSource(FrameSource):
    """
    Reads Isaac Sim USD-recorder / Replicator export.
    Expected directory structure:
      <export_dir>/
        rgb/
          0000.png  0001.png  ...
        camera_params.json   (optional: focal_length, sensor size)
        pose_log.json        (optional: [{"ts": float, "x": ..., "y": ..., "z": ..., "qx": ..., "qy": ..., "qz": ..., "qw": ...}])
    """

    def __init__(self, export_dir: str):
        self.export_dir = Path(export_dir)
        self._images: List[Path] = []
        self._pose_log: List[dict] = []

    def open(self):
        rgb_dir = self.export_dir / "rgb"
        if not rgb_dir.exists():
            # Try flat layout
            rgb_dir = self.export_dir
        exts = {".png", ".jpg", ".jpeg"}
        self._images = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in exts)
        if not self._images:
            raise RuntimeError(f"No images found in {rgb_dir}")

        pose_file = self.export_dir / "pose_log.json"
        if pose_file.exists():
            self._pose_log = json.load(open(pose_file))
            print(f"  Loaded {len(self._pose_log)} Isaac Sim poses")

    def frames(self):
        from PIL import Image
        import numpy as np
        n = len(self._images)
        for i, img_path in enumerate(self._images):
            try:
                rgb = np.array(Image.open(img_path).convert("RGB"))
                ts = float(i)
                pose = None
                if i < len(self._pose_log):
                    pl = self._pose_log[i]
                    # Isaac Sim uses Z-up; convert to ROS convention (Y-up)
                    qx = pl.get("qx", 0); qy = pl.get("qy", 0)
                    qz = pl.get("qz", 0); qw = pl.get("qw", 1)
                    yaw = math.atan2(2 * (qw * qz + qx * qy),
                                     1 - 2 * (qy ** 2 + qz ** 2))
                    pose = {"x": pl.get("x", 0), "y": pl.get("y", 0),
                            "z": pl.get("z", 0), "yaw_rad": yaw}
                yield ts, rgb, pose
            except Exception as e:
                print(f"  [WARN] Frame {i}: {e}")


# ── Source factory ────────────────────────────────────────────────────────────

def detect_source_type(path: str) -> str:
    p = path.lower()
    if p.endswith(".db3") or (os.path.isdir(path) and os.path.exists(os.path.join(path, "metadata.yaml"))):
        return "ros2"
    if p.endswith(".bag"):
        return "ros1"
    if any(p.endswith(ext) for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm")):
        return "video"
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "rgb")):
        return "isaac"
    if os.path.isdir(path):
        return "imgdir"
    return "video"


def make_source(source_type: str, args) -> FrameSource:
    t = source_type
    if t == "auto":
        t = detect_source_type(args.bag_path or "")
    if t == "ros2":
        return ROS2BagSource(args.bag_path, args.image_topic, args.odom_topic)
    if t == "ros1":
        return ROS1BagSource(args.bag_path, args.image_topic, args.odom_topic)
    if t == "video":
        return VideoSource(args.bag_path, sample_fps=args.sample_fps)
    if t == "imgdir":
        return ImageDirSource(args.bag_path)
    if t == "isaac":
        return IsaacSimSource(args.bag_path)
    raise ValueError(f"Unknown source type: {t}")


# ── Frame alignment ──────────────────────────────────────────────────────────

def align_odom(all_frames: List[Tuple], key_waypoints: List[Dict]) -> List[Tuple]:
    """
    ODOM alignment: for each key waypoint, find the frame whose odom position is
    nearest to the waypoint 2D position (XZ in world / XY in odom).

    all_frames: [(ts, rgb, pose), ...]  where pose = {"x","y","z","yaw_rad"}
    key_waypoints: from select_key_waypoints()
    Returns: [(rgb, pose, waypoint_meta), ...] — one per key waypoint
    """
    import numpy as np
    odom_frames = [(i, f) for i, f in enumerate(all_frames) if f[2] is not None]
    if not odom_frames:
        return None  # fall back to temporal

    chosen = []
    used = set()
    for kw in key_waypoints:
        wp_pos = kw["position"]  # [x, y, z] in world coords
        best_i, best_d = None, float("inf")
        for fi, (_, rgb, pose) in odom_frames:
            if fi in used:
                continue
            # Compare in XZ (or XY if odom is 2D)
            dx = pose["x"] - wp_pos[0]
            dz = pose.get("z", pose.get("y", 0)) - wp_pos[2]
            d = math.sqrt(dx * dx + dz * dz)
            if d < best_d:
                best_d, best_i = d, fi
        if best_i is not None:
            used.add(best_i)
            ts, rgb, pose = all_frames[best_i]
            chosen.append((rgb, pose, kw, best_d))

    return chosen


def align_temporal(all_frames: List[Tuple], key_waypoints: List[Dict]) -> List[Tuple]:
    """
    TEMPORAL alignment: distribute key waypoints uniformly across the recording.
    waypoint[0] → frame[0], waypoint[-1] → frame[-1], intermediates proportional.
    """
    n_frames = len(all_frames)
    n_kw = len(key_waypoints)
    if n_frames == 0 or n_kw == 0:
        return []

    chosen = []
    for ki, kw in enumerate(key_waypoints):
        # Map ki/(n_kw-1) to frame index
        frac = ki / max(n_kw - 1, 1)
        fi = min(int(frac * (n_frames - 1)), n_frames - 1)
        ts, rgb, pose = all_frames[fi]
        chosen.append((rgb, pose, kw, None))  # None = no odom distance
    return chosen


# ── Resize helper ─────────────────────────────────────────────────────────────

def resize_rgb(rgb, target_w: int, target_h: int):
    """Resize RGB numpy array to target size using PIL (no cv2 required)."""
    from PIL import Image
    import numpy as np
    h, w = rgb.shape[:2]
    if w == target_w and h == target_h:
        return rgb
    img = Image.fromarray(rgb).resize((target_w, target_h), Image.LANCZOS)
    return np.array(img)


# ── Save episode frames (Gate 1 format) ──────────────────────────────────────

def save_episode_frames(
    chosen: List[Tuple],
    episode_id: int,
    scene_id: str,
    output_dir: Path,
    camera_cfg: dict = CAMERA_CFG,
    overwrite: bool = False,
) -> Dict:
    """
    Save chosen frames as JPEG + poses.json in Gate 1 output format.
    chosen: [(rgb, pose_or_None, waypoint_meta, odom_dist_or_None), ...]
    Returns the poses.json dict.
    """
    from PIL import Image

    ep_dir = output_dir / f"episode_{episode_id:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    poses_path = ep_dir / "poses.json"
    if poses_path.exists() and not overwrite:
        existing = json.load(open(poses_path))
        if existing.get("n_frames", 0) > 0:
            print(f"  ep={episode_id}: already exists ({existing['n_frames']} frames), skipping")
            return existing

    frame_metadata = []
    tw, th = camera_cfg["width"], camera_cfg["height"]

    for fi, (rgb, pose, kw, odom_dist) in enumerate(chosen):
        # Resize to camera spec
        rgb_resized = resize_rgb(rgb, tw, th)

        img_name = f"frame_{fi:04d}_rgb.jpg"
        Image.fromarray(rgb_resized).save(ep_dir / img_name, "JPEG", quality=JPEG_QUALITY)

        # Build position/rotation from pose or waypoint geometry
        if pose is not None:
            position = [pose["x"], pose.get("y", 0.0), pose.get("z", 0.0)]
            rotation = quat_from_yaw(pose.get("yaw_rad", 0.0))
        else:
            position = kw["position"]
            rotation = kw["rotation"]

        frame_meta = {
            "frame_idx": fi,
            "path": img_name,
            "position": position,
            "rotation": rotation,
            "waypoint_idx": kw["waypoint_idx"],
            "label": kw["label"],
        }
        if odom_dist is not None:
            frame_meta["odom_match_dist_m"] = round(odom_dist, 3)
        frame_metadata.append(frame_meta)

    result = {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "n_frames": len(frame_metadata),
        "camera_config": camera_cfg,
        "frames": frame_metadata,
    }
    with open(poses_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Single-episode extraction ─────────────────────────────────────────────────

def extract_episode(
    source: FrameSource,
    episode_id: int,
    scene_id: str,
    waypoints: List[List[float]],
    start_rotation: Optional[List[float]],
    output_dir: Path,
    force_temporal: bool = False,
    overwrite: bool = False,
) -> Dict:
    """
    Full extraction pipeline for one episode:
      1. Load all frames from source
      2. Select key waypoints from path
      3. Align frames to waypoints (odom or temporal)
      4. Save in Gate 1 format
    """
    print(f"  ep={episode_id}: loading frames from source...", flush=True)
    t0 = time.time()

    all_frames = list(source.frames())
    if not all_frames:
        raise RuntimeError("Source returned no frames")

    key_wps = select_key_waypoints(waypoints, start_rotation)
    print(f"  ep={episode_id}: {len(all_frames)} source frames, {len(key_wps)} key waypoints")

    has_odom = any(f[2] is not None for f in all_frames)
    if has_odom and not force_temporal:
        chosen = align_odom(all_frames, key_wps)
        if chosen is None:
            chosen = align_temporal(all_frames, key_wps)
            mode = "temporal (odom fallback)"
        else:
            mode = "odom"
            avg_d = sum(c[3] for c in chosen if c[3] is not None) / max(len(chosen), 1)
            mode = f"odom (avg_match={avg_d:.2f}m)"
    else:
        chosen = align_temporal(all_frames, key_wps)
        mode = "temporal"

    result = save_episode_frames(chosen, episode_id, scene_id, output_dir, overwrite=overwrite)
    elapsed = time.time() - t0
    print(f"  ep={episode_id}: {result['n_frames']} frames saved | align={mode} | {elapsed:.1f}s")
    return result


# ── Batch processing ──────────────────────────────────────────────────────────

def load_batch_manifest(manifest_path: str) -> List[Dict]:
    """
    Batch manifest JSON format:
    [
      {
        "episode_id": 1001,
        "scene_id": "rosbag/scout_run_001",
        "bag_path": "/mnt/data/bags/scout_run_001.db3",
        "waypoints": [[x,y,z], ...],
        "start_rotation": [qx,qy,qz,qw],   (optional)
        "image_topic": "/camera/camera/color/image_raw",  (optional, overrides CLI)
        "odom_topic": "/gdq/msg/gdq_odom"   (optional)
      },
      ...
    ]
    """
    with open(manifest_path) as f:
        return json.load(f)


def load_waypoints_json(path: str) -> Tuple[List[List[float]], Optional[List[float]]]:
    """
    Load waypoints from a JSON file.
    Formats accepted:
      {"waypoints": [[x,y,z], ...], "start_rotation": [qx,qy,qz,qw]}
      {"reference_path": [[x,y,z], ...], "start_rotation": [...]}
      [[x,y,z], ...]  (bare list)
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, None
    wpts = data.get("waypoints") or data.get("reference_path") or data.get("path") or []
    rot = data.get("start_rotation")
    return wpts, rot


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gate 1b: Universal Frame Extractor (replaces Habitat renderer)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--source", default="auto",
                   choices=["auto", "ros2", "ros1", "video", "imgdir", "isaac"],
                   help="Input source type (default: auto-detect)")
    p.add_argument("--bag-path", default=None,
                   help="Path to rosbag / video file / image directory / Isaac export")
    p.add_argument("--image-topic", default="/camera/camera/color/image_raw",
                   help="ROS image topic (ros2/ros1 only)")
    p.add_argument("--odom-topic", default=None,
                   help="ROS odometry topic for pose alignment (optional)")
    p.add_argument("--waypoints-json", default=None,
                   help="JSON file with path waypoints and optional start_rotation")
    p.add_argument("--waypoints-inline", default=None,
                   help='Inline JSON waypoints: "[[x,y,z],...]"')
    p.add_argument("--episode-id", type=int, default=None,
                   help="Episode ID (used for output directory naming)")
    p.add_argument("--scene-id", default=None,
                   help="Scene identifier string (stored in poses.json)")
    p.add_argument("--output-dir", required=True,
                   help="Output directory (same as Gate 1 rendered_frames dir)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract even if output already exists")
    p.add_argument("--force-temporal", action="store_true",
                   help="Skip odometry alignment, always use temporal (uniform) sampling")
    p.add_argument("--sample-fps", type=float, default=2.0,
                   help="Frame sampling rate for video source (default: 2.0 Hz)")
    p.add_argument("--batch-manifest", default=None,
                   help="JSON manifest for batch processing multiple episodes")
    p.add_argument("--dry-run", action="store_true",
                   help="Show extraction plan without writing files")
    return p


def run_single(args) -> Dict:
    """Extract a single episode from CLI args."""
    if args.waypoints_json:
        waypoints, start_rot = load_waypoints_json(args.waypoints_json)
    elif args.waypoints_inline:
        waypoints = json.loads(args.waypoints_inline)
        start_rot = None
    else:
        raise ValueError("Provide --waypoints-json or --waypoints-inline")

    if not waypoints:
        raise ValueError("No waypoints provided")

    episode_id = args.episode_id or 9999
    scene_id = args.scene_id or args.bag_path or "unknown"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        key_wps = select_key_waypoints(waypoints, start_rot)
        print(f"DRY RUN — episode {episode_id}")
        print(f"  Waypoints:    {len(waypoints)}")
        print(f"  Key frames:   {len(key_wps)}")
        for kw in key_wps:
            print(f"    [{kw['label']:12s}] wpt={kw['waypoint_idx']}  pos={[round(x,3) for x in kw['position']]}")
        return {}

    source = make_source(args.source, args)
    with source:
        return extract_episode(
            source, episode_id, scene_id, waypoints, start_rot,
            output_dir, args.force_temporal, args.overwrite
        )


def run_batch(args) -> None:
    """Process a JSON manifest of multiple episodes."""
    entries = load_batch_manifest(args.batch_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(entries)
    done, errors = 0, 0

    print(f"Batch: {total} episodes from manifest {args.batch_manifest}")
    t0 = time.time()

    for i, entry in enumerate(entries):
        ep_id = entry["episode_id"]
        bag = entry.get("bag_path", args.bag_path)
        scene_id = entry.get("scene_id", bag or "unknown")
        waypoints = entry.get("waypoints") or entry.get("reference_path") or []
        start_rot = entry.get("start_rotation")
        img_topic = entry.get("image_topic", args.image_topic)
        odom_topic = entry.get("odom_topic", args.odom_topic)

        print(f"\n[{i+1}/{total}] ep={ep_id}  bag={os.path.basename(bag or '')}")

        if not waypoints:
            print(f"  [SKIP] No waypoints for ep={ep_id}")
            errors += 1
            continue

        # Patch args for this entry
        args_copy = argparse.Namespace(**vars(args))
        args_copy.bag_path = bag
        args_copy.image_topic = img_topic
        args_copy.odom_topic = odom_topic

        try:
            source = make_source(args.source if args.source != "auto" else detect_source_type(bag or ""), args_copy)
            with source:
                extract_episode(
                    source, ep_id, scene_id, waypoints, start_rot,
                    output_dir, args.force_temporal, args.overwrite
                )
            done += 1
        except Exception as e:
            print(f"  [ERR] ep={ep_id}: {e}")
            errors += 1

        elapsed = time.time() - t0
        rate = (done + errors) / elapsed if elapsed > 0 else 0
        eta = (total - done - errors) / rate if rate > 0 else 0
        print(f"  Progress: {done} ok / {errors} err / {total} total | ETA {eta/60:.1f}m")

    elapsed = time.time() - t0
    print(f"\n=== Batch complete: {done}/{total} ok, {errors} errors in {elapsed:.1f}s ===")


def main():
    args = build_parser().parse_args()

    print("=== Gate 1b: Universal Frame Extractor ===")
    print(f"  Source type: {args.source}")

    if args.batch_manifest:
        run_batch(args)
    elif args.bag_path or args.waypoints_inline:
        result = run_single(args)
        if result:
            print(f"\nOutput: {args.output_dir}/episode_{(args.episode_id or 9999):06d}/")
            print(f"Frames: {result.get('n_frames', 0)}")
    else:
        build_parser().print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
