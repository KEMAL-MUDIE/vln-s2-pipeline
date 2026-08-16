#!/usr/bin/env python3
"""
Gate 1b Helper: Build a batch manifest for rosbag_extractor.py

Converts any of the following into the batch manifest JSON format:

  MODE 1 — from GT R2R JSON + bag directory
    Each bag file in --bag-dir is matched to an episode by episode_id embedded
    in the filename (e.g. episode_001042.db3, or scout_run_1042.bag).
    Waypoints are taken from the GT JSON reference_path.

  MODE 2 — from a path log CSV or JSON (real-robot recording)
    Columns / keys: episode_id, bag_path, [x,y,z] waypoints in the log itself.
    Used when there is no GT reference_path but odometry is available.

  MODE 3 — from Isaac Sim export directory
    Each subdirectory episode_XXXXXX/ is treated as one episode with images +
    pose_log.json. Waypoints extracted from pose_log.json (start + end + turns).

Output: bag_manifest.json ready for:
  python3 gate1_renderer/rosbag_extractor.py \\
      --source auto --batch-manifest bag_manifest.json \\
      --output-dir outputs/rendered_frames

Usage:
  # Mode 1: GT R2R episodes + ROS2 bag directory
  python3 gate1_renderer/make_manifest.py \\
      --mode gt-bags \\
      --gt-path /mnt/data/val_unseen_patched.json.gz \\
      --bag-dir /mnt/nvme0/rosbags/ \\
      --bag-ext .db3 \\
      --image-topic /camera/camera/color/image_raw \\
      --odom-topic /gdq/msg/gdq_odom \\
      --output manifest.json

  # Mode 3: Isaac Sim export
  python3 gate1_renderer/make_manifest.py \\
      --mode isaac-dir \\
      --bag-dir /mnt/data/isaac_exports/ \\
      --output manifest.json
"""

import argparse
import gzip
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional

TURN_THRESHOLD_DEG = 25.0


def heading_xz(p1, p2):
    dx = p2[0] - p1[0]; dz = p2[2] - p1[2]
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, -dz))


def angle_diff(a, b):
    return (b - a + 180.0) % 360.0 - 180.0


def extract_turn_waypoints(path: List[List[float]]) -> List[List[float]]:
    """
    Reduce a dense path to key waypoints (start + turns + goal).
    Used for Isaac Sim pose_log where all poses are available.
    """
    if len(path) <= 2:
        return path
    key = [path[0]]
    prev_h = None
    for i in range(len(path) - 1):
        h = heading_xz(path[i], path[i + 1])
        if prev_h is not None and abs(angle_diff(prev_h, h)) >= TURN_THRESHOLD_DEG:
            key.append(path[i])
        prev_h = h
    key.append(path[-1])
    return key


def match_bag_to_episode(bag_files: List[Path], episode_id: int) -> Optional[Path]:
    """Find a bag file whose name contains the episode_id."""
    id_str = str(episode_id)
    for bag in bag_files:
        if id_str in bag.stem:
            return bag
    return None


# ── Mode 1: GT R2R + bag directory ───────────────────────────────────────────

def build_manifest_gt_bags(args) -> List[Dict]:
    if not args.gt_path:
        sys.exit("--gt-path required for mode gt-bags")
    if not args.bag_dir:
        sys.exit("--bag-dir required for mode gt-bags")

    opener = gzip.open if args.gt_path.endswith(".gz") else open
    with opener(args.gt_path, "rt") as f:
        gt_data = json.load(f)
    episodes = gt_data["episodes"]

    bag_dir = Path(args.bag_dir)
    ext = args.bag_ext or ".db3"
    bag_files = sorted(bag_dir.rglob(f"*{ext}"))
    print(f"Found {len(bag_files)} bag files in {bag_dir}")
    print(f"Matching against {len(episodes)} GT episodes...")

    manifest = []
    matched = 0
    for ep in episodes:
        ep_id = ep["episode_id"]
        bag = match_bag_to_episode(bag_files, ep_id)
        if bag is None:
            continue
        matched += 1
        manifest.append({
            "episode_id": ep_id,
            "scene_id": ep.get("scene_id", f"bag/{bag.stem}"),
            "bag_path": str(bag),
            "waypoints": ep["reference_path"],
            "start_rotation": ep.get("start_rotation"),
            "image_topic": args.image_topic or "/camera/camera/color/image_raw",
            "odom_topic": args.odom_topic or None,
        })

    print(f"Matched: {matched}/{len(episodes)} episodes")
    return manifest


# ── Mode 2: Path log CSV/JSON ─────────────────────────────────────────────────

def build_manifest_path_log(args) -> List[Dict]:
    if not args.path_log:
        sys.exit("--path-log required for mode path-log")

    with open(args.path_log) as f:
        if args.path_log.endswith(".json"):
            entries = json.load(f)
        else:
            import csv
            reader = csv.DictReader(f)
            entries = list(reader)

    manifest = []
    for row in entries:
        ep_id = int(row.get("episode_id", row.get("id", 0)))
        bag = row.get("bag_path", row.get("bag", ""))
        wpts_raw = row.get("waypoints") or row.get("reference_path") or "[]"
        if isinstance(wpts_raw, str):
            wpts = json.loads(wpts_raw)
        else:
            wpts = wpts_raw
        manifest.append({
            "episode_id": ep_id,
            "scene_id": row.get("scene_id", f"log/{ep_id}"),
            "bag_path": bag,
            "waypoints": wpts,
            "start_rotation": json.loads(row["start_rotation"]) if "start_rotation" in row else None,
            "image_topic": args.image_topic or "/camera/camera/color/image_raw",
            "odom_topic": args.odom_topic or None,
        })
    return manifest


# ── Mode 3: Isaac Sim export directory ───────────────────────────────────────

def build_manifest_isaac(args) -> List[Dict]:
    if not args.bag_dir:
        sys.exit("--bag-dir required for mode isaac-dir")

    root = Path(args.bag_dir)
    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    manifest = []

    for d in subdirs:
        # Parse episode_id from directory name (episode_000001 or ep_1 etc.)
        m = re.search(r"(\d+)", d.name)
        ep_id = int(m.group(1)) if m else len(manifest) + 1

        pose_file = d / "pose_log.json"
        waypoints = []
        if pose_file.exists():
            poses = json.load(open(pose_file))
            raw_path = [[p.get("x", 0), p.get("y", 0), p.get("z", 0)] for p in poses]
            waypoints = extract_turn_waypoints(raw_path) if raw_path else []

        manifest.append({
            "episode_id": ep_id,
            "scene_id": f"isaac/{d.name}",
            "bag_path": str(d),
            "waypoints": waypoints,
            "start_rotation": None,
        })

    print(f"Found {len(manifest)} Isaac Sim episodes in {root}")
    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Gate 1b: Build batch manifest for rosbag_extractor")
    p.add_argument("--mode", required=True,
                   choices=["gt-bags", "path-log", "isaac-dir"],
                   help="How to build the manifest")
    p.add_argument("--gt-path", help="GT R2R JSON.gz (mode gt-bags)")
    p.add_argument("--bag-dir", help="Directory containing bag files or Isaac exports")
    p.add_argument("--bag-ext", default=".db3", help="Bag file extension (default: .db3)")
    p.add_argument("--path-log", help="CSV or JSON path log (mode path-log)")
    p.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    p.add_argument("--odom-topic", default=None)
    p.add_argument("--output", required=True, help="Output manifest JSON path")
    args = p.parse_args()

    if args.mode == "gt-bags":
        manifest = build_manifest_gt_bags(args)
    elif args.mode == "path-log":
        manifest = build_manifest_path_log(args)
    elif args.mode == "isaac-dir":
        manifest = build_manifest_isaac(args)
    else:
        sys.exit(f"Unknown mode: {args.mode}")

    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest)} entries → {args.output}")
    if manifest:
        print(f"Sample entry:\n{json.dumps(manifest[0], indent=2)}")


if __name__ == "__main__":
    main()
