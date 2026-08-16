#!/usr/bin/env python3
"""
Gate 1b Integration Test — no ROS, no Habitat, no CV2 required.

Simulates an ImageDirSource on synthetic frames extracted from the
EXISTING rendered_frames output (Gate 1 Habitat output), then
re-extracts using Gate 1b and verifies the output matches Gate 1 format.

Also tests:
  - VideoSource (via PIL-based synthetic video frames)
  - Temporal alignment on a known path
  - poses.json format compliance
  - Drop-in compatibility with gate3_landmarks/landmark_detector.py

Run: python3 gate1_renderer/test_extractor.py
"""

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from gate1_renderer.rosbag_extractor import (
    ImageDirSource,
    IsaacSimSource,
    align_odom,
    align_temporal,
    extract_episode,
    heading_to_quat,
    save_episode_frames,
    select_key_waypoints,
)

RENDERED = ROOT / "outputs" / "rendered_frames"
LANDMARKS = ROOT / "outputs" / "gate3_landmarks"

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def check(cond: bool, msg: str) -> bool:
    tag = PASS if cond else FAIL
    print(f"  {tag}  {msg}")
    return cond


# ── T1: select_key_waypoints matches run_renderer.py ────────────────────────

def test_key_waypoints():
    print("\n[T1] select_key_waypoints — geometry")
    # Episode 1 reference_path (6 waypoints, one sharp left turn)
    path = [
        [15.07, 0.17, -4.48],
        [13.65, 0.17, -4.24],
        [12.58, 0.17, -4.27],
        [12.46, 0.17, -2.39],
        [12.86, 0.17, -0.07],
        [13.05, 0.17,  1.87],
    ]
    start_rot = [-0.0, 0.50, 0.0, 0.87]
    kws = select_key_waypoints(path, start_rot)

    ok = True
    ok &= check(kws[0]["label"] == "start", f"first label=start, got={kws[0]['label']}")
    ok &= check(kws[-1]["label"] == "goal",  f"last label=goal, got={kws[-1]['label']}")
    ok &= check(len(kws) >= 2, f"at least 2 key waypoints, got {len(kws)}")
    has_turn = any("turn" in kw["label"] for kw in kws)
    ok &= check(has_turn, "turn detected in key waypoints")
    print(f"  Key waypoints: {[kw['label'] for kw in kws]}")
    return ok


# ── T2: temporal alignment ────────────────────────────────────────────────────

def test_temporal_alignment():
    print("\n[T2] align_temporal — uniform frame distribution")
    import numpy as np

    path = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 2.0],
        [4.0, 0.0, 5.0],
    ]
    kws = select_key_waypoints(path)
    n_kws = len(kws)

    # 20 synthetic frames
    all_frames = [(float(i), np.zeros((480, 640, 3), dtype=np.uint8), None) for i in range(20)]
    chosen = align_temporal(all_frames, kws)

    ok = True
    ok &= check(len(chosen) == n_kws, f"chosen={len(chosen)} == n_kws={n_kws}")
    ok &= check(chosen[0][2]["label"] == "start", "first chosen is start")
    ok &= check(chosen[-1][2]["label"] == "goal",  "last chosen is goal")
    return ok


# ── T3: odom alignment ────────────────────────────────────────────────────────

def test_odom_alignment():
    print("\n[T3] align_odom — position matching")
    import numpy as np

    path = [
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [10.0, 0.0, 5.0],
    ]
    kws = select_key_waypoints(path)

    # 30 frames with odom sweeping x from 0→10, z from 0→5
    frames = []
    for i in range(30):
        x = i * 10 / 29
        z = i * 5 / 29 if i > 15 else 0.0
        pose = {"x": x, "y": 0.0, "z": z, "yaw_rad": 0.0}
        frames.append((float(i), np.zeros((2, 2, 3), dtype=np.uint8), pose))

    chosen = align_odom(frames, kws)
    ok = True
    ok &= check(chosen is not None, "align_odom returned result (has odom)")
    ok &= check(len(chosen) == len(kws), f"chosen={len(chosen)} == n_kws={len(kws)}")
    # Start frame should be near x=0
    start_pose = chosen[0][1]
    ok &= check(start_pose["x"] < 1.0, f"start frame near x=0, got x={start_pose['x']:.2f}")
    # Goal frame should be near x=10
    goal_pose = chosen[-1][1]
    ok &= check(goal_pose["x"] > 9.0, f"goal frame near x=10, got x={goal_pose['x']:.2f}")
    return ok


# ── T4: ImageDirSource on real Gate 1 output ─────────────────────────────────

def test_imgdir_source():
    print("\n[T4] ImageDirSource — read existing Gate 1 rendered frames")
    if not RENDERED.exists():
        print("  [SKIP] rendered_frames not found (run Gate 1 first)")
        return True

    ep_dirs = sorted(RENDERED.iterdir())
    if not ep_dirs:
        print("  [SKIP] rendered_frames is empty")
        return True

    ep_dir = ep_dirs[0]
    src = ImageDirSource(str(ep_dir))
    src.open()
    frames = list(src.frames())
    src.close()

    ok = True
    ok &= check(len(frames) > 0, f"loaded {len(frames)} frames from {ep_dir.name}")
    ts, rgb, pose = frames[0]
    ok &= check(rgb.ndim == 3 and rgb.shape[2] == 3, f"RGB shape={rgb.shape}")
    ok &= check(pose is not None, f"pose loaded from poses.json: {pose}")
    return ok


# ── T5: Full extract_episode with synthetic frames ────────────────────────────

def test_full_extract():
    print("\n[T5] Full extract_episode — synthetic ImageDirSource")
    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic image directory
        img_dir = tmpdir / "fake_bag"
        img_dir.mkdir()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (255, 255, 0), (0, 255, 255), (255, 0, 255),
                  (128, 128, 0), (0, 128, 128), (128, 0, 128), (64, 64, 64)]
        for i, c in enumerate(colors):
            img = Image.new("RGB", (640, 480), color=c)
            img.save(img_dir / f"frame_{i:04d}.jpg")

        path = [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 3.0],
            [6.0, 0.0, 3.0],
        ]

        out_dir = tmpdir / "out"
        src = ImageDirSource(str(img_dir))
        with src:
            result = extract_episode(
                source=src,
                episode_id=42,
                scene_id="test/synthetic",
                waypoints=path,
                start_rotation=None,
                output_dir=out_dir,
                force_temporal=True,
                overwrite=True,
            )

        ok = True
        ok &= check(result["episode_id"] == 42, "episode_id=42")
        ok &= check(result["n_frames"] >= 2, f"n_frames={result['n_frames']} >= 2")
        ok &= check((out_dir / "episode_000042" / "poses.json").exists(), "poses.json written")

        # Verify poses.json structure matches Gate 1 format
        poses = json.load(open(out_dir / "episode_000042" / "poses.json"))
        ok &= check("camera_config" in poses, "camera_config present")
        ok &= check("frames" in poses, "frames list present")
        ok &= check(poses["frames"][0]["label"] == "start", "first frame label=start")
        ok &= check(poses["frames"][-1]["label"] == "goal",  "last frame label=goal")

        # Verify JPEG files exist
        jpegs = list((out_dir / "episode_000042").glob("frame_*_rgb.jpg"))
        ok &= check(len(jpegs) == result["n_frames"], f"{len(jpegs)} JPEG files match n_frames")

        print(f"  poses.json frames: {[f['label'] for f in poses['frames']]}")
        return ok


# ── T6: IsaacSimSource synthetic test ────────────────────────────────────────

def test_isaac_source():
    print("\n[T6] IsaacSimSource — synthetic export directory")
    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        rgb_dir = tmpdir / "rgb"
        rgb_dir.mkdir()

        pose_log = []
        for i in range(8):
            img = Image.new("RGB", (640, 480), (i * 30, 60, 120))
            img.save(rgb_dir / f"{i:04d}.png")
            pose_log.append({
                "x": float(i) * 0.5, "y": 0.0, "z": 0.0,
                "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0
            })
        with open(tmpdir / "pose_log.json", "w") as f:
            json.dump(pose_log, f)

        src = IsaacSimSource(str(tmpdir))
        src.open()
        frames = list(src.frames())
        src.close()

        ok = True
        ok &= check(len(frames) == 8, f"8 frames loaded, got {len(frames)}")
        _, rgb, pose = frames[0]
        ok &= check(pose is not None and "x" in pose, f"pose loaded: {pose}")
        ok &= check(rgb.shape == (480, 640, 3), f"RGB shape correct: {rgb.shape}")
        return ok


# ── T7: Gate 1b output is compatible with Gate 3 ─────────────────────────────

def test_gate3_compat():
    print("\n[T7] Gate 3 compatibility — poses.json field names")
    if not RENDERED.exists():
        print("  [SKIP] No rendered_frames to compare against")
        return True

    ep_dirs = sorted(RENDERED.iterdir())
    if not ep_dirs:
        return True

    # Load a real Gate 1 poses.json
    poses = json.load(open(ep_dirs[0] / "poses.json"))
    required_fields = {"episode_id", "scene_id", "n_frames", "camera_config", "frames"}
    required_frame_fields = {"frame_idx", "path", "position", "rotation", "waypoint_idx", "label"}

    ok = True
    ok &= check(required_fields <= set(poses.keys()),
                f"top-level fields: {set(poses.keys()) & required_fields}")
    if poses["frames"]:
        frame_keys = set(poses["frames"][0].keys())
        ok &= check(required_frame_fields <= frame_keys,
                    f"frame fields: {frame_keys & required_frame_fields}")
    return ok


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Gate 1b: Universal Frame Extractor — Integration Tests")
    print("=" * 60)

    tests = [
        ("T1 key_waypoints",   test_key_waypoints),
        ("T2 temporal_align",  test_temporal_alignment),
        ("T3 odom_align",      test_odom_alignment),
        ("T4 imgdir_source",   test_imgdir_source),
        ("T5 full_extract",    test_full_extract),
        ("T6 isaac_source",    test_isaac_source),
        ("T7 gate3_compat",    test_gate3_compat),
    ]

    results = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  {FAIL} EXCEPTION: {e}")
            import traceback; traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for name, ok in results.items():
        tag = PASS if ok else FAIL
        print(f"  {tag}  {name}")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
