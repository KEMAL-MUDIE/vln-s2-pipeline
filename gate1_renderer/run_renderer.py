#!/usr/bin/env python3
"""
Gate 1: Batch Frame Renderer for S2 Pipeline.

Renders key-frame RGB images along reference_paths using Habitat-Sim.
Only renders key frames (start + turns + goal) to minimize disk usage:
  1839 episodes × ~4 key frames × ~300KB/JPEG ≈ 2.2 GB total

Output structure:
  outputs/rendered_frames/
    episode_000001/
      frame_0000_rgb.jpg   (start frame)
      frame_0001_rgb.jpg   (turn frame)
      ...
      poses.json

Usage (inside vlnav/habitat-eval:rebuilt container):
  python3 gate1_renderer/run_renderer.py \
      --gt-path data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz \
      --scenes-root data/InternData-N1/scene_data/mp3d_ce \
      --output-dir /workspace/s2_pipeline/outputs/rendered_frames \
      --n-episodes 50
"""
import argparse
import gzip
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


# ── Render config ──────────────────────────────────────────────────────────────
CAMERA_CFG = {
    "width": 640,
    "height": 480,
    "hfov": 90.0,
    "sensor_height": 1.25,   # meters above floor (matches Habitat R2R eval)
}
JPEG_QUALITY = 88            # good quality / small size balance
TURN_THRESHOLD_DEG = 25.0    # degrees — headings changed more than this = key turn frame


# ── Geometry helpers ───────────────────────────────────────────────────────────

def heading_xz(p1: List[float], p2: List[float]) -> float:
    """Compass heading (deg, Y-up) from p1 to p2 in XZ plane."""
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2]
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, -dz))


def angle_diff(a: float, b: float) -> float:
    """Signed difference between two angles (degrees), wrapped to [-180, 180]."""
    d = (b - a + 180.0) % 360.0 - 180.0
    return d


def heading_to_quat(heading_deg: float) -> List[float]:
    """Y-axis rotation quaternion [qx, qy, qz, qw] from heading degrees."""
    half = math.radians(heading_deg) / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def select_key_frames(path: List[List[float]], start_rotation: Optional[List[float]]) -> List[Dict]:
    """
    Choose key frames to render: start, each significant turn, and goal.
    Returns list of dicts with {position, rotation, label}.
    """
    n = len(path)
    frames = []

    # Start frame — use provided start_rotation if available
    rot = start_rotation if start_rotation else [0.0, 0.0, 0.0, 1.0]
    frames.append({"position": path[0], "rotation": rot, "waypoint_idx": 0, "label": "start"})

    prev_heading = None
    for i in range(n - 1):
        h = heading_xz(path[i], path[i + 1])
        rot = heading_to_quat(h)

        # Detect turn relative to previous segment
        if prev_heading is not None:
            delta = abs(angle_diff(prev_heading, h))
            if delta >= TURN_THRESHOLD_DEG:
                # Render AT the turn point (path[i]) facing new direction
                frames.append({
                    "position": path[i],
                    "rotation": rot,
                    "waypoint_idx": i,
                    "label": f"turn_{i}",
                })

        prev_heading = h

    # Goal frame — face inward (reverse of last segment heading)
    if n >= 2:
        last_heading = heading_xz(path[-2], path[-1])
        goal_rot = heading_to_quat(last_heading)
        frames.append({
            "position": path[-1],
            "rotation": goal_rot,
            "waypoint_idx": n - 1,
            "label": "goal",
        })

    # Deduplicate frames at same position (can happen on single-segment paths)
    seen = set()
    unique = []
    for f in frames:
        key = tuple(round(x, 2) for x in f["position"])
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


# ── Habitat-Sim renderer ───────────────────────────────────────────────────────

def make_scene_path(scene_id: str, scenes_root: str) -> str:
    """Resolve scene_id to filesystem path."""
    if scene_id.endswith(".glb"):
        p = os.path.join(scenes_root, scene_id)
        if os.path.exists(p):
            return p
        # Try without mp3d/ prefix
        p2 = os.path.join(scenes_root, scene_id.replace("mp3d/", ""))
        if os.path.exists(p2):
            return p2
    else:
        bn = scene_id.split("/")[-1]
        p = os.path.join(scenes_root, "mp3d", bn, f"{bn}.glb")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Scene not found: {scene_id} under {scenes_root}")


def render_episodes_in_scene(sim, episodes: List[Dict], output_dir: Path) -> tuple:
    """
    Render all episodes for one already-loaded Habitat-Sim scene.
    Returns (done, errors) counts.
    """
    import habitat_sim
    from PIL import Image

    done = 0
    errors = 0

    for episode in episodes:
        ep_id = episode["episode_id"]
        ep_dir = output_dir / f"episode_{ep_id:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already rendered
        poses_file = ep_dir / "poses.json"
        if poses_file.exists():
            with open(poses_file) as f:
                existing = json.load(f)
            if existing.get("n_frames", 0) > 0:
                done += 1
                continue

        key_frames = select_key_frames(episode["reference_path"], episode.get("start_rotation"))
        frame_metadata = []

        try:
            for fi, kf in enumerate(key_frames):
                state = habitat_sim.AgentState()
                state.position = kf["position"]
                state.rotation = kf["rotation"]
                sim.get_agent(0).set_state(state)

                obs = sim.get_sensor_observations()
                rgb_arr = obs["rgb"][:, :, :3]

                img_name = f"frame_{fi:04d}_rgb.jpg"
                img = Image.fromarray(rgb_arr)
                img.save(ep_dir / img_name, "JPEG", quality=JPEG_QUALITY)

                frame_metadata.append({
                    "frame_idx": fi,
                    "path": img_name,
                    "position": kf["position"],
                    "rotation": kf["rotation"],
                    "waypoint_idx": kf["waypoint_idx"],
                    "label": kf["label"],
                })

            result = {
                "episode_id": ep_id,
                "scene_id": episode["scene_id"],
                "n_frames": len(frame_metadata),
                "camera_config": CAMERA_CFG,
                "frames": frame_metadata,
            }
            with open(poses_file, "w") as f:
                json.dump(result, f, indent=2)
            done += 1

        except Exception as e:
            errors += 1
            print(f"    [ERR] ep={ep_id}: {e}", flush=True)

    return done, errors


# ── Multi-scene batched rendering ──────────────────────────────────────────────

def render_batch(episodes: List[Dict], scenes_root: str, output_dir: Path,
                 start: int = 0, end: Optional[int] = None) -> Dict:
    """
    Render a batch of episodes.
    Groups by scene to load Habitat-Sim Simulator ONCE per scene (~30x speedup).
    All 1839 val_unseen episodes are in only 11 scenes.
    """
    try:
        import habitat_sim
    except ImportError as e:
        raise RuntimeError(f"habitat-sim not available: {e}")

    subset = episodes[start:end]
    total = len(subset)

    # Group by scene_id — reuse same simulator for all episodes in a scene
    from collections import defaultdict
    by_scene = defaultdict(list)
    for ep in subset:
        by_scene[ep["scene_id"]].append(ep)

    print(f"Rendering {total} episodes across {len(by_scene)} scenes (sim reuse per scene)")
    print(f"Output: {output_dir}")
    print()

    done = 0
    errors = 0
    t0 = time.time()

    for scene_id, scene_eps in by_scene.items():
        scene_name = scene_id.split("/")[-1].replace(".glb", "")
        print(f"  Loading scene: {scene_name} ({len(scene_eps)} episodes)...", flush=True)

        try:
            scene_path = make_scene_path(scene_id, scenes_root)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            errors += len(scene_eps)
            continue

        # Build simulator config
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.enable_physics = False
        sim_cfg.allow_sliding = False

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [CAMERA_CFG["height"], CAMERA_CFG["width"]]
        rgb_spec.hfov = CAMERA_CFG["hfov"]
        rgb_spec.position = [0.0, CAMERA_CFG["sensor_height"], 0.0]

        agent_cfg = habitat_sim.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec]

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        try:
            sim = habitat_sim.Simulator(cfg)
        except Exception as e:
            print(f"  [ERR] Simulator init failed for {scene_name}: {e}")
            errors += len(scene_eps)
            continue

        try:
            sc_done, sc_err = render_episodes_in_scene(sim, scene_eps, output_dir)
        finally:
            sim.close()

        done += sc_done
        errors += sc_err

        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done}/{total}] scene={scene_name}: {sc_done} ok, {sc_err} err | "
              f"rate={rate:.1f}/s ETA={eta/60:.1f}m", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {done}/{total} rendered, {errors} errors in {elapsed:.1f}s ({done/elapsed:.1f}/s)")
    return {"rendered": done, "errors": errors, "total": total}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Gate 1: Habitat Frame Renderer")
    p.add_argument("--gt-path", required=True,
                   help="Path to val_unseen_patched.json.gz")
    p.add_argument("--scenes-root", required=True,
                   help="Root directory of scene data (mp3d .glb files)")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for rendered frames")
    p.add_argument("--n-episodes", type=int, default=None,
                   help="Max episodes to render (default: all)")
    p.add_argument("--start", type=int, default=0,
                   help="Start episode index")
    p.add_argument("--dry-run", action="store_true",
                   help="Show render plan without rendering")
    return p.parse_args()


def main():
    args = parse_args()

    print("=== Gate 1: Habitat Frame Renderer ===")
    print(f"  GT:         {args.gt_path}")
    print(f"  Scenes:     {args.scenes_root}")
    print(f"  Output:     {args.output_dir}")
    print()

    # Load episodes
    with gzip.open(args.gt_path, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]
    print(f"  {len(episodes)} total episodes")

    # Verify scenes root
    if not os.path.isdir(args.scenes_root):
        print(f"ERROR: scenes_root not found: {args.scenes_root}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    end_idx = args.start + args.n_episodes if args.n_episodes else None

    if args.dry_run:
        # Show render plan stats without rendering
        print("\n=== DRY RUN: Key Frame Plan ===")
        subset = episodes[args.start:end_idx]
        total_frames = 0
        for ep in subset[:10]:
            kfs = select_key_frames(ep["reference_path"], ep.get("start_rotation"))
            total_frames += len(kfs)
            labels = [f["label"] for f in kfs]
            print(f"  ep={ep['episode_id']:6d}  wpts={len(ep['reference_path'])}  "
                  f"key_frames={len(kfs)}  [{', '.join(labels)}]")
        if len(subset) > 10:
            avg = total_frames / min(10, len(subset))
            est_total = int(avg * len(subset))
            est_gb = est_total * 300e3 / 1e9
            print(f"  ... ({len(subset)-10} more)")
            print(f"\n  Avg key frames/ep: {avg:.1f}")
            print(f"  Estimated total:   {est_total} frames ≈ {est_gb:.1f} GB (JPEG@88)")
        return

    # Check habitat_sim is available
    try:
        import habitat_sim
        print(f"  habitat-sim: {habitat_sim.__version__}")
    except ImportError:
        print("ERROR: habitat-sim not installed. Run inside vlnav/habitat-eval:rebuilt container.")
        sys.exit(1)

    # Run render batch
    render_batch(episodes, args.scenes_root, output_dir,
                 start=args.start, end=end_idx)


if __name__ == "__main__":
    main()
