#!/usr/bin/env python3
"""
Gate 1: Scene Renderer
Renders RGB (and optionally depth) frames along a reference_path using Habitat-Sim.
Produces the same visual input that a real robot/rosbag would give.

Input:  episode dict (scene_id, reference_path, start_rotation, ...)
Output: frames/ directory with PNG images + poses.json

This simulates what a real sensor would capture — enabling the pipeline to work
identically on Habitat scenes, Isaac Sim, rosbags, and real video.
"""
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Optional

SCENES_ROOT = "/mnt/nvme0/vln_habitat/habitat_data/scene_datasets"
OUTPUT_DIR = Path("/home/kemal/VLNav/s2_pipeline_new/outputs/rendered_frames")

# Camera config matching Habitat R2R evaluation
CAMERA_CONFIG = {
    "width": 640,
    "height": 480,
    "hfov": 90.0,        # horizontal field of view in degrees
    "sensor_height": 1.25,  # meters above floor
}
FRAMES_PER_SEGMENT = 4   # interpolated frames between waypoints


def scene_path_from_id(scene_id: str) -> str:
    """Convert 'mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb' to full filesystem path."""
    return os.path.join(SCENES_ROOT, scene_id)


def interpolate_poses(p1: List[float], p2: List[float], n: int) -> List[List[float]]:
    """Linearly interpolate n poses between p1 and p2."""
    if n <= 1:
        return [p1]
    poses = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        poses.append([p1[j] + t * (p2[j] - p1[j]) for j in range(3)])
    return poses


def heading_to_quaternion(heading_deg: float) -> List[float]:
    """Convert heading angle (degrees, Y-axis rotation) to [qx, qy, qz, qw]."""
    angle_rad = math.radians(heading_deg / 2)
    return [0.0, math.sin(angle_rad), 0.0, math.cos(angle_rad)]


def heading_between_xz(p1: List[float], p2: List[float]) -> float:
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2]
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, -dz))


def build_render_plan(episode: Dict) -> List[Dict]:
    """
    Build a list of (position, rotation) frames to render for the episode.
    Returns:
        [{"position": [x,y,z], "rotation": [qx,qy,qz,qw], "waypoint_idx": int, "frame_type": str}]
    frame_type: "waypoint" | "interpolated" | "start" | "goal"
    """
    path = episode["reference_path"]
    n = len(path)
    frames = []

    # Start frame (use provided start_rotation)
    frames.append({
        "position": path[0],
        "rotation": episode.get("start_rotation", [0, 0, 0, 1]),
        "waypoint_idx": 0,
        "frame_type": "start",
    })

    for i in range(n - 1):
        heading = heading_between_xz(path[i], path[i + 1])
        rot = heading_to_quaternion(heading)

        # Interpolated frames between waypoints
        interp_positions = interpolate_poses(path[i], path[i + 1], FRAMES_PER_SEGMENT + 1)[1:]
        for j, pos in enumerate(interp_positions[:-1]):
            frames.append({
                "position": pos,
                "rotation": rot,
                "waypoint_idx": i,
                "frame_type": "interpolated",
            })

        # Waypoint frame
        frame_type = "goal" if i + 1 == n - 1 else "waypoint"
        frames.append({
            "position": path[i + 1],
            "rotation": rot,
            "waypoint_idx": i + 1,
            "frame_type": frame_type,
        })

    return frames


def render_episode_habitat(episode: Dict, output_dir: Path, save_depth: bool = False) -> Dict:
    """
    Render frames for one episode using Habitat-Sim.
    Returns metadata dict with frame paths + poses.

    Requires: habitat-sim installed (run inside vlnav/habitat-eval:rebuilt container).
    """
    try:
        import habitat_sim
        import numpy as np
        from PIL import Image
    except ImportError:
        raise RuntimeError("habitat-sim not installed. Run inside vlnav/habitat-eval:rebuilt container.")

    scene_path = scene_path_from_id(episode["scene_id"])
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene not found: {scene_path}")

    ep_id = episode["episode_id"]
    ep_dir = output_dir / f"episode_{ep_id:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    # Configure Habitat-Sim
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False
    sim_cfg.allow_sliding = False

    # RGB sensor
    sensors = []
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [CAMERA_CONFIG["height"], CAMERA_CONFIG["width"]]
    rgb_spec.hfov = CAMERA_CONFIG["hfov"]
    rgb_spec.position = [0.0, CAMERA_CONFIG["sensor_height"], 0.0]
    sensors.append(rgb_spec)

    if save_depth:
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [CAMERA_CONFIG["height"], CAMERA_CONFIG["width"]]
        depth_spec.hfov = CAMERA_CONFIG["hfov"]
        depth_spec.position = [0.0, CAMERA_CONFIG["sensor_height"], 0.0]
        sensors.append(depth_spec)

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    render_plan = build_render_plan(episode)
    frame_metadata = []

    for frame_idx, frame_info in enumerate(render_plan):
        # Set agent state
        state = habitat_sim.AgentState()
        state.position = frame_info["position"]
        state.rotation = frame_info["rotation"]  # [qx,qy,qz,qw]
        sim.get_agent(0).set_state(state)

        # Render
        obs = sim.get_sensor_observations()

        # Save RGB
        rgb_path = ep_dir / f"frame_{frame_idx:04d}_rgb.png"
        rgb_img = Image.fromarray(obs["rgb"][:, :, :3])
        rgb_img.save(rgb_path)

        frame_meta = {
            "frame_idx": frame_idx,
            "rgb_path": str(rgb_path.relative_to(output_dir)),
            "position": frame_info["position"],
            "rotation": frame_info["rotation"],
            "waypoint_idx": frame_info["waypoint_idx"],
            "frame_type": frame_info["frame_type"],
        }

        if save_depth:
            depth_path = ep_dir / f"frame_{frame_idx:04d}_depth.npy"
            import numpy as np
            np.save(depth_path, obs["depth"])
            frame_meta["depth_path"] = str(depth_path.relative_to(output_dir))

        frame_metadata.append(frame_meta)

    sim.close()

    # Save poses
    poses_path = ep_dir / "poses.json"
    result = {
        "episode_id": ep_id,
        "scene_id": episode["scene_id"],
        "n_frames": len(frame_metadata),
        "camera_config": CAMERA_CONFIG,
        "frames": frame_metadata,
    }
    with open(poses_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def render_episode_dry_run(episode: Dict) -> Dict:
    """
    Dry run: generate render plan without actually rendering.
    Useful for testing the plan structure before running Habitat.
    """
    render_plan = build_render_plan(episode)
    print(f"Episode {episode['episode_id']}: {len(render_plan)} frames to render")
    print(f"  Waypoints: {len(episode['reference_path'])}")
    for i, f in enumerate(render_plan[:5]):
        print(f"  Frame {i:03d}: type={f['frame_type']}, wpt={f['waypoint_idx']}, pos={[round(x,2) for x in f['position']]}")
    if len(render_plan) > 5:
        print(f"  ... ({len(render_plan) - 5} more frames)")
    return {"episode_id": episode["episode_id"], "n_frames": len(render_plan), "plan": render_plan}


if __name__ == "__main__":
    import gzip

    GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
    with gzip.open(GT_PATH, "rt") as f:
        data = json.load(f)

    print("=== Render Plan Dry Run (first 3 episodes) ===\n")
    for ep in data["episodes"][:3]:
        render_episode_dry_run(ep)
        print()

    print("To actually render frames (requires Habitat-Sim), run inside container:")
    print("  docker exec -it vlnav_habitat_eval_n1_s2_gt bash")
    print("  python3 /workspace/s2_pipeline_new/gate1_renderer/renderer.py --render")
