#!/usr/bin/env python3
"""
Gate 2: Path Analyzer
Converts 3D waypoints + start_rotation into human-readable motion primitives.
Inputs:  reference_path (list of [x,y,z]), start_rotation ([qx,qy,qz,qw])
Outputs: list of motion primitives for instruction generation
"""
import math
from typing import List, Dict, Any


def quaternion_yaw_deg(qxyzw: List[float]) -> float:
    """Extract compass yaw in degrees from [qx, qy, qz, qw] (Habitat Y-up, -Z forward).

    Convention: 0°=North(-Z), +90°=East(+X), -90°=West(-X) — matches heading_between_xz.
    The raw atan2 formula gives the opposite sign (positive=West) because Habitat Y-up
    TurnLeft rotates counter-clockwise. We negate to match compass convention.
    """
    qx, qy, qz, qw = qxyzw
    siny_cosp = 2 * (qw * qy + qz * qx)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return -math.degrees(math.atan2(siny_cosp, cosy_cosp))  # negate: raw gives West=+90, we want East=+90


def heading_between_xz(p1: List[float], p2: List[float]) -> float:
    """Compass heading (degrees) from p1 to p2 projected onto XZ plane."""
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2]
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, -dz))


def signed_angle_diff(from_deg: float, to_deg: float) -> float:
    """Signed angular difference from_deg → to_deg in (-180, 180]."""
    return (to_deg - from_deg + 180) % 360 - 180


def dist3d(p1: List[float], p2: List[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def elevation_change(p1: List[float], p2: List[float]) -> float:
    """Y-axis elevation change (positive = up)."""
    return p2[1] - p1[1]


def classify_turn(angle_deg: float) -> str:
    """Classify turn angle into direction label."""
    if angle_deg <= -30:
        return "left"
    elif angle_deg >= 30:
        return "right"
    else:
        return "straight"


def analyze_path(
    reference_path: List[List[float]],
    start_rotation: List[float] = None,
    turn_threshold_deg: float = 30.0,
    min_segment_dist_m: float = 0.3,
) -> Dict[str, Any]:
    """
    Analyze a reference path and return structured motion primitives.

    Returns:
        {
          "primitives": [
            {"type": "straight", "distance_m": 2.3},
            {"type": "left_turn", "angle_deg": 87.0, "sharp": False},
            {"type": "right_turn", "angle_deg": 45.0, "sharp": False},
            {"type": "elevation", "direction": "up", "change_m": 0.5},
            {"type": "stop"}
          ],
          "summary": {
            "total_distance_m": 7.8,
            "n_left_turns": 1,
            "n_right_turns": 0,
            "n_waypoints": 6,
            "elevation_change_m": 0.0,
            "path_type": "mostly_straight" | "winding" | "single_turn"
          },
          "key_frame_indices": [0, 3, 5],   # indices into reference_path
          "segment_headings": [87.2, 92.1, ...],  # heading at each waypoint
        }
    """
    if len(reference_path) < 2:
        return {"primitives": [{"type": "stop"}], "summary": {}, "key_frame_indices": [0], "segment_headings": []}

    n = len(reference_path)
    headings = []
    for i in range(n - 1):
        headings.append(heading_between_xz(reference_path[i], reference_path[i + 1]))
    headings.append(headings[-1])  # repeat last for stop frame

    primitives = []
    key_frame_indices = [0]  # always include start

    # v213: Detect initial turn from start_rotation vs first segment heading.
    # Use CORRECT compass convention (negate raw quaternion yaw — see quaternion_yaw_deg).
    # Threshold 30°: GT annotators mention turns for ~60°+ misalignment; below 30° the
    # model corrects visually. 30° catches genuine directional errors without false positives
    # on episodes where annotators just said "walk forward" despite small angular offset.
    initial_turn = None
    if start_rotation is not None:
        agent_yaw = quaternion_yaw_deg(start_rotation)
        init_angle = signed_angle_diff(agent_yaw, headings[0])
        if abs(init_angle) >= 30.0:
            direction = "left" if init_angle < 0 else "right"
            initial_turn = {
                "direction": direction,
                "angle_deg": round(abs(init_angle), 1),
                "sharp": abs(init_angle) > 90,
                "is_around": abs(init_angle) >= 150,
            }

    current_seg_dist = 0.0
    current_seg_start = 0

    for i in range(n - 1):
        seg_dist = dist3d(reference_path[i], reference_path[i + 1])
        elev = elevation_change(reference_path[i], reference_path[i + 1])
        current_seg_dist += seg_dist

        is_last = (i == n - 2)

        # Check for significant elevation change
        if abs(elev) > 0.3:
            direction = "up" if elev > 0 else "down"
            primitives.append({"type": "elevation", "direction": direction, "change_m": round(abs(elev), 2)})

        # Check for turn at next waypoint (look ahead)
        if not is_last:
            turn_angle = signed_angle_diff(headings[i], headings[i + 1])
            if abs(turn_angle) >= turn_threshold_deg:
                # Emit accumulated straight segment
                if current_seg_dist > min_segment_dist_m:
                    primitives.append({"type": "straight", "distance_m": round(current_seg_dist, 2)})
                current_seg_dist = 0.0
                # Emit turn
                direction = "left" if turn_angle < 0 else "right"
                sharp = abs(turn_angle) > 75
                primitives.append({
                    "type": f"{direction}_turn",
                    "angle_deg": round(abs(turn_angle), 1),
                    "sharp": sharp,
                })
                key_frame_indices.append(i + 1)  # frame at turn decision point

    # Final straight segment + stop
    if current_seg_dist > min_segment_dist_m:
        primitives.append({"type": "straight", "distance_m": round(current_seg_dist, 2)})
    key_frame_indices.append(max(0, n - 2))  # goal approach frame
    key_frame_indices.append(n - 1)          # stop frame
    primitives.append({"type": "stop"})

    # Summary statistics
    total_dist = sum(dist3d(reference_path[i], reference_path[i + 1]) for i in range(n - 1))
    n_left = sum(1 for p in primitives if p["type"] == "left_turn")
    n_right = sum(1 for p in primitives if p["type"] == "right_turn")
    total_elev = reference_path[-1][1] - reference_path[0][1]

    if n_left + n_right == 0:
        path_type = "mostly_straight"
    elif n_left + n_right == 1:
        path_type = "single_turn"
    else:
        path_type = "winding"

    return {
        "primitives": primitives,
        "initial_turn": initial_turn,  # v212: initial orientation turn, None if aligned
        "summary": {
            "total_distance_m": round(total_dist, 2),
            "n_left_turns": n_left,
            "n_right_turns": n_right,
            "n_waypoints": n,
            "elevation_change_m": round(total_elev, 3),
            "path_type": path_type,
        },
        "key_frame_indices": sorted(set(key_frame_indices)),
        "segment_headings": [round(h, 1) for h in headings],
    }


def primitives_to_text(primitives: List[Dict]) -> str:
    """
    Convert motion primitives to a compact text description for VLM prompting.
    E.g.: 'go straight 2.3m → turn left 87° → go straight 1.5m → STOP'
    """
    parts = []
    for p in primitives:
        t = p["type"]
        if t == "straight":
            parts.append(f"straight {p['distance_m']}m")
        elif t in ("left_turn", "right_turn"):
            direction = "left" if t == "left_turn" else "right"
            sharp = " (sharp)" if p.get("sharp") else ""
            parts.append(f"turn {direction} ~{p['angle_deg']:.0f}°{sharp}")
        elif t == "elevation":
            parts.append(f"go {p['direction']} (Δ{p['change_m']}m)")
        elif t == "stop":
            parts.append("STOP")
    return " → ".join(parts)


if __name__ == "__main__":
    import gzip, json

    GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
    with gzip.open(GT_PATH, "rt") as f:
        data = json.load(f)

    print("Analyzing first 5 episodes:\n")
    for ep in data["episodes"][:5]:
        result = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        text = primitives_to_text(result["primitives"])
        print(f"Episode {ep['episode_id']}:")
        print(f"  GT instruction: {ep['instruction']['instruction_text'].strip()}")
        print(f"  Path analysis:  {text}")
        print(f"  Summary: {result['summary']}")
        print(f"  Key frames: {result['key_frame_indices']}")
        print()
