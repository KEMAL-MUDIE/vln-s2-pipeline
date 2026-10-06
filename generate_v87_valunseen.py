#!/usr/bin/env python3
"""
Generate v87: v86 + initial turn handling from start_rotation.

CRITICAL FINDING:
  82.5% of val_unseen episodes require initial turn ≥30° to face the path.
  54.6% require ≥90° turn (genuine left/right turn).
  21.4% require "turn around" (≥150°).

  v62-v86 ALL IGNORE start_rotation → agent walks wrong direction in 82.5% of eps!
  metadata_reproducer has handled this since v212 (which helps explain 36.87% vs our ~27%).

  v87 FIX: Detect initial orientation misalignment from start_rotation + reference_path[0→1].
  Prepend appropriate turn prefix to instruction.

DESIGN:
  Initial turn prefix:
    ≥150°: "Turn around and"
    ≥90°:  "Turn [dir] and"
    ≥45°:  "Turn [dir] and"
    ≥30°:  "Turn slightly [dir] and"
    <30°:  no prefix

  Merged with existing instruction start:
    "Walk forward and turn left near [lm]. → "Turn right and walk forward and turn left near [lm]."
    "Walk past [lm] and turn [dir]..."    → "Turn right and walk past [lm] and turn [dir]..."
    "Walk straight ahead..."              → "Turn around and walk straight ahead..."

  BASE: v86 instructions (best combination: PRE/AT/BETWEEN vc_turns for turn eps,
        vc_turns for 0-turn, through/past based on doorway context, v71 stop phrases)
"""
import gzip
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gate2_path.path_analyzer import quaternion_yaw_deg, heading_between_xz, signed_angle_diff

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
GT_PATH = BASE / "val_unseen" / "val_unseen_patched.json.gz"
V86_PATH = BASE / "val_unseen" / "val_unseen_v86.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v87.json.gz"

INITIAL_TURN_THRESHOLD = 30.0  # degrees — below this, no prefix needed


def get_initial_turn(qxyzw, path, threshold=INITIAL_TURN_THRESHOLD):
    """Compute initial turn from start_rotation to first path segment direction."""
    if not qxyzw or len(path) < 2:
        return None
    agent_yaw = quaternion_yaw_deg(qxyzw)
    first_heading = heading_between_xz(path[0], path[1])
    angle = signed_angle_diff(agent_yaw, first_heading)
    if abs(angle) < threshold:
        return None
    return {
        'dir': 'left' if angle < 0 else 'right',
        'angle': abs(angle),
        'is_around': abs(angle) >= 150,
        'is_sharp': abs(angle) >= 90,
        'is_slight': abs(angle) < 45,
    }


def build_initial_prefix(turn):
    """Generate turn prefix string from initial turn dict."""
    if turn is None:
        return None
    if turn['is_around']:
        return "Turn around and"
    if turn['is_slight']:
        return f"Turn slightly {turn['dir']} and"
    return f"Turn {turn['dir']} and"


def apply_initial_turn(instruction, prefix):
    """Prepend initial turn prefix to instruction, merging with first verb."""
    if not prefix:
        return instruction

    # Pattern: instruction starts with capitalized action verb
    # "Walk forward and..." → "Turn right and walk forward and..."
    # "Walk past the..." → "Turn right and walk past the..."
    # "Walk straight ahead..." → "Turn around and walk straight ahead..."
    # The prefix ends with "and", so we lowercase the first char of instruction

    first_char = instruction[0].lower()
    rest = instruction[1:]
    return f"{prefix} {first_char}{rest}"


def main():
    print("=== v87 generator: v86 + initial turn handling (CRITICAL FIX) ===")
    print("  82.5% of val_unseen episodes need initial turn but v62-v86 all ignored it!")
    print("  This should be the BIGGEST single SR improvement in the chain.")
    print()

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V86_PATH) as f:
        v86_data = json.load(f)

    gt_by_eid = {ep["episode_id"]: ep for ep in gt_data["episodes"]}
    v86_by_eid = {ep["episode_id"]: ep for ep in v86_data["episodes"]}

    stats = {
        'no_turn': 0,
        'slight_turn': 0,
        'turn': 0,
        'sharp_turn': 0,
        'turn_around': 0,
    }

    new_episodes = []
    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        path = ep["reference_path"]
        qxyzw = ep.get("start_rotation")

        v86_ep = v86_by_eid[eid]
        v86_inst = v86_ep["instruction"]["instruction_text"]

        init_turn = get_initial_turn(qxyzw, path)
        prefix = build_initial_prefix(init_turn)

        if prefix is None:
            new_inst = v86_inst
            stats['no_turn'] += 1
        else:
            new_inst = apply_initial_turn(v86_inst, prefix)
            if init_turn['is_around']:
                stats['turn_around'] += 1
            elif init_turn['is_sharp']:
                stats['sharp_turn'] += 1
            elif init_turn['is_slight']:
                stats['slight_turn'] += 1
            else:
                stats['turn'] += 1

        new_ep = dict(v86_ep)
        new_ep["instruction"] = dict(v86_ep["instruction"])
        new_ep["instruction"]["instruction_text"] = new_inst
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    print(f"Initial turn stats:")
    for k, v in stats.items():
        print(f"  {k}: {v} ({v/len(new_episodes)*100:.1f}%)")

    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]
    HALLWAY_RE = re.compile(r'\b(hallway|hall)\b', re.I)
    TURN_RE = re.compile(r'\bturn\b', re.I)

    print(f"\nQuality metrics (n={n}):")
    print(f"  avg_words: {sum(words)/n:.1f}  [v86=20.1, GT=26.8]")
    print(f"  hallway:   {sum(1 for i in all_insts if HALLWAY_RE.search(i))/n*100:.1f}%")
    print(f"  through:   {sum(1 for i in all_insts if 'through' in i.lower())/n*100:.1f}%")
    print(f"  toward:    {sum(1 for i in all_insts if 'toward' in i.lower())/n*100:.1f}%")
    print(f"  turn%:     {sum(1 for i in all_insts if TURN_RE.search(i))/n*100:.1f}%")
    print(f"  unique:    {len(set(all_insts))/n*100:.1f}%")

    print(f"\nSample instructions (with initial turn):")
    shown = 0
    for ep in gt_data["episodes"]:
        if shown >= 8:
            break
        eid = ep["episode_id"]
        init_turn = get_initial_turn(ep.get("start_rotation"), ep["reference_path"])
        if not init_turn:
            continue
        inst = next(e["instruction"]["instruction_text"] for e in new_episodes if e["episode_id"] == eid)
        print(f"  [{init_turn['angle']:.0f}°-{init_turn['dir']}] {inst[:160]}")
        shown += 1

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
