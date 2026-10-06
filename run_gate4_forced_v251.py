#!/usr/bin/env python3
"""
Gate 4 v251 — Goal-Anchored + Forced Turn Coverage + Correct Goal Frame.

v250 had a systematic bug: the midpoint-based goal frame selector always picked
waypoint N-2 instead of the rendered frame AT the actual goal waypoint (N-1).
Audit found 100% of trajectories were affected.

v251 fixes this by reading poses.json to find the rendered frame with the
highest waypoint_idx (which is the actual goal waypoint frame).

All other v250 innovations retained:
1. GOAL PRE-PASS: goal frame → 5-10 word description → injected as hard stop constraint
2. MANDATORY TURN CONSTRAINT: all N turns must appear in instruction
3. DYNAMIC WORD LIMIT: max_words = max(45, total_dist * 4.5)
   → short 5m path: 45 words;  long 16m path: 72 words.
   Prevents Gemma from truncating for long paths.

No few-shot examples (v245/v246 analysis: GT examples suppress ordinals).
Quality scorer picks best of 3 styles (from v247).
"""
import asyncio
import base64
import gzip
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent
GT_UNSEEN    = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz"
GT_SEEN      = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_seen/val_seen.json.gz"
RF_DIR       = ROOT / "outputs" / "rendered_frames"
RF_SEEN_DIR  = ROOT / "outputs" / "rendered_frames_val_seen"
MID_DIR      = Path("/mnt/nvme0/vln_habitat/habitat_data/midpoint_frames")
MID_SEEN_DIR = Path("/mnt/nvme0/vln_habitat/habitat_data/midpoint_frames_val_seen")
OUT_DIR      = ROOT / "outputs" / "datasets"
HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")

# ── Gemma API ─────────────────────────────────────────────────────────────────

VLLM_BASE_URL    = "http://10.77.32.231:8000/v1"
VLLM_MODEL       = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
CONCURRENCY      = 4
API_TIMEOUT      = 90
GOAL_DESC_TOKENS = 20
TEMPERATURES     = [0.20, 0.30, 0.25]

# ── Geometry ──────────────────────────────────────────────────────────────────

def dist3d(p1, p2) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def heading_xz(p1, p2) -> float:
    dx, dz = p2[0] - p1[0], p2[2] - p1[2]
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, -dz))

def signed_diff(a, b) -> float:
    return (b - a + 180) % 360 - 180

STAIR_DY_THRESH   = 0.25
STAIR_STEP_HEIGHT = 0.18
ORDINALS = ["first", "second", "third", "fourth", "fifth"]


def parse_odometry(ep: dict) -> dict:
    path = ep["reference_path"]
    total_dist = sum(dist3d(path[i], path[i + 1]) for i in range(len(path) - 1))
    waypoints = []
    cum_dist = 0.0
    total_ascent = 0.0
    total_descent = 0.0
    stair_segments = []

    for i, pos in enumerate(path):
        wp = {"idx": i, "pos": pos[:3], "cum_dist": round(cum_dist, 2), "height": round(pos[1], 2)}
        if i < len(path) - 1:
            d = dist3d(pos, path[i + 1])
            h_out = heading_xz(pos, path[i + 1])
            wp["dist_to_next"] = round(d, 2)
            wp["heading_out"] = round(h_out, 1)
            dy = path[i + 1][1] - pos[1]
            wp["dy"] = round(dy, 3)
            if dy > 0:
                total_ascent += dy
            else:
                total_descent += abs(dy)
            if abs(dy) >= STAIR_DY_THRESH:
                n_steps = max(1, round(abs(dy) / STAIR_STEP_HEIGHT))
                stair_segments.append({
                    "cum_dist": round(cum_dist, 2),
                    "dy": round(dy, 2),
                    "direction": "up" if dy > 0 else "down",
                    "steps": n_steps,
                })
        if i > 0 and i < len(path) - 1:
            h_in  = heading_xz(path[i - 1], path[i])
            h_out2 = heading_xz(path[i], path[i + 1])
            turn = signed_diff(h_in, h_out2)
            if abs(turn) > 20:
                wp["turn_deg"] = round(turn, 1)
                wp["turn_dir"] = "left" if turn < 0 else "right"
        if i > 0:
            cum_dist += dist3d(path[i - 1], pos)
        waypoints.append(wp)

    turns = [wp for wp in waypoints if "turn_dir" in wp]
    turn_parts = []
    for k, t in enumerate(turns):
        ordinal = ORDINALS[k] if k < len(ORDINALS) else f"turn-{k+1}"
        turn_parts.append(f"{ordinal} turn: {t['turn_dir']} {abs(t['turn_deg']):.0f}° at {t['cum_dist']}m")
    turn_summary = ", ".join(turn_parts) if turn_parts else "no major turns"

    return {
        "total_dist": round(total_dist, 1),
        "n_waypoints": len(path),
        "n_turns": len(turns),
        "turns": turns,
        "turn_summary": turn_summary,
        "waypoints": waypoints,
        "total_ascent": round(total_ascent, 2),
        "total_descent": round(total_descent, 2),
        "stair_segments": stair_segments,
        "is_multifloor": len(stair_segments) > 0,
    }


def build_odom_context(odom: dict) -> str:
    n_turns = odom["n_turns"]
    lines = [
        f"Total path: {odom['total_dist']}m, {n_turns} turn(s) total",
        f"Turn sequence ({n_turns} total): {odom['turn_summary']}",
    ]
    if odom["is_multifloor"]:
        up = odom["total_ascent"]
        dn = odom["total_descent"]
        segs = odom["stair_segments"]
        up_segs  = [s for s in segs if s["direction"] == "up"]
        dn_segs  = [s for s in segs if s["direction"] == "down"]
        v_summary = []
        if up > 0.1:
            v_summary.append(f"ascend {up:.1f}m (~{sum(s['steps'] for s in up_segs)} steps up)")
        if dn > 0.1:
            v_summary.append(f"descend {dn:.1f}m (~{sum(s['steps'] for s in dn_segs)} steps down)")
        lines.append(f"Vertical profile: {', '.join(v_summary)}")
        for s in segs:
            lines.append(f"  ↕ Stair at {s['cum_dist']}m: go {s['direction']} ~{s['steps']} steps ({s['dy']:+.2f}m)")
    else:
        lines.append("Vertical profile: flat (no stairs)")
    lines.append("Step sequence:")
    turn_counter = 0
    for wp in odom["waypoints"]:
        i = wp["idx"]
        if "turn_dir" in wp:
            ordinal = ORDINALS[turn_counter] if turn_counter < len(ORDINALS) else f"turn-{turn_counter+1}"
            lines.append(f"  • Waypoint {i} ({wp['cum_dist']}m): {ordinal} turn {wp['turn_dir'].upper()} {abs(wp['turn_deg']):.0f}°, "
                         f"then walk {wp.get('dist_to_next', 0):.1f}m")
            turn_counter += 1
        elif i == 0:
            lines.append(f"  • Start: walk {wp.get('dist_to_next', 0):.1f}m straight")
        elif i == odom["n_waypoints"] - 1:
            lines.append(f"  • Goal (waypoint {i}, {wp['cum_dist']}m): STOP HERE")
    return "\n".join(lines)


def build_mandatory_turns_text(odom: dict) -> str:
    if odom["n_turns"] == 0:
        return "No major turns — walk straight to the goal."
    parts = []
    for k, t in enumerate(odom["turns"]):
        ordinal = ORDINALS[k] if k < len(ORDINALS) else f"turn-{k+1}"
        parts.append(f"{k+1}. {ordinal} turn {t['turn_dir'].upper()} at {t['cum_dist']}m")
    return "\n".join(parts)


def dynamic_word_limit(total_dist: float) -> Tuple[int, int]:
    """Returns (min_words, max_words) scaled to path length."""
    min_w = 10
    max_w = max(45, int(total_dist * 4.5))
    return min_w, min(max_w, 100)


# ── Frame selection ────────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class FrameSelector:
    def __init__(self, rf_dir: Path, mid_dir: Path):
        self.rf_dir = rf_dir
        self.mid_dir = mid_dir

    def _goal_frame_from_poses(self, eid: int) -> Optional[str]:
        """Return the rendered frame at the highest waypoint_idx (actual goal frame)."""
        ep_str = f"episode_{eid:06d}"
        poses_file = self.rf_dir / ep_str / "poses.json"
        if not poses_file.exists():
            return None
        try:
            with open(poses_file) as f:
                poses = json.load(f)
            best_idx = -1
            best_frame = None
            for i, fr in enumerate(poses.get("frames", [])):
                wp = fr.get("waypoint_idx", -1)
                cand = self.rf_dir / ep_str / f"frame_{i:04d}_rgb.jpg"
                if wp > best_idx and cand.exists():
                    best_idx = wp
                    best_frame = str(cand)
            return best_frame
        except Exception:
            return None

    def get_frames(self, eid: int, odom: dict) -> List[Dict]:
        ep_str = f"episode_{eid:06d}"
        rf  = self.rf_dir / ep_str
        mid = self.mid_dir / ep_str
        frames = []

        start = rf / "frame_0000_rgb.jpg"
        if start.exists():
            frames.append({"path": str(start), "label": "Start view (waypoint 0)"})

        turns = odom["turns"]
        if turns:
            t1 = turns[0]
            for cand in [mid / "apv_turn_1_rgb.jpg", mid / "ts_turn_1_rgb.jpg", rf / "frame_0001_rgb.jpg"]:
                if cand.exists():
                    frames.append({"path": str(cand), "label": f"Approaching FIRST turn ({t1['turn_dir']} at ~{t1['cum_dist']}m)"}); break
        if len(turns) >= 2 and len(frames) < 3:
            t2 = turns[1]
            for cand in [mid / "apv_turn_2_rgb.jpg", mid / "ts_turn_2_rgb.jpg"]:
                if cand.exists():
                    frames.append({"path": str(cand), "label": f"Approaching SECOND turn ({t2['turn_dir']} at ~{t2['cum_dist']}m)"}); break

        # Use poses.json to find the rendered frame AT the actual goal waypoint.
        # This is strictly more accurate than heuristic midpoint-based selection.
        used_paths = {f["path"] for f in frames}
        goal_path = self._goal_frame_from_poses(eid)
        if goal_path and goal_path not in used_paths:
            frames.append({"path": goal_path, "label": "Goal area view — STOP HERE"})
        elif not goal_path:
            # Fallback: use frame_0002 (often maps to goal in shorter paths)
            n_wpts = odom["n_waypoints"]
            for i in range(min(4, n_wpts - 1), 0, -1):
                cand = rf / f"frame_{i:04d}_rgb.jpg"
                if cand.exists() and str(cand) not in used_paths:
                    frames.append({"path": str(cand), "label": "Goal area view — STOP HERE"}); break
        return frames[:4]

    def get_goal_frame(self, eid: int, odom: dict) -> Optional[str]:
        # Prefer poses.json-based goal frame (at actual goal waypoint)
        goal = self._goal_frame_from_poses(eid)
        if goal:
            return goal
        all_frames = self.get_frames(eid, odom)
        for f in reversed(all_frames):
            if "Goal" in f["label"]:
                return f["path"]
        return all_frames[-1]["path"] if all_frames else None


# ── Step 1: Goal description ──────────────────────────────────────────────────

GOAL_DESC_PROMPT = """\
Look at this image showing the DESTINATION of a navigation path.
Describe in 5-10 words what is visible at the stopping location.
Focus on the most distinctive object or area feature (e.g., "wooden door on the left", "bottom of stairs near window").
Reply with ONLY the short description, no other words."""


async def get_goal_description(client: AsyncOpenAI, goal_b64: str) -> str:
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{goal_b64}", "detail": "low"}},
        {"type": "text", "text": GOAL_DESC_PROMPT},
    ]
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=GOAL_DESC_TOKENS,
                temperature=0.1,
            ),
            timeout=30,
        )
        return resp.choices[0].message.content.strip().rstrip(".")
    except Exception:
        return "the goal area"


# ── Style prompts ─────────────────────────────────────────────────────────────

STYLE_A_TEMPLATE = """\
You are writing a navigation instruction for a VLN robot (action-centric style).

GOAL: The robot must stop at "{goal_desc}". Your instruction MUST end there.

ODOMETRY (turn numbers are EXACT):
{odom_context}

MANDATORY TURNS — your instruction MUST include ALL {n_turns} turn(s) in order:
{mandatory_turns}

VISUAL CONTEXT:
{image_labels}

Write a crisp ACTION-CENTRIC instruction ({min_words}-{max_words} words):
- Start with a movement command ("Walk forward...", "Turn left...", "Head toward...")
- Use ORDINAL markers: "through the FIRST door", "at the SECOND corridor", "take the THIRD left"
- Anchor each turn to a visible landmark seen in the images
- If odometry shows stairs (↕ Stair), explicitly name the floor transition
- End with EXACTLY: stop when you reach "{goal_desc}"

Write ONLY the instruction:"""

STYLE_B_TEMPLATE = """\
You are writing a navigation instruction for a VLN robot (landmark-descriptive style).

GOAL: The robot must stop at "{goal_desc}". Your instruction MUST end there.

ODOMETRY:
{odom_context}

MANDATORY TURNS — describe ALL {n_turns} turn(s) in your instruction:
{mandatory_turns}

VISUAL CONTEXT:
{image_labels}

Write a LANDMARK-DESCRIPTIVE instruction ({min_words}-{max_words} words):
- Describe what the robot SEES at each key decision point
- Use ORDINAL markers for repeated features: "first door", "second hallway on the left"
- At branch points, name the DISTINCTIVE feature; say which to AVOID if similar options exist
- If odometry shows stairs (↕ Stair), describe the staircase and direction
- Close with: stop when you see "{goal_desc}"

Write ONLY the instruction:"""

STYLE_C_TEMPLATE = """\
You are writing a navigation instruction for a VLN robot (concise-spatial style).

GOAL: stop at "{goal_desc}".

ODOMETRY:
{odom_context}

MANDATORY TURNS (include ALL {n_turns}):
{mandatory_turns}

VISUAL CONTEXT:
{image_labels}

Write a CONCISE instruction ({min_words}-{max_words} words):
- Pack spatial commands: ordinal + feature + direction
- Use ordinals: "2nd door right", "left staircase", "far corridor"
- If stairs present, include: "up the stairs" / "down the left staircase"
- End with: stop at "{goal_desc}"

Write ONLY the instruction:"""

STYLE_TEMPLATES = [STYLE_A_TEMPLATE, STYLE_B_TEMPLATE, STYLE_C_TEMPLATE]
STYLE_NAMES     = ["action_centric", "landmark_descriptive", "concise_spatial"]


def build_messages(frames: List[Dict], odom_context: str, mandatory_turns: str,
                   n_turns: int, style_idx: int, goal_desc: str,
                   min_words: int, max_words: int) -> list:
    image_labels = "\n".join(f"  [{i+1}] {f['label']}" for i, f in enumerate(frames))
    prompt = STYLE_TEMPLATES[style_idx].format(
        goal_desc=goal_desc,
        odom_context=odom_context,
        mandatory_turns=mandatory_turns,
        n_turns=n_turns,
        image_labels=image_labels,
        min_words=min_words,
        max_words=max_words,
    )
    content = []
    for i, frame in enumerate(frames):
        content.append({"type": "text", "text": f"[Image {i+1}: {frame['label']}]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame['b64']}", "detail": "low"},
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


# ── Quality scorer (adapted from v247) ─────────────────────────────────────────

def score_instruction(text: str, odom: dict) -> float:
    t = text.lower()
    score = 0.0
    words = text.split()
    n_words = len(words)

    # Ordinal markers (critical for turn disambiguation)
    ordinal_words = {"first", "second", "third", "fourth", "1st", "2nd", "3rd", "4th"}
    has_ordinal = any(w in ordinal_words for w in t.split())
    if odom["n_turns"] > 1 and has_ordinal:
        score += 2.5

    # Turn direction coverage: count left/right mentions vs expected turns
    n_left = t.count(" left")
    n_right = t.count(" right")
    expected_turns = odom["n_turns"]
    covered = min(n_left + n_right, expected_turns)
    score += 1.5 * covered  # reward each covered turn

    # Stair coverage
    stair_words = {"stair", "stairs", "step", "steps", "floor", "ascend", "descend", "climb", "up the", "down the"}
    has_stair = any(sw in t for sw in stair_words)
    if odom["is_multifloor"]:
        score += 3.0 if has_stair else -3.0
    elif has_stair:
        score -= 1.0  # hallucinated stairs

    # Stop condition specificity
    stop_words = {"stop", "wait", "halt", "stand", "arrive", "pause"}
    if any(w in t.split() for w in stop_words):
        score += 2.0
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    last_sent = sentences[-1].lower() if sentences else ""
    visual_nouns = {"door", "table", "chair", "window", "counter", "couch", "sofa", "wall",
                    "stairs", "staircase", "bathroom", "bedroom", "kitchen", "hallway",
                    "fireplace", "mirror", "bench", "cabinet", "dresser", "shelf", "porch",
                    "bookcase", "plant", "rug", "carpet", "tile", "wood", "glass", "pillar"}
    if any(n in last_sent for n in visual_nouns):
        score += 2.0

    # Goal description anchoring (reward mentioning the injected goal phrase)
    # This is hard to check without knowing goal_desc, so skip here

    # Length (dynamic: prefer instructions close to expected_len)
    expected_len = max(15, int(odom["total_dist"] * 3.5))
    length_diff = abs(n_words - expected_len)
    if length_diff < 8:
        score += 1.0
    elif n_words < 10 or n_words > 90:
        score -= 2.0

    return score


# ── Async generation ──────────────────────────────────────────────────────────

async def generate_instructions(
    client: AsyncOpenAI,
    traj_id: str,
    encoded_frames: List[Dict],
    goal_b64: Optional[str],
    odom: dict,
    fallback: str,
) -> Tuple[str, List[str]]:
    if not encoded_frames:
        return traj_id, [fallback, fallback, fallback]

    odom_ctx        = build_odom_context(odom)
    mandatory_turns = build_mandatory_turns_text(odom)
    min_w, max_w    = dynamic_word_limit(odom["total_dist"])
    max_tokens_val  = max(130, int(max_w * 1.8))

    # Step 1: goal description
    goal_desc = await get_goal_description(client, goal_b64) if goal_b64 else "the goal area"

    # Step 2: 3 style instructions in parallel
    tasks = []
    for style_idx in range(3):
        msgs = build_messages(
            encoded_frames, odom_ctx, mandatory_turns,
            odom["n_turns"], style_idx, goal_desc, min_w, max_w,
        )
        tasks.append(asyncio.wait_for(
            client.chat.completions.create(
                model=VLLM_MODEL, messages=msgs,
                max_tokens=max_tokens_val,
                temperature=TEMPERATURES[style_idx],
            ),
            timeout=API_TIMEOUT,
        ))

    results = []
    style_responses = await asyncio.gather(*tasks, return_exceptions=True)
    for style_idx, resp in enumerate(style_responses):
        if isinstance(resp, Exception):
            print(f"  [WARN] {traj_id} style {STYLE_NAMES[style_idx]}: {type(resp).__name__}: {resp}", flush=True)
            results.append(fallback)
        else:
            text = resp.choices[0].message.content.strip()
            if not any(w in text.lower() for w in ["stop", "wait", "halt", "stand", "arrive", "destination", "end"]):
                text += f" Stop at {goal_desc}."
            results.append(text)

    # Quality scorer: pick best of 3 styles
    scored = [(score_instruction(r, odom), i, r) for i, r in enumerate(results)]
    scored.sort(key=lambda x: -x[0])
    best = scored[0][2]
    second = scored[1][2]
    third = scored[2][2]
    return traj_id, [best, second, third]


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_dataset(gt_data: dict, instruction_map: Dict[str, List[str]]) -> dict:
    episodes = []
    traj_annotation_counter: Dict[str, int] = {}
    for ep in gt_data["episodes"]:
        traj_id = f"traj_{ep['trajectory_id']}" if "trajectory_id" in ep else f"traj_ep{ep['episode_id']}"
        idx = traj_annotation_counter.get(traj_id, 0)
        traj_annotation_counter[traj_id] = idx + 1
        if traj_id in instruction_map:
            instrs = instruction_map[traj_id]
            instr_text = instrs[idx % len(instrs)]
        else:
            instr_text = ep["instruction"]["instruction_text"]
        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        new_ep["instruction"]["instruction_text"] = instr_text
        episodes.append(new_ep)
    return {"episodes": episodes}


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async(split: str = "unseen"):
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY", timeout=120.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if split == "unseen":
        gt_path  = GT_UNSEEN
        rf_dir   = RF_DIR
        mid_dir  = MID_DIR
        out_name = "val_unseen_v251.json.gz"
        ck_path  = ROOT / "outputs" / "gate4_v251_unseen_checkpoint.json"
    elif split == "seen":
        gt_path  = GT_SEEN
        rf_dir   = RF_SEEN_DIR
        mid_dir  = MID_SEEN_DIR
        out_name = "val_seen_v251.json.gz"
        ck_path  = ROOT / "outputs" / "gate4_v251_seen_checkpoint.json"
    else:
        raise ValueError(f"Unknown split: {split}")

    with gzip.open(gt_path, "rt") as f:
        gt_data = json.load(f)

    traj_to_first_ep: Dict[str, dict] = {}
    for ep in gt_data["episodes"]:
        traj_id = f"traj_{ep['trajectory_id']}" if "trajectory_id" in ep else f"traj_ep{ep['episode_id']}"
        if traj_id not in traj_to_first_ep:
            traj_to_first_ep[traj_id] = ep

    n_traj = len(traj_to_first_ep)
    print(f"v251 | Split: {split} | {len(gt_data['episodes'])} eps | {n_traj} unique trajectories")

    instructions: Dict[str, List[str]] = {}
    if ck_path.exists():
        with open(ck_path) as f:
            raw = json.load(f)
            for k, v in raw.items():
                instructions[k] = v if isinstance(v, list) else [v, v, v]
        print(f"Resumed from checkpoint: {len(instructions)}/{n_traj}")

    selector = FrameSelector(rf_dir, mid_dir)
    pending = [(tid, ep) for tid, ep in traj_to_first_ep.items() if tid not in instructions]
    print(f"Remaining: {len(pending)} trajectories to generate", flush=True)

    print("Pre-encoding frames...", flush=True)
    pre_encoded = []
    for tid, ep in pending:
        odom   = parse_odometry(ep)
        frames = selector.get_frames(ep["episode_id"], odom)
        fallback = f"Walk {odom['total_dist']}m with {odom['n_turns']} turn(s). {odom['turn_summary']}. Stop at goal."
        encoded_frames = []
        for fr in frames:
            try:
                encoded_frames.append({**fr, "b64": encode_image(fr["path"])})
            except Exception:
                pass
        goal_path = selector.get_goal_frame(ep["episode_id"], odom)
        goal_b64 = None
        if goal_path:
            try:
                goal_b64 = encode_image(goal_path)
            except Exception:
                pass
        pre_encoded.append((tid, encoded_frames, goal_b64, odom, fallback))
    print(f"Pre-encoded {len(pre_encoded)} trajectories.", flush=True)

    t0 = time.time()
    done = 0

    for batch_start in range(0, len(pre_encoded), CONCURRENCY):
        batch = pre_encoded[batch_start:batch_start + CONCURRENCY]
        batch_results = await asyncio.gather(*[
            generate_instructions(client, tid, enc_frames, goal_b64, odom, fallback)
            for tid, enc_frames, goal_b64, odom, fallback in batch
        ], return_exceptions=True)
        for r in batch_results:
            if isinstance(r, Exception):
                print(f"  [ERROR] batch item failed: {r}", flush=True)
                continue
            traj_id, instrs = r
            instructions[traj_id] = instrs
            done += 1
            if done % 50 == 0 or done <= 5:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remain = (len(pre_encoded) - done) / rate if rate > 0 else 0
                a, b, c = instrs
                print(f"  [{done}/{len(pre_encoded)}] {traj_id}", flush=True)
                print(f"    best: {a[:100]}", flush=True)
                print(f"    2nd:  {b[:80]}", flush=True)
                print(f"    Rate: {rate:.2f} traj/s | ETA: {remain/60:.1f}m", flush=True)
        if done % 100 < CONCURRENCY:
            with open(ck_path, "w") as f:
                json.dump(instructions, f, ensure_ascii=False)
            print(f"  [ckpt] Saved {len(instructions)} trajectories", flush=True)

    with open(ck_path, "w") as f:
        json.dump(instructions, f, ensure_ascii=False)
    print(f"Generation complete: {len(instructions)}/{n_traj} trajectories")

    # Build dataset with instruction_vocab
    dataset = build_dataset(gt_data, instructions)

    # Inject instruction_vocab from v238
    vocab_src = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v238.json.gz"
    with gzip.open(vocab_src, "rt") as f:
        dataset["instruction_vocab"] = json.load(f)["instruction_vocab"]

    out_path = HABITAT_BASE / ("val_unseen" if split == "unseen" else "val_seen") / out_name
    tmp_path = str(out_path) + ".tmp"
    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)
    import shutil as _shutil
    _shutil.move(tmp_path, str(out_path))

    eps = dataset["episodes"]
    unique = len(set(e["instruction"]["instruction_text"] for e in eps))
    words = [len(e["instruction"]["instruction_text"].split()) for e in eps]
    print(f"\nDataset saved: {out_path}")
    print(f"  {len(eps)} eps, {unique} unique ({unique/len(eps)*100:.1f}%)")
    print(f"  Avg words: {sum(words)/len(words):.1f}, range: {min(words)}-{max(words)}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="unseen", choices=["unseen", "seen"])
    args = p.parse_args()
    asyncio.run(main_async(args.split))


if __name__ == "__main__":
    main()
