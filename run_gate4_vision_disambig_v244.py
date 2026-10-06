#!/usr/bin/env python3
"""
Gate 4 Vision+Odom v244 — Ordinal disambiguation + decision-point emphasis.

Key improvement over v243: improved prompt engineering that targets the most
common VLN failure mode — selecting the WRONG door/hallway/staircase at a
branch point. Three improvements:

1. ORDINAL MARKERS: prompts explicitly instruct Gemma to use "first/second/third"
   for repeated features (doors, corridors, turns). E.g., "enter the second doorway
   on the right" instead of "enter the doorway on the right".

2. NEGATIVE DISAMBIGUATION: prompts ask Gemma to call out false paths ("do NOT
   enter the bathroom on your left — continue to the kitchen").

3. TURN COUNT ANCHORING: the odometry context now includes explicit turn count
   ("This path has N turns total: turn 1 of N at Xm, ...") so Gemma knows when
   the robot is at a branching point vs. a dead-end.

All other aspects retained from v243 (height-aware stairs, 3 styles, asyncio
batch processing, CONCURRENCY=4, API_TIMEOUT=90).
"""
import asyncio
import base64
import gzip
import json
import math
import os
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
META_DIR     = ROOT / "outputs" / "v244_metadata"
OUT_DIR      = ROOT / "outputs" / "datasets"

HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")

# ── Gemma API ─────────────────────────────────────────────────────────────────

VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL    = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
CONCURRENCY   = 4    # trajectories in-flight at once; 3 calls each = 12 concurrent API calls
API_TIMEOUT   = 90   # seconds per API call before retry-as-fallback
MAX_TOKENS    = 140  # slightly higher to allow ordinal markers without truncation
TEMPERATURES  = [0.20, 0.30, 0.25]  # A, B, C styles

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

STAIR_DY_THRESH = 0.25   # meters of height change per segment to count as stairs
STAIR_STEP_HEIGHT = 0.18  # standard stair riser height in meters

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
            h_in = heading_xz(path[i - 1], path[i])
            h_out2 = heading_xz(path[i], path[i + 1])
            turn = signed_diff(h_in, h_out2)
            if abs(turn) > 20:
                wp["turn_deg"] = round(turn, 1)
                wp["turn_dir"] = "left" if turn < 0 else "right"
        if i > 0:
            cum_dist += dist3d(path[i - 1], pos)
        waypoints.append(wp)

    turns = [wp for wp in waypoints if "turn_dir" in wp]
    # Build turn summary with ordinal markers for disambiguation
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

    # Vertical profile
    if odom["is_multifloor"]:
        up = odom["total_ascent"]
        dn = odom["total_descent"]
        segs = odom["stair_segments"]
        up_segs = [s for s in segs if s["direction"] == "up"]
        dn_segs = [s for s in segs if s["direction"] == "down"]
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


# ── Frame selection (same as v241-v243) ───────────────────────────────────────

class FrameSelector:
    def __init__(self, rf_dir: Path, mid_dir: Path):
        self.rf_dir = rf_dir
        self.mid_dir = mid_dir

    def get_ep_str(self, eid: int) -> str:
        return f"episode_{eid:06d}"

    def get_frames(self, eid: int, odom: dict) -> List[Dict]:
        ep_str = self.get_ep_str(eid)
        rf = self.rf_dir / ep_str
        mid = self.mid_dir / ep_str
        frames = []

        start = rf / "frame_0000_rgb.jpg"
        if start.exists():
            frames.append({"path": str(start), "label": "Start view (waypoint 0)", "wpt_idx": 0})

        turns = odom["turns"]
        if turns:
            t1 = turns[0]
            apv = mid / "apv_turn_1_rgb.jpg"
            if apv.exists():
                frames.append({"path": str(apv), "label": f"Approaching FIRST turn ({t1['turn_dir']} at ~{t1['cum_dist']}m)", "wpt_idx": t1["idx"]})
            else:
                ts = mid / "ts_turn_1_rgb.jpg"
                if ts.exists():
                    frames.append({"path": str(ts), "label": f"AT first turn ({t1['turn_dir']}, wp {t1['idx']})", "wpt_idx": t1["idx"]})
                else:
                    f1 = rf / "frame_0001_rgb.jpg"
                    if f1.exists():
                        frames.append({"path": str(f1), "label": "Mid-path view", "wpt_idx": 1})

        if len(turns) >= 2 and len(frames) < 3:
            apv2 = mid / "apv_turn_2_rgb.jpg"
            ts2  = mid / "ts_turn_2_rgb.jpg"
            t2   = turns[1]
            if apv2.exists():
                frames.append({"path": str(apv2), "label": f"Approaching SECOND turn ({t2['turn_dir']} at ~{t2['cum_dist']}m)", "wpt_idx": t2["idx"]})
            elif ts2.exists():
                frames.append({"path": str(ts2), "label": f"AT second turn ({t2['turn_dir']}, wp {t2['idx']})", "wpt_idx": t2["idx"]})

        n_wpts = odom["n_waypoints"]
        goal_candidates = [rf / f"frame_{min(4,n_wpts-1):04d}_rgb.jpg", rf / "frame_0002_rgb.jpg"]
        for i in range(4, 0, -1):
            goal_candidates.append(mid / f"mid_000{i}_rgb.jpg")
        goal_candidates.append(rf / "frame_0001_rgb.jpg")
        for gc in goal_candidates:
            if gc.exists() and str(gc) not in {f["path"] for f in frames}:
                frames.append({"path": str(gc), "label": "Goal area view — STOP HERE", "wpt_idx": n_wpts - 1})
                break

        return frames[:4]


# ── 3-style prompts (v244: ordinal disambiguation focus) ──────────────────────

STYLE_A_PROMPT = """\
You are writing a navigation instruction for a VLN robot (action-centric style).

ODOMETRY (turn numbers are EXACT — use them as ordinals in your instruction):
{odom_context}

VISUAL CONTEXT:
{image_labels}

Write a crisp ACTION-CENTRIC instruction (1-3 sentences, 12-45 words):
- Start immediately with a movement command ("Walk forward...", "Turn left...", "Head toward...")
- Use ORDINAL markers for repeated features: "through the FIRST door", "at the SECOND corridor", "take the stairs on the LEFT side"
- If images show multiple similar options at a turn point, name the SPECIFIC one to take
- If odometry shows stairs (↕ Stair), explicitly name the floor transition ("go down the stairs", "climb the staircase")
- End with a concrete stop condition referencing what is visible in the goal frame

Example: "Walk through the first doorway and turn right at the stone column. Pass the second hallway junction and stop at the window ledge."

Write ONLY the instruction:"""

STYLE_B_PROMPT = """\
You are writing a navigation instruction for a VLN robot (landmark-descriptive style).

ODOMETRY (turn numbers are EXACT — refer to them to distinguish similar features):
{odom_context}

VISUAL CONTEXT:
{image_labels}

Write a LANDMARK-DESCRIPTIVE instruction (1-3 sentences, 15-50 words):
- Describe what the robot SEES at each key decision point ("You'll see...", "Ahead is...", "Notice the...")
- At branch points (doors, corridors, staircases), name the DISTINCTIVE feature that identifies the correct path
- If there are similar-looking options, say which to AVOID: "Continue past the bathroom — do not enter it"
- If odometry shows stairs (↕ Stair), describe the staircase visually and specify direction
- Close with a clear visual description of the stopping location

Example: "Ahead you'll see two doorways — enter the one on the left with the wooden frame. Continue past the bathroom door and stop at the carpeted alcove at the end."

Write ONLY the instruction:"""

STYLE_C_PROMPT = """\
You are writing a navigation instruction for a VLN robot (concise-spatial style).

ODOMETRY (turn count is exact — include ordinals for clarity):
{odom_context}

VISUAL CONTEXT:
{image_labels}

Write a CONCISE, HIGH-DENSITY instruction (1-2 short sentences, 8-28 words):
- Pack spatial commands tightly: ordinal + feature + direction
- Use ordinals when needed to prevent ambiguity: "2nd door right", "left staircase", "far corridor"
- If odometry shows stairs (↕ Stair), include briefly: "up the stairs", "down the left staircase"
- Every word must carry navigation meaning; no filler

Example (disambiguation): "First right door, straight past bathroom, left staircase down. Stop at the landing rug."
Example (simple): "Past the tiled counter, right at the wooden beam. Stop by the glass partition."

Write ONLY the instruction:"""

STYLE_PROMPTS = [STYLE_A_PROMPT, STYLE_B_PROMPT, STYLE_C_PROMPT]
STYLE_NAMES   = ["action_centric", "landmark_descriptive", "concise_spatial"]


# ── Async generation ──────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_messages(frames: List[Dict], odom_context: str, style_idx: int) -> list:
    image_labels = "\n".join(f"  [{i+1}] {f['label']}" for i, f in enumerate(frames))
    prompt = STYLE_PROMPTS[style_idx].format(
        odom_context=odom_context,
        image_labels=image_labels,
    )
    content = []
    for i, frame in enumerate(frames):
        content.append({
            "type": "text",
            "text": f"[Image {i+1}: {frame['label']}]",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame['b64']}",
                "detail": "low",
            },
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


async def generate_three_instructions(
    client: AsyncOpenAI,
    traj_id: str,
    encoded_frames: List[Dict],
    odom_ctx: str,
    fallback: str,
) -> Tuple[str, List[str]]:
    if not encoded_frames:
        return traj_id, [fallback, fallback, fallback]

    results = []
    for style_idx in range(3):
        msgs = build_messages(encoded_frames, odom_ctx, style_idx)
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=VLLM_MODEL,
                    messages=msgs,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURES[style_idx],
                ),
                timeout=API_TIMEOUT,
            )
            text = resp.choices[0].message.content.strip()
            if not any(w in text.lower() for w in ["stop", "wait", "halt", "stand", "arrive", "destination", "end"]):
                text += " Stop here."
            results.append(text)
        except (asyncio.TimeoutError, Exception) as e:
            print(f"  [WARN] {traj_id} style {STYLE_NAMES[style_idx]}: {type(e).__name__}: {e}", flush=True)
            results.append(fallback)

    return traj_id, results


# ── Dataset builder ───────────────────────────────────────────────────────────

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
    client  = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY", timeout=120.0)
    META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if split == "unseen":
        gt_path  = GT_UNSEEN
        rf_dir   = RF_DIR
        mid_dir  = MID_DIR
        out_name = "val_unseen_v244.json.gz"
        ck_path  = ROOT / "outputs" / "gate4_v244_unseen_checkpoint.json"
    elif split == "seen":
        gt_path  = GT_SEEN
        rf_dir   = RF_SEEN_DIR
        mid_dir  = MID_SEEN_DIR
        out_name = "val_seen_v244.json.gz"
        ck_path  = ROOT / "outputs" / "gate4_v244_seen_checkpoint.json"
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
    print(f"Split: {split} | {len(gt_data['episodes'])} episodes | {n_traj} unique trajectories")
    print(f"Generating 3 instruction styles per trajectory = {n_traj * 3} API calls")

    instructions: Dict[str, List[str]] = {}
    if ck_path.exists():
        with open(ck_path) as f:
            raw = json.load(f)
            for k, v in raw.items():
                instructions[k] = v if isinstance(v, list) else [v, v, v]
        print(f"Resumed from checkpoint: {len(instructions)}/{n_traj} trajectories")

    selector = FrameSelector(rf_dir, mid_dir)

    pending = [(tid, ep) for tid, ep in traj_to_first_ep.items() if tid not in instructions]
    print(f"Remaining: {len(pending)} trajectories to generate", flush=True)

    print("Pre-encoding frames...", flush=True)
    pre_encoded: List[Tuple] = []
    for tid, ep in pending:
        odom = parse_odometry(ep)
        odom_ctx = build_odom_context(odom)
        frames = selector.get_frames(ep["episode_id"], odom)
        fallback = f"Walk {odom['total_dist']}m. {odom['turn_summary']}. Stop at goal."
        encoded_frames = []
        for f in frames:
            try:
                encoded_frames.append({**f, "b64": encode_image(f["path"])})
            except Exception:
                pass
        pre_encoded.append((tid, encoded_frames, odom_ctx, fallback))
    print(f"Pre-encoded {len(pre_encoded)} trajectories.", flush=True)

    t0 = time.time()
    done = 0

    for batch_start in range(0, len(pre_encoded), CONCURRENCY):
        batch = pre_encoded[batch_start:batch_start + CONCURRENCY]
        batch_results = await asyncio.gather(*[
            generate_three_instructions(client, tid, enc_frames, odom_ctx, fallback)
            for tid, enc_frames, odom_ctx, fallback in batch
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
                rate = done / elapsed
                remain = (len(pre_encoded) - done) / rate if rate > 0 else 0
                a, b, c = instrs
                print(f"  [{done}/{len(pre_encoded)}] {traj_id}", flush=True)
                print(f"    A: {a[:70]}", flush=True)
                print(f"    B: {b[:70]}", flush=True)
                print(f"    C: {c[:70]}", flush=True)
                print(f"    Rate: {rate:.2f} traj/s | ETA: {remain/60:.1f}m", flush=True)
        if done % 100 < CONCURRENCY:
            with open(ck_path, "w") as f:
                json.dump(instructions, f, ensure_ascii=False)
            print(f"  [checkpoint] {done}/{len(pre_encoded)} saved", flush=True)

    with open(ck_path, "w") as f:
        json.dump(instructions, f, ensure_ascii=False)
    print(f"Checkpoint saved: {ck_path}", flush=True)

    out_data = build_dataset(gt_data, instructions)
    out_path = OUT_DIR / out_name
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)

    habitat_dir = HABITAT_BASE / f"val_{split}"
    habitat_out = habitat_dir / out_name
    shutil.copy2(out_path, habitat_out)
    print(f"Saved: {out_path}")
    print(f"Copied: {habitat_out}")

    ep_texts = [ep["instruction"]["instruction_text"] for ep in out_data["episodes"]]
    unique = len(set(ep_texts))
    avg_words = sum(len(t.split()) for t in ep_texts) / len(ep_texts)
    print(f"\nDiversity: {unique}/{len(ep_texts)} unique ({unique/len(ep_texts)*100:.0f}%) | avg_words={avg_words:.1f}")

    # Check ordinal usage
    ordinal_count = sum(1 for t in ep_texts if any(o in t.lower() for o in ["first", "second", "third", "1st", "2nd"]))
    print(f"Ordinal markers: {ordinal_count}/{len(ep_texts)} instructions ({ordinal_count/len(ep_texts)*100:.1f}%)")

    print("\nStyle diversity sample (trajectory 'traj_15' or first available):")
    sample_tid = "traj_15" if "traj_15" in instructions else list(instructions.keys())[0]
    for i, (style, instr) in enumerate(zip(STYLE_NAMES, instructions[sample_tid])):
        print(f"  {style[:1].upper()}: {instr[:90]}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="unseen", choices=["unseen", "seen", "both"])
    args = p.parse_args()

    if args.split == "both":
        for split in ["unseen", "seen"]:
            asyncio.run(main_async(split))
    else:
        asyncio.run(main_async(args.split))

if __name__ == "__main__":
    main()
