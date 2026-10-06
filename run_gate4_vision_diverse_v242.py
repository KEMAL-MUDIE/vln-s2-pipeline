#!/usr/bin/env python3
"""
Gate 4 Vision+Odom v242 — Diverse 3-annotation instruction generator.

Key improvement over v241: generates 3 genuinely different instruction styles
per trajectory and assigns them across the 3 GT annotations.

This addresses the core limitation of v241: all 3 annotations per trajectory
got the SAME instruction, while R2R GT annotations are genuinely diverse.

3 instruction styles:
  Style A (action-centric): movement commands + turn anchors, crisp imperative
  Style B (landmark-led): visual scene description leads each segment, "when you see..."
  Style C (concise-spatial): terse, high-density directions matching short GT annotations

All 3 styles still use full odometry context + up to 4 rendered frames.
Token budget: ~1500-1600 per call, 3 calls per trajectory, 1839 total calls.
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
META_DIR     = ROOT / "outputs" / "v242_metadata"
OUT_DIR      = ROOT / "outputs" / "datasets"

HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")

# ── Gemma API ─────────────────────────────────────────────────────────────────

VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL    = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
CONCURRENCY   = 4    # trajectories in-flight at once; 3 calls each = 12 concurrent API calls
API_TIMEOUT   = 90   # seconds per API call before retry-as-fallback
MAX_TOKENS    = 130
TEMPERATURES  = [0.20, 0.30, 0.25]  # A, B, C styles

# ── Geometry (same as v241) ───────────────────────────────────────────────────

def dist3d(p1, p2) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def heading_xz(p1, p2) -> float:
    dx, dz = p2[0] - p1[0], p2[2] - p1[2]
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, -dz))

def signed_diff(a, b) -> float:
    return (b - a + 180) % 360 - 180

def parse_odometry(ep: dict) -> dict:
    path = ep["reference_path"]
    total_dist = sum(dist3d(path[i], path[i + 1]) for i in range(len(path) - 1))
    waypoints = []
    cum_dist = 0.0
    for i, pos in enumerate(path):
        wp = {"idx": i, "pos": pos[:3], "cum_dist": round(cum_dist, 2)}
        if i < len(path) - 1:
            d = dist3d(pos, path[i + 1])
            h_out = heading_xz(pos, path[i + 1])
            wp["dist_to_next"] = round(d, 2)
            wp["heading_out"] = round(h_out, 1)
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
    turn_summary = ", ".join(
        f"{t['turn_dir']} {abs(t['turn_deg']):.0f}° at {t['cum_dist']}m" for t in turns
    ) if turns else "no major turns"
    return {
        "total_dist": round(total_dist, 1),
        "n_waypoints": len(path),
        "turns": turns,
        "turn_summary": turn_summary,
        "waypoints": waypoints,
    }


def build_odom_context(odom: dict) -> str:
    lines = [f"Total path: {odom['total_dist']}m, {len(odom['turns'])} turn(s): {odom['turn_summary']}"]
    lines.append("Step sequence:")
    for wp in odom["waypoints"]:
        i = wp["idx"]
        if "turn_dir" in wp:
            lines.append(f"  • Waypoint {i} ({wp['cum_dist']}m): "
                         f"turn {wp['turn_dir'].upper()} {abs(wp['turn_deg']):.0f}°, "
                         f"then walk {wp.get('dist_to_next', 0):.1f}m")
        elif i == 0:
            lines.append(f"  • Start: walk {wp.get('dist_to_next', 0):.1f}m straight")
        elif i == odom["n_waypoints"] - 1:
            lines.append(f"  • Goal (waypoint {i}, {wp['cum_dist']}m): STOP HERE")
    return "\n".join(lines)


# ── Frame selection (same as v241) ────────────────────────────────────────────

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
                frames.append({"path": str(apv), "label": f"View approaching {t1['turn_dir']} turn (~{t1['cum_dist']}m)", "wpt_idx": t1["idx"]})
            else:
                ts = mid / "ts_turn_1_rgb.jpg"
                if ts.exists():
                    frames.append({"path": str(ts), "label": f"View at {t1['turn_dir']} turn (wp {t1['idx']})", "wpt_idx": t1["idx"]})
                else:
                    f1 = rf / "frame_0001_rgb.jpg"
                    if f1.exists():
                        frames.append({"path": str(f1), "label": "Mid-path view", "wpt_idx": 1})

        if len(turns) >= 2 and len(frames) < 3:
            apv2 = mid / "apv_turn_2_rgb.jpg"
            ts2  = mid / "ts_turn_2_rgb.jpg"
            t2   = turns[1]
            if apv2.exists():
                frames.append({"path": str(apv2), "label": f"View approaching {t2['turn_dir']} turn (~{t2['cum_dist']}m)", "wpt_idx": t2["idx"]})
            elif ts2.exists():
                frames.append({"path": str(ts2), "label": f"View at {t2['turn_dir']} turn (wp {t2['idx']})", "wpt_idx": t2["idx"]})

        n_wpts = odom["n_waypoints"]
        goal_candidates = [rf / f"frame_{min(4,n_wpts-1):04d}_rgb.jpg", rf / "frame_0002_rgb.jpg"]
        for i in range(4, 0, -1):
            goal_candidates.append(mid / f"mid_000{i}_rgb.jpg")
        goal_candidates.append(rf / "frame_0001_rgb.jpg")
        for gc in goal_candidates:
            if gc.exists() and str(gc) not in {f["path"] for f in frames}:
                frames.append({"path": str(gc), "label": "Goal area view (stop here)", "wpt_idx": n_wpts - 1})
                break

        return frames[:4]


# ── 3-style prompts ───────────────────────────────────────────────────────────

STYLE_A_PROMPT = """\
You are writing a navigation instruction for a VLN robot (action-centric style).

ODOMETRY:
{odom_context}

VISUAL CONTEXT:
{image_labels}

Write a crisp ACTION-CENTRIC instruction (1-3 sentences, 12-45 words):
- Start immediately with a movement command ("Walk forward...", "Turn left...", "Head toward...")
- Anchor each turn to a SPECIFIC visual landmark from the images (named objects, colors, materials)
- End with a clear stop action referencing the goal image

Example style: "Walk forward through the wooden double doors and turn right at the stone column. Continue past the leather sofa and stop at the far end of the hallway."

Write ONLY the instruction:"""

STYLE_B_PROMPT = """\
You are writing a navigation instruction for a VLN robot (landmark-descriptive style).

ODOMETRY:
{odom_context}

VISUAL CONTEXT:
{image_labels}

Write a LANDMARK-DESCRIPTIVE instruction (1-3 sentences, 15-50 words):
- Describe what the robot SEES at each key point ("You'll see...", "Ahead is...", "Notice the...")
- Use visual cues from the images to signal turns ("When the wooden door is on your right...")
- Close with what marks the destination in the goal image

Example style: "Ahead you'll see a marble-tiled corridor with glass panels on the left. When you reach the wooden staircase on your right, turn and continue to the window-lit alcove at the end."

Write ONLY the instruction:"""

STYLE_C_PROMPT = """\
You are writing a navigation instruction for a VLN robot (concise-spatial style).

ODOMETRY:
{odom_context}

VISUAL CONTEXT:
{image_labels}

Write a CONCISE, HIGH-DENSITY instruction (1-2 short sentences, 8-25 words):
- Pack spatial commands tightly: direction + landmark + next direction
- Minimal filler words — every word must carry navigation meaning
- Name key visual objects from the images as waypoint markers

Example style: "Past the tiled counter, right at the wooden beam. Stop by the glass partition."

Write ONLY the instruction:"""

STYLE_PROMPTS = [STYLE_A_PROMPT, STYLE_B_PROMPT, STYLE_C_PROMPT]
STYLE_NAMES   = ["action_centric", "landmark_descriptive", "concise_spatial"]


# ── Async generation ──────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_messages(frames: List[Dict], odom_context: str, style_idx: int) -> list:
    """frames must have a 'b64' key with pre-encoded image data (avoids blocking I/O in event loop)."""
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
    encoded_frames: List[Dict],   # frames with pre-loaded 'b64' key
    odom_ctx: str,
    fallback: str,
) -> Tuple[str, List[str]]:
    """Generate 3 diverse instructions. Frames are pre-encoded to avoid sync I/O in event loop."""
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
    """Build final dataset. Each trajectory gets 3 different instructions assigned in order."""
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
        out_name = "val_unseen_v242.json.gz"
        ck_path  = ROOT / "outputs" / "gate4_v242_unseen_checkpoint.json"
    elif split == "seen":
        gt_path  = GT_SEEN
        rf_dir   = RF_SEEN_DIR
        mid_dir  = MID_SEEN_DIR
        out_name = "val_seen_v242.json.gz"
        ck_path  = ROOT / "outputs" / "gate4_v242_seen_checkpoint.json"
    else:
        raise ValueError(f"Unknown split: {split}")

    with gzip.open(gt_path, "rt") as f:
        gt_data = json.load(f)

    # Group episodes by trajectory
    traj_to_first_ep: Dict[str, dict] = {}
    for ep in gt_data["episodes"]:
        traj_id = f"traj_{ep['trajectory_id']}" if "trajectory_id" in ep else f"traj_ep{ep['episode_id']}"
        if traj_id not in traj_to_first_ep:
            traj_to_first_ep[traj_id] = ep

    n_traj = len(traj_to_first_ep)
    print(f"Split: {split} | {len(gt_data['episodes'])} episodes | {n_traj} unique trajectories")
    print(f"Generating 3 instruction styles per trajectory = {n_traj * 3} API calls")

    # Load checkpoint
    instructions: Dict[str, List[str]] = {}
    if ck_path.exists():
        with open(ck_path) as f:
            raw = json.load(f)
            # Convert to list if stored as old string format
            for k, v in raw.items():
                instructions[k] = v if isinstance(v, list) else [v, v, v]
        print(f"Resumed from checkpoint: {len(instructions)}/{n_traj} trajectories")

    selector = FrameSelector(rf_dir, mid_dir)

    # Generate remaining trajectories
    pending = [(tid, ep) for tid, ep in traj_to_first_ep.items() if tid not in instructions]
    print(f"Remaining: {len(pending)} trajectories to generate", flush=True)

    # ── Pre-encode all frames synchronously (avoids blocking event loop later) ──
    print("Pre-encoding frames...", flush=True)
    pre_encoded: List[Tuple] = []  # (traj_id, encoded_frames, odom_ctx, fallback)
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

    # ── Batch processing: CONCURRENCY trajectories at a time ──
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
        # Checkpoint every 100 trajectories
        if done % 100 < CONCURRENCY:
            with open(ck_path, "w") as f:
                json.dump(instructions, f, ensure_ascii=False)
            print(f"  [checkpoint] {done}/{len(pre_encoded)} saved", flush=True)

    # Final checkpoint
    with open(ck_path, "w") as f:
        json.dump(instructions, f, ensure_ascii=False)
    print(f"Checkpoint saved: {ck_path}", flush=True)

    # Build final dataset
    out_data = build_dataset(gt_data, instructions)
    out_path = OUT_DIR / out_name
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)

    # Also copy to Habitat dataset dir
    habitat_dir = HABITAT_BASE / split.replace("seen", "val_seen").replace("unseen", "val_unseen")
    habitat_dir = HABITAT_BASE / f"val_{split}"
    habitat_out = habitat_dir / out_name
    shutil.copy2(out_path, habitat_out)
    print(f"Saved: {out_path}")
    print(f"Copied: {habitat_out}")

    # Verify diversity
    ep_texts = [ep["instruction"]["instruction_text"] for ep in out_data["episodes"]]
    unique = len(set(ep_texts))
    unique_pct = unique / len(ep_texts) * 100
    avg_words = sum(len(t.split()) for t in ep_texts) / len(ep_texts)
    print(f"\nDiversity: {unique}/{len(ep_texts)} unique ({unique_pct:.0f}%) | avg_words={avg_words:.1f}")

    # Show style distribution sample
    print("\nStyle diversity sample (trajectory 'traj_15' or first available):")
    sample_tid = "traj_15" if "traj_15" in instructions else list(instructions.keys())[0]
    for i, (style, instr) in enumerate(zip(STYLE_NAMES, instructions[sample_tid])):
        print(f"  {style}: {instr[:90]}")


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
