#!/usr/bin/env python3
"""
Gate 4 Vision+Odom v245 — Trajectory-matched few-shot GT examples.

Key improvement over v244: each prompt includes 2 real R2R GT instructions
selected by similarity to the current trajectory (path length, turn count,
stair profile). This anchors Gemma to the authentic R2R style — vocabulary,
landmark density, sentence structure — that InternVLA-N1 was trained on.

Selection metric (lower = more similar):
  score = |Δtotal_dist| + 2.0 * |Δn_turns| - 5.0 * (floor_match)
  floor_match = 1 if both or neither have stairs, 0 otherwise

All other aspects retained from v244:
  - height-aware odometry (stair segment detection)
  - 3 diverse styles (action-centric, landmark-descriptive, concise-spatial)
  - ordinal turn markers in odom context
  - CONCURRENCY=4, API_TIMEOUT=90, asyncio batching
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
META_DIR     = ROOT / "outputs" / "v245_metadata"
OUT_DIR      = ROOT / "outputs" / "datasets"

HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")

# ── Gemma API ─────────────────────────────────────────────────────────────────

VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL    = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
CONCURRENCY   = 4
API_TIMEOUT   = 90
MAX_TOKENS    = 140
TEMPERATURES  = [0.20, 0.30, 0.25]

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


# ── Trajectory-matched few-shot selection ─────────────────────────────────────

def build_gt_corpus(gt_data: dict) -> List[dict]:
    """Build a de-duped corpus (one entry per trajectory) with pre-parsed odom features."""
    seen_traj = set()
    corpus = []
    for ep in gt_data["episodes"]:
        tid = ep.get("trajectory_id", ep["episode_id"])
        if tid in seen_traj:
            continue
        seen_traj.add(tid)
        path = ep.get("reference_path", [])
        if len(path) < 2:
            continue
        total_dist = sum(dist3d(path[i], path[i+1]) for i in range(len(path)-1))
        # count turns (>20 deg)
        n_turns = 0
        for i in range(1, len(path)-1):
            h_in  = heading_xz(path[i-1], path[i])
            h_out = heading_xz(path[i], path[i+1])
            if abs(signed_diff(h_in, h_out)) > 20:
                n_turns += 1
        # detect stairs
        has_stairs = any(
            abs(path[i+1][1] - path[i][1]) >= STAIR_DY_THRESH
            for i in range(len(path)-1)
        )
        corpus.append({
            "total_dist": round(total_dist, 1),
            "n_turns": n_turns,
            "has_stairs": has_stairs,
            "instruction": ep["instruction"]["instruction_text"],
        })
    return corpus


def find_similar_instructions(odom: dict, corpus: List[dict], n: int = 2) -> List[str]:
    """Return n GT instructions most similar to the current trajectory."""
    q_dist    = odom["total_dist"]
    q_turns   = odom["n_turns"]
    q_stairs  = odom["is_multifloor"]
    scored = []
    for entry in corpus:
        dist_diff  = abs(entry["total_dist"] - q_dist)
        turn_diff  = abs(entry["n_turns"] - q_turns)
        floor_match = 1 if entry["has_stairs"] == q_stairs else 0
        score = dist_diff + 2.0 * turn_diff - 5.0 * floor_match
        scored.append((score, entry["instruction"]))
    scored.sort(key=lambda x: x[0])
    # deduplicate
    seen, result = set(), []
    for _, instr in scored:
        instr_clean = instr.strip()
        if instr_clean not in seen:
            seen.add(instr_clean)
            result.append(instr_clean)
        if len(result) == n:
            break
    return result


# ── Frame selection (same as v241–v244) ───────────────────────────────────────

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
            t2 = turns[1]
            apv2 = mid / "apv_turn_2_rgb.jpg"
            ts2  = mid / "ts_turn_2_rgb.jpg"
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


# ── 3-style prompts with few-shot GT examples ─────────────────────────────────

STYLE_A_TEMPLATE = """\
You are writing a navigation instruction for a VLN robot (action-centric style).

REFERENCE EXAMPLES (authentic R2R navigation instructions — match this style):
1. "{ex1}"
2. "{ex2}"

ODOMETRY for the new path (turn numbers are EXACT):
{odom_context}

VISUAL CONTEXT:
{image_labels}

Write a crisp ACTION-CENTRIC instruction (1-3 sentences, 12-45 words) matching the reference style:
- Start immediately with a movement command ("Walk forward...", "Turn left...", "Head toward...")
- Use ORDINAL markers for repeated features: "through the FIRST door", "at the SECOND corridor"
- If odometry shows stairs (↕ Stair), explicitly name the floor transition
- Anchor each turn to a SPECIFIC visual landmark visible in the images
- End with a concrete stop condition

Write ONLY the instruction:"""

STYLE_B_TEMPLATE = """\
You are writing a navigation instruction for a VLN robot (landmark-descriptive style).

REFERENCE EXAMPLES (authentic R2R navigation instructions — match this style):
1. "{ex1}"
2. "{ex2}"

ODOMETRY for the new path:
{odom_context}

VISUAL CONTEXT:
{image_labels}

Write a LANDMARK-DESCRIPTIVE instruction (1-3 sentences, 15-50 words) matching the reference style:
- Describe what the robot SEES at each key decision point
- At branch points, name the DISTINCTIVE feature that identifies the correct path
- If there are similar-looking options, specify which to avoid
- If odometry shows stairs (↕ Stair), describe the staircase and direction
- Close with a clear visual description of the stopping location

Write ONLY the instruction:"""

STYLE_C_TEMPLATE = """\
You are writing a navigation instruction for a VLN robot (concise-spatial style).

REFERENCE EXAMPLES (short, terse authentic R2R instructions — match this brevity):
1. "{ex1}"
2. "{ex2}"

ODOMETRY for the new path:
{odom_context}

VISUAL CONTEXT:
{image_labels}

Write a CONCISE instruction (1-2 sentences, 8-25 words) matching the reference brevity:
- Use ordinals when needed: "2nd door", "left staircase", "far corridor"
- Every word must carry navigation meaning; no filler
- If stairs present, name them briefly

Write ONLY the instruction:"""

STYLE_TEMPLATES = [STYLE_A_TEMPLATE, STYLE_B_TEMPLATE, STYLE_C_TEMPLATE]
STYLE_NAMES     = ["action_centric", "landmark_descriptive", "concise_spatial"]


# ── Async generation ──────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_messages(frames: List[Dict], odom_context: str, style_idx: int,
                   few_shot: List[str]) -> list:
    image_labels = "\n".join(f"  [{i+1}] {f['label']}" for i, f in enumerate(frames))
    ex1 = few_shot[0] if len(few_shot) > 0 else "Walk forward and turn right. Stop at the door."
    ex2 = few_shot[1] if len(few_shot) > 1 else "Continue straight and stop near the window."
    prompt = STYLE_TEMPLATES[style_idx].format(
        ex1=ex1, ex2=ex2,
        odom_context=odom_context,
        image_labels=image_labels,
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


async def generate_three_instructions(
    client: AsyncOpenAI,
    traj_id: str,
    encoded_frames: List[Dict],
    odom_ctx: str,
    few_shot: List[str],
    fallback: str,
) -> Tuple[str, List[str]]:
    if not encoded_frames:
        return traj_id, [fallback, fallback, fallback]
    results = []
    for style_idx in range(3):
        msgs = build_messages(encoded_frames, odom_ctx, style_idx, few_shot)
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=VLLM_MODEL, messages=msgs,
                    max_tokens=MAX_TOKENS, temperature=TEMPERATURES[style_idx],
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
        out_name = "val_unseen_v245.json.gz"
        ck_path  = ROOT / "outputs" / "gate4_v245_unseen_checkpoint.json"
    elif split == "seen":
        gt_path  = GT_SEEN
        rf_dir   = RF_SEEN_DIR
        mid_dir  = MID_SEEN_DIR
        out_name = "val_seen_v245.json.gz"
        ck_path  = ROOT / "outputs" / "gate4_v245_seen_checkpoint.json"
    else:
        raise ValueError(f"Unknown split: {split}")

    with gzip.open(gt_path, "rt") as f:
        gt_data = json.load(f)

    # Build trajectory-matched corpus from GT
    print("Building GT corpus for few-shot matching...", flush=True)
    gt_corpus = build_gt_corpus(gt_data)
    print(f"GT corpus: {len(gt_corpus)} unique trajectories", flush=True)

    traj_to_first_ep: Dict[str, dict] = {}
    for ep in gt_data["episodes"]:
        traj_id = f"traj_{ep['trajectory_id']}" if "trajectory_id" in ep else f"traj_ep{ep['episode_id']}"
        if traj_id not in traj_to_first_ep:
            traj_to_first_ep[traj_id] = ep

    n_traj = len(traj_to_first_ep)
    print(f"Split: {split} | {len(gt_data['episodes'])} episodes | {n_traj} unique trajectories")

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

    print("Pre-encoding frames and computing few-shot matches...", flush=True)
    pre_encoded: List[Tuple] = []
    for tid, ep in pending:
        odom = parse_odometry(ep)
        odom_ctx = build_odom_context(odom)
        frames = selector.get_frames(ep["episode_id"], odom)
        few_shot = find_similar_instructions(odom, gt_corpus, n=2)
        fallback = f"Walk {odom['total_dist']}m. {odom['turn_summary']}. Stop at goal."
        encoded_frames = []
        for f in frames:
            try:
                encoded_frames.append({**f, "b64": encode_image(f["path"])})
            except Exception:
                pass
        pre_encoded.append((tid, encoded_frames, odom_ctx, few_shot, fallback))
    print(f"Pre-encoded {len(pre_encoded)} trajectories.", flush=True)

    t0 = time.time()
    done = 0

    for batch_start in range(0, len(pre_encoded), CONCURRENCY):
        batch = pre_encoded[batch_start:batch_start + CONCURRENCY]
        batch_results = await asyncio.gather(*[
            generate_three_instructions(client, tid, enc_frames, odom_ctx, few_shot, fallback)
            for tid, enc_frames, odom_ctx, few_shot, fallback in batch
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
    ordinal_count = sum(1 for t in ep_texts if any(o in t.lower() for o in ["first", "second", "third"]))
    print(f"\nDiversity: {unique}/{len(ep_texts)} unique ({unique/len(ep_texts)*100:.0f}%) | avg_words={avg_words:.1f}")
    print(f"Ordinal markers: {ordinal_count/len(ep_texts)*100:.1f}%")

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
