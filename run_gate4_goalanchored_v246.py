#!/usr/bin/env python3
"""
Gate 4 Vision+Odom v246 — Goal-Anchored Few-Shot with Ordinals.

Combines three techniques:
1. Trajectory-matched few-shot GT examples (v245): selects 2 authentic R2R GT
   instructions by similarity to the current path (length, turns, stairs).
2. Explicit ordinal markers (v244): prompts instruct Gemma to use "first/second/
   third" for repeated features.
3. Goal-grounded stop condition (NEW): a short pre-pass uses only the goal area
   frame to generate a 5–10 word goal description ("yellow couch on left side"),
   which is then injected into the main instruction prompt so the stop condition
   is visually specific rather than generic.

Architecture:
  For each trajectory:
    Step 1 (goal_desc): send goal frame alone → 10-token description
    Step 2 (3 styles): send all frames + goal_desc → full instruction per style
  Total API calls: 4 per trajectory (1 goal + 3 style).
  Pre-encoded at generation start; asyncio CONCURRENCY=4.
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
OUT_DIR      = ROOT / "outputs" / "datasets"
HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")

# ── Gemma API ─────────────────────────────────────────────────────────────────

VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL    = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
CONCURRENCY   = 4
API_TIMEOUT   = 90
MAX_TOKENS    = 140
GOAL_DESC_TOKENS = 20
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


# ── GT corpus for few-shot matching ───────────────────────────────────────────

def build_gt_corpus(gt_data: dict) -> List[dict]:
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
        n_turns = 0
        for i in range(1, len(path)-1):
            if abs(signed_diff(heading_xz(path[i-1], path[i]), heading_xz(path[i], path[i+1]))) > 20:
                n_turns += 1
        has_stairs = any(abs(path[i+1][1] - path[i][1]) >= STAIR_DY_THRESH for i in range(len(path)-1))
        corpus.append({
            "total_dist": round(total_dist, 1),
            "n_turns": n_turns,
            "has_stairs": has_stairs,
            "instruction": ep["instruction"]["instruction_text"],
        })
    return corpus


def find_similar_instructions(odom: dict, corpus: List[dict], n: int = 2) -> List[str]:
    scored = []
    q_dist, q_turns, q_stairs = odom["total_dist"], odom["n_turns"], odom["is_multifloor"]
    for entry in corpus:
        score = abs(entry["total_dist"] - q_dist) + 2.0 * abs(entry["n_turns"] - q_turns) - 5.0 * (entry["has_stairs"] == q_stairs)
        scored.append((score, entry["instruction"]))
    scored.sort(key=lambda x: x[0])
    seen, result = set(), []
    for _, instr in scored:
        instr_clean = instr.strip()
        if instr_clean not in seen:
            seen.add(instr_clean); result.append(instr_clean)
        if len(result) == n:
            break
    return result


# ── Frame selection ────────────────────────────────────────────────────────────

class FrameSelector:
    def __init__(self, rf_dir: Path, mid_dir: Path):
        self.rf_dir = rf_dir
        self.mid_dir = mid_dir

    def get_frames(self, eid: int, odom: dict) -> List[Dict]:
        ep_str = f"episode_{eid:06d}"
        rf = self.rf_dir / ep_str
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
                    label = f"Approaching FIRST turn ({t1['turn_dir']} at ~{t1['cum_dist']}m)"
                    frames.append({"path": str(cand), "label": label}); break
        if len(turns) >= 2 and len(frames) < 3:
            t2 = turns[1]
            for cand in [mid / "apv_turn_2_rgb.jpg", mid / "ts_turn_2_rgb.jpg"]:
                if cand.exists():
                    frames.append({"path": str(cand), "label": f"Approaching SECOND turn ({t2['turn_dir']} at ~{t2['cum_dist']}m)"}); break

        # goal frame
        n_wpts = odom["n_waypoints"]
        goal_cands = [rf / f"frame_{min(4,n_wpts-1):04d}_rgb.jpg", rf / "frame_0002_rgb.jpg"]
        for i in range(4, 0, -1):
            goal_cands.append(mid / f"mid_000{i}_rgb.jpg")
        goal_cands.append(rf / "frame_0001_rgb.jpg")
        for gc in goal_cands:
            if gc.exists() and str(gc) not in {f["path"] for f in frames}:
                frames.append({"path": str(gc), "label": "Goal area view — STOP HERE"})
                break
        return frames[:4]

    def get_goal_frame(self, eid: int, odom: dict) -> Optional[str]:
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


# ── Step 2: 3-style instruction prompts ──────────────────────────────────────

STYLE_A_TEMPLATE = """\
You are writing a navigation instruction for a VLN robot (action-centric style).

REFERENCE EXAMPLES (authentic R2R navigation instructions — match this style):
1. "{ex1}"
2. "{ex2}"

GOAL LOCATION: The destination shows "{goal_desc}".

ODOMETRY for the new path (turn numbers are EXACT):
{odom_context}

VISUAL CONTEXT:
{image_labels}

Write a crisp ACTION-CENTRIC instruction (1-3 sentences, 12-45 words):
- Start immediately with a movement command ("Walk forward...", "Turn left...", "Head toward...")
- Use ORDINAL markers for repeated features: "through the FIRST door", "at the SECOND corridor"
- If odometry shows stairs (↕ Stair), explicitly name the floor transition
- Anchor each turn to a SPECIFIC visual landmark visible in the images
- End with a SPECIFIC stop: include the goal description "{goal_desc}"

Write ONLY the instruction:"""

STYLE_B_TEMPLATE = """\
You are writing a navigation instruction for a VLN robot (landmark-descriptive style).

REFERENCE EXAMPLES (authentic R2R navigation instructions — match this style):
1. "{ex1}"
2. "{ex2}"

GOAL LOCATION: The destination shows "{goal_desc}".

ODOMETRY for the new path:
{odom_context}

VISUAL CONTEXT:
{image_labels}

Write a LANDMARK-DESCRIPTIVE instruction (1-3 sentences, 15-50 words):
- Describe what the robot SEES at each key decision point
- Use ORDINAL markers when naming repeated features: "first door", "second hallway"
- At branch points, name the DISTINCTIVE feature for the correct path; say which to AVOID
- If odometry shows stairs (↕ Stair), describe the staircase and direction
- Close with a specific visual stop: stop when you see "{goal_desc}"

Write ONLY the instruction:"""

STYLE_C_TEMPLATE = """\
You are writing a navigation instruction for a VLN robot (concise-spatial style).

REFERENCE EXAMPLES (short, terse authentic R2R instructions — match this brevity):
1. "{ex1}"
2. "{ex2}"

GOAL LOCATION: The destination shows "{goal_desc}".

ODOMETRY for the new path:
{odom_context}

VISUAL CONTEXT:
{image_labels}

Write a CONCISE instruction (1-2 sentences, 8-25 words):
- Use ordinals when needed: "2nd door", "left staircase", "far corridor"
- Every word must carry navigation meaning; no filler
- End with the goal: "{goal_desc}"

Write ONLY the instruction:"""

STYLE_TEMPLATES = [STYLE_A_TEMPLATE, STYLE_B_TEMPLATE, STYLE_C_TEMPLATE]
STYLE_NAMES     = ["action_centric", "landmark_descriptive", "concise_spatial"]


def build_messages(frames: List[Dict], odom_context: str, style_idx: int,
                   few_shot: List[str], goal_desc: str) -> list:
    image_labels = "\n".join(f"  [{i+1}] {f['label']}" for i, f in enumerate(frames))
    ex1 = few_shot[0] if len(few_shot) > 0 else "Walk forward and turn right. Stop at the door."
    ex2 = few_shot[1] if len(few_shot) > 1 else "Continue straight and stop near the window."
    prompt = STYLE_TEMPLATES[style_idx].format(
        ex1=ex1, ex2=ex2,
        goal_desc=goal_desc,
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


async def generate_instructions(
    client: AsyncOpenAI,
    traj_id: str,
    encoded_frames: List[Dict],
    goal_b64: Optional[str],
    odom_ctx: str,
    few_shot: List[str],
    fallback: str,
) -> Tuple[str, List[str]]:
    if not encoded_frames:
        return traj_id, [fallback, fallback, fallback]

    # Step 1: goal description
    if goal_b64:
        goal_desc = await get_goal_description(client, goal_b64)
    else:
        goal_desc = "the goal area"

    # Step 2: 3 style instructions in parallel
    tasks = []
    for style_idx in range(3):
        msgs = build_messages(encoded_frames, odom_ctx, style_idx, few_shot, goal_desc)
        tasks.append(asyncio.wait_for(
            client.chat.completions.create(
                model=VLLM_MODEL, messages=msgs,
                max_tokens=MAX_TOKENS, temperature=TEMPERATURES[style_idx],
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
                text += " Stop here."
            results.append(text)
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if split == "unseen":
        gt_path  = GT_UNSEEN
        rf_dir   = RF_DIR
        mid_dir  = MID_DIR
        out_name = "val_unseen_v246.json.gz"
        ck_path  = ROOT / "outputs" / "gate4_v246_unseen_checkpoint.json"
    elif split == "seen":
        gt_path  = GT_SEEN
        rf_dir   = RF_SEEN_DIR
        mid_dir  = MID_SEEN_DIR
        out_name = "val_seen_v246.json.gz"
        ck_path  = ROOT / "outputs" / "gate4_v246_seen_checkpoint.json"
    else:
        raise ValueError(f"Unknown split: {split}")

    with gzip.open(gt_path, "rt") as f:
        gt_data = json.load(f)

    print("Building GT corpus for few-shot matching...", flush=True)
    gt_corpus = build_gt_corpus(gt_data)
    print(f"GT corpus: {len(gt_corpus)} unique trajectories", flush=True)

    traj_to_first_ep: Dict[str, dict] = {}
    for ep in gt_data["episodes"]:
        traj_id = f"traj_{ep['trajectory_id']}" if "trajectory_id" in ep else f"traj_ep{ep['episode_id']}"
        if traj_id not in traj_to_first_ep:
            traj_to_first_ep[traj_id] = ep

    n_traj = len(traj_to_first_ep)
    print(f"Split: {split} | {len(gt_data['episodes'])} eps | {n_traj} unique trajectories")

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
    pre_encoded: List[Tuple] = []
    for tid, ep in pending:
        odom = parse_odometry(ep)
        odom_ctx = build_odom_context(odom)
        frames = selector.get_frames(ep["episode_id"], odom)
        few_shot = find_similar_instructions(odom, gt_corpus, n=2)
        fallback = f"Walk {odom['total_dist']}m. {odom['turn_summary']}. Stop at goal."
        encoded_frames = []
        for fr in frames:
            try:
                encoded_frames.append({**fr, "b64": encode_image(fr["path"])})
            except Exception:
                pass
        # separate goal b64
        goal_path = selector.get_goal_frame(ep["episode_id"], odom)
        goal_b64 = None
        if goal_path:
            try:
                goal_b64 = encode_image(goal_path)
            except Exception:
                pass
        pre_encoded.append((tid, encoded_frames, goal_b64, odom_ctx, few_shot, fallback))
    print(f"Pre-encoded {len(pre_encoded)} trajectories.", flush=True)

    t0 = time.time()
    done = 0

    for batch_start in range(0, len(pre_encoded), CONCURRENCY):
        batch = pre_encoded[batch_start:batch_start + CONCURRENCY]
        batch_results = await asyncio.gather(*[
            generate_instructions(client, tid, enc_frames, goal_b64, odom_ctx, few_shot, fallback)
            for tid, enc_frames, goal_b64, odom_ctx, few_shot, fallback in batch
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
                print(f"    A: {a[:80]}", flush=True)
                print(f"    B: {b[:80]}", flush=True)
                print(f"    C: {c[:80]}", flush=True)
                print(f"    Rate: {rate:.2f} traj/s | ETA: {remain/60:.1f}m", flush=True)
        if done % 100 < CONCURRENCY:
            with open(ck_path, "w") as f:
                json.dump(instructions, f, ensure_ascii=False)
            print(f"  [checkpoint] {done}/{len(pre_encoded)}", flush=True)

    with open(ck_path, "w") as f:
        json.dump(instructions, f, ensure_ascii=False)

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
    print(f"\nDiversity: {unique}/{len(ep_texts)} ({unique/len(ep_texts)*100:.0f}%) | avg_words={avg_words:.1f}")
    print(f"Ordinal markers: {ordinal_count/len(ep_texts)*100:.1f}%")
    print(f"\nSample (first key):")
    k = list(instructions.keys())[0]
    for sname, instr in zip(STYLE_NAMES, instructions[k]):
        print(f"  {sname[:1].upper()}: {instr[:90]}")


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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
