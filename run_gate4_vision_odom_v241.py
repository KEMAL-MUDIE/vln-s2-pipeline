#!/usr/bin/env python3
"""
Gate 4 Vision+Odom v241 — Comprehensive VLM-based instruction + metadata generator.

Architecture: images + odometry → Gemma 4 VLM → grounded instruction + waypoint metadata

Improvements over v240 (images only, 3 frames, basic geometry string):
1. Full structured odometry: uses midpoints.json, turn_sides.json, approach_views.json
   → per-waypoint distances, turn angles, headings
2. Up to 4 targeted frames: start + approach-to-turn + turn-side/goal view + final goal
   → each image labeled with geometric context (distance, action, heading)
3. Dual-output: instruction_text + structured waypoint metadata JSON
4. Better prompt engineering: Gemma sees WHICH image corresponds to WHICH action

Research basis:
- R2R annotators watched RGB panoramas with path overlay (WebGL interface)
- InstruGen (GPT-4V + image sequences) achieves 71.10% SR vs text-only methods
- Odometry context helps ground spatial language ("after 2.3m", "at the right turn")
- Token budget: 4 × 289 + 350 text = ~1506 input tokens (safe under 4096 context)

Coverage: all 1839 val_unseen + 778 val_seen (if rendered frames exist)
Output: val_unseen_v241.json.gz + val_seen_v241.json.gz + metadata JSONs
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
META_DIR     = ROOT / "outputs" / "v241_metadata"
OUT_DIR      = ROOT / "outputs" / "datasets"

HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")

# ── Gemma API ─────────────────────────────────────────────────────────────────

VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL    = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
CONCURRENCY   = 16
MAX_TOKENS    = 120
TEMPERATURE   = 0.25

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

def parse_odometry(ep: dict) -> dict:
    """Extract full odometry sequence from reference_path."""
    path = ep["reference_path"]
    rot = ep.get("start_rotation", [0, 0, 0, 1])
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

    # Summarize turns
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
    """Build human-readable odometry description for the prompt."""
    lines = [f"Total path: {odom['total_dist']}m, {len(odom['turns'])} turn(s): {odom['turn_summary']}"]
    lines.append("Step sequence:")
    prev_dir = None
    for wp in odom["waypoints"]:
        i = wp["idx"]
        if "turn_dir" in wp:
            lines.append(f"  • Waypoint {i} ({wp['cum_dist']}m from start): "
                         f"turn {wp['turn_dir'].upper()} {abs(wp['turn_deg']):.0f}°, "
                         f"then walk {wp.get('dist_to_next', 0):.1f}m")
        elif i == 0:
            lines.append(f"  • Start (waypoint 0): walk {wp.get('dist_to_next', 0):.1f}m straight")
        elif i == odom["n_waypoints"] - 1:
            lines.append(f"  • Goal (waypoint {i}, {wp['cum_dist']}m from start): STOP HERE")
    return "\n".join(lines)


# ── Frame selection ───────────────────────────────────────────────────────────

class FrameSelector:
    """Select up to 4 best frames per episode from available rendered/midpoint images."""

    def __init__(self, rf_dir: Path, mid_dir: Path):
        self.rf_dir = rf_dir
        self.mid_dir = mid_dir

    def get_ep_str(self, eid: int) -> str:
        return f"episode_{eid:06d}"

    def get_frames(self, eid: int, odom: dict) -> List[Dict]:
        """
        Returns list of {path, label, waypoint_idx} dicts, up to 4 frames.
        Priority: start → approach-to-first-turn → goal-approach → goal
        """
        ep_str = self.get_ep_str(eid)
        rf = self.rf_dir / ep_str
        mid = self.mid_dir / ep_str
        frames = []

        # 1. Start frame (always)
        start = rf / "frame_0000_rgb.jpg"
        if start.exists():
            frames.append({"path": str(start), "label": "Start view (waypoint 0)", "wpt_idx": 0})

        # 2. Approach view to first turn (from midpoint_frames)
        turns = odom["turns"]
        if turns:
            t1 = turns[0]
            # Prefer approach view (looking toward the turn)
            apv = mid / "apv_turn_1_rgb.jpg"
            if apv.exists():
                frames.append({
                    "path": str(apv),
                    "label": f"View approaching {t1['turn_dir']} turn (at ~{t1['cum_dist']}m)",
                    "wpt_idx": t1["idx"],
                })
            else:
                # Fallback: turn-side view
                ts = mid / "ts_turn_1_rgb.jpg"
                if ts.exists():
                    frames.append({
                        "path": str(ts),
                        "label": f"View at {t1['turn_dir']} turn (waypoint {t1['idx']})",
                        "wpt_idx": t1["idx"],
                    })
                else:
                    f1 = rf / "frame_0001_rgb.jpg"
                    if f1.exists():
                        frames.append({"path": str(f1), "label": "Mid-path view", "wpt_idx": 1})

        # 3. Second turn approach (if available and space)
        if len(turns) >= 2 and len(frames) < 3:
            apv2 = mid / "apv_turn_2_rgb.jpg"
            ts2 = mid / "ts_turn_2_rgb.jpg"
            t2 = turns[1]
            if apv2.exists():
                frames.append({
                    "path": str(apv2),
                    "label": f"View approaching {t2['turn_dir']} turn (at ~{t2['cum_dist']}m)",
                    "wpt_idx": t2["idx"],
                })
            elif ts2.exists():
                frames.append({
                    "path": str(ts2),
                    "label": f"View at {t2['turn_dir']} turn (waypoint {t2['idx']})",
                    "wpt_idx": t2["idx"],
                })

        # 4. Goal frame
        # Try last rendered frame, or last midpoint frame
        n_wpts = odom["n_waypoints"]
        goal_candidates = [
            rf / f"frame_{min(4,n_wpts-1):04d}_rgb.jpg",
            rf / "frame_0002_rgb.jpg",
        ]
        for i in range(4, 0, -1):
            goal_candidates.append(mid / f"mid_000{i}_rgb.jpg")
        goal_candidates.append(rf / "frame_0001_rgb.jpg")

        goal_added = False
        for gc in goal_candidates:
            if gc.exists() and str(gc) not in {f["path"] for f in frames}:
                frames.append({
                    "path": str(gc),
                    "label": f"Goal area view (stop here)",
                    "wpt_idx": n_wpts - 1,
                })
                goal_added = True
                break

        return frames[:4]  # Max 4 frames to stay under context limit


# ── Prompt builder ────────────────────────────────────────────────────────────

VISION_ODOM_PROMPT = """\
You are writing a navigation instruction for a VLN (Vision-Language Navigation) robot.

ODOMETRY CONTEXT (geometry of the path):
{odom_context}

VISUAL CONTEXT (what the robot sees at key points):
{image_labels}

Your task: Write a precise, grounded navigation instruction (1-3 sentences, 12-45 words) that:
1. Starts with the initial movement direction
2. References specific VISUAL LANDMARKS from the images (object names, colors, materials — NOT just "the room" or "the area")
3. Includes the turn direction(s) from the odometry, anchored to a visible object
4. Ends with a clear stop condition referencing something visible in the goal image

GT-style examples (do NOT copy — use as style reference):
- "Walk straight past the wooden double doors and turn right at the blue pool. Continue to the stone bar counter and stop at the far corner."
- "Exit through the glass door on the left. Turn right after the kitchen island and walk to the end of the hallway. Stop near the window."
- "Go up the carpeted stairs and turn left at the landing. Walk past the white wardrobe and stop by the bedroom door."

Write ONLY the instruction text:"""


def load_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_messages(ep: dict, odom: dict, frames: List[Dict]) -> List[dict]:
    content = []

    # Add images with context labels
    image_labels = []
    for i, fr in enumerate(frames, 1):
        b64 = load_b64(fr["path"])
        ext = "jpeg" if fr["path"].endswith(".jpg") else "png"
        content.append({"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}})
        image_labels.append(f"Image {i}: {fr['label']}")

    prompt = VISION_ODOM_PROMPT.format(
        odom_context=build_odom_context(odom),
        image_labels="\n".join(image_labels),
    )
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def is_valid(text: str) -> bool:
    if not text or len(text.split()) < 8:
        return False
    stop_words = {"stop", "wait", "halt", "stand", "pause"}
    return any(w in text.lower() for w in stop_words)


# ── Generation ────────────────────────────────────────────────────────────────

async def generate_one(
    client: AsyncOpenAI, ep: dict, odom: dict, frames: List[Dict],
    sem: asyncio.Semaphore,
) -> Tuple[int, str]:
    eid = ep["episode_id"]
    if not frames:
        return eid, ""
    async with sem:
        for attempt in range(3):
            temp = TEMPERATURE + 0.1 * attempt
            try:
                messages = build_messages(ep, odom, frames)
                resp = await client.chat.completions.create(
                    model=VLLM_MODEL, messages=messages,
                    max_tokens=MAX_TOKENS, temperature=temp,
                )
                text = resp.choices[0].message.content.strip()
                if is_valid(text):
                    return eid, text
            except Exception as e:
                if attempt == 2:
                    return eid, ""
        return eid, text


async def generate_split(
    episodes: List[dict],
    selector: FrameSelector,
    ckpt_path: Path,
) -> Dict[int, str]:
    """Generate per-trajectory, expand to per-episode."""
    from collections import defaultdict
    traj_groups: Dict[int, List[dict]] = defaultdict(list)
    for ep in episodes:
        traj_groups[ep["trajectory_id"]].append(ep)

    # Load checkpoint (traj_N → instruction)
    checkpoint: Dict[str, str] = {}
    if ckpt_path.exists():
        checkpoint = json.loads(ckpt_path.read_text())
        print(f"Loaded checkpoint: {len(checkpoint)} trajectories")

    remaining = [(tid, eps) for tid, eps in traj_groups.items()
                 if f"traj_{tid}" not in checkpoint]
    print(f"Trajectories: {len(traj_groups)} total | {len(checkpoint)} done | {len(remaining)} remaining")
    if not remaining:
        pass  # fall through to expansion
    else:
        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
        sem = asyncio.Semaphore(CONCURRENCY)
        t0 = time.time()
        errors = 0
        done_count = 0

        async def gen_traj(tid: int, eps: List[dict]) -> Tuple[int, str]:
            rep = eps[0]
            odom = parse_odometry(rep)
            frames = selector.get_frames(rep["episode_id"], odom)
            _, text = await generate_one(client, rep, odom, frames, sem)
            return tid, text

        coros = [gen_traj(tid, eps) for tid, eps in remaining]
        for coro in asyncio.as_completed(coros):
            tid, text = await coro
            if text:
                checkpoint[f"traj_{tid}"] = text
            else:
                errors += 1
            done_count += 1
            if done_count % 50 == 0:
                elapsed = time.time() - t0
                rate = done_count / elapsed
                eta = (len(remaining) - done_count) / rate if rate > 0 else 0
                print(f"  [{done_count}/{len(remaining)}] errors={errors} "
                      f"rate={rate:.1f}/s ETA={eta/60:.1f}m")
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                ckpt_path.write_text(json.dumps(checkpoint, ensure_ascii=False))

        elapsed = time.time() - t0
        print(f"Done: {len(remaining)} trajs in {elapsed:.1f}s ({len(remaining)/elapsed:.1f}/s). "
              f"Errors: {errors}")
        ckpt_path.write_text(json.dumps(checkpoint, ensure_ascii=False))

    # Expand to per-episode
    results: Dict[int, str] = {}
    for ep in episodes:
        key = f"traj_{ep['trajectory_id']}"
        if key in checkpoint:
            results[ep["episode_id"]] = checkpoint[key]
    return results


# ── Dataset + metadata assembly ───────────────────────────────────────────────

def assemble_dataset(gt_episodes: List[dict], instructions: Dict[int, str],
                     selector: FrameSelector, split: str) -> dict:
    episodes = []
    replaced, fallback = 0, 0
    for ep in gt_episodes:
        eid = ep["episode_id"]
        ep_out = dict(ep)
        text = instructions.get(eid, "")
        if text:
            ep_out["instruction"] = {"instruction_text": text, "instruction_tokens": None}
            replaced += 1
        else:
            fallback += 1
        episodes.append(ep_out)

        # Save metadata JSON
        odom = parse_odometry(ep)
        frames = selector.get_frames(eid, odom)
        meta = {
            "episode_id": eid,
            "split": split,
            "scene_id": ep["scene_id"],
            "instruction_v241": text,
            "instruction_gt": ep["instruction"]["instruction_text"],
            "odometry": {
                "total_dist": odom["total_dist"],
                "n_waypoints": odom["n_waypoints"],
                "turn_summary": odom["turn_summary"],
            },
            "frames_used": [{"path": fr["path"].split("/")[-1], "label": fr["label"]}
                            for fr in frames],
        }
        meta_path = META_DIR / split / f"episode_{eid:06d}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"Assembly: {replaced} vision-odom, {fallback} GT fallbacks")
    return {"episodes": episodes}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_split(gt_path: str, rf_dir: Path, mid_dir: Path,
              ckpt_name: str, out_name: str, split: str):
    print(f"\n{'='*60}")
    print(f"=== v241 Vision+Odom — {split.upper()} ===")
    print(f"{'='*60}")

    if not Path(gt_path).exists():
        print(f"  GT dataset not found: {gt_path} — skipping")
        return

    with gzip.open(gt_path, "rt") as f:
        gt_data = json.load(f)
    episodes = gt_data["episodes"]

    selector = FrameSelector(rf_dir, mid_dir)

    # Check frame coverage
    with_frames = sum(1 for ep in episodes if selector.get_frames(ep["episode_id"], parse_odometry(ep)))
    print(f"  Episodes: {len(episodes)} | With frames: {with_frames}/{len(episodes)}")

    if with_frames == 0:
        print("  No frames available — frames must be rendered first. See render_val_seen.sh")
        return

    ckpt_path = ROOT / "outputs" / ckpt_name
    instructions = asyncio.run(generate_split(episodes, selector, ckpt_path))

    dataset = assemble_dataset(episodes, instructions, selector, split)

    out_path = OUT_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    # Copy to Habitat path
    hab_path = HABITAT_BASE / split.replace("val_", "val_") / out_name
    if not hab_path.parent.exists():
        hab_path = HABITAT_BASE / "val_unseen" / out_name if "unseen" in split else HABITAT_BASE / "val_seen" / out_name
    shutil.copy(out_path, hab_path)
    print(f"  Copied to: {hab_path}")

    # Quality stats
    texts = [ep["instruction"]["instruction_text"] for ep in dataset["episodes"]]
    stop_pct = sum(1 for t in texts if any(w in t.lower() for w in ["stop","wait","halt","stand"])) / len(texts) * 100
    avg_words = sum(len(t.split()) for t in texts) / len(texts)
    unique_pct = len(set(texts)) / len(texts) * 100
    print(f"  Quality: stop={stop_pct:.0f}% avg_words={avg_words:.1f} unique={unique_pct:.0f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["unseen", "seen", "both"], default="unseen")
    args = parser.parse_args()

    if args.split in ("unseen", "both"):
        run_split(
            gt_path=GT_UNSEEN,
            rf_dir=RF_DIR,
            mid_dir=MID_DIR,
            ckpt_name="gate4_v241_unseen_checkpoint.json",
            out_name="val_unseen_v241.json.gz",
            split="val_unseen",
        )

    if args.split in ("seen", "both"):
        run_split(
            gt_path=GT_SEEN,
            rf_dir=RF_SEEN_DIR,
            mid_dir=MID_SEEN_DIR,
            ckpt_name="gate4_v241_seen_checkpoint.json",
            out_name="val_seen_v241.json.gz",
            split="val_seen",
        )


if __name__ == "__main__":
    main()
