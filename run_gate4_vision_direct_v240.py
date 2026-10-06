#!/usr/bin/env python3
"""
Gate 4 Vision-Direct v240 — Single-call VLM instruction generation from rendered frames.

Instead of the multi-stage pipeline (gate2 geometry → gate3 text captions → gate4 Gemma text),
this script feeds 3 rendered RGB frames (start, mid-path, goal) directly to Gemma 4 VLM
alongside path geometry context, producing visually-grounded navigation instructions in one call.

Why this should outperform text-only generation:
- R2R annotators watched actual path video → GT instructions are vision-grounded
- Direct image input → Gemma sees real colors, objects, textures (not imperfect text captions)
- No 200-step text post-processing needed — diversity comes naturally from visual content
- 100% episode coverage (no truly-diff filter needed)

Token budget per episode:
- 3 images × ~289 tokens = 867 tokens
- Text prompt: ~200 tokens
- Total input: ~1067 tokens (well under 4096 context limit)
- Completion: ~80 tokens
- Total: ~1147 tokens

Expected generation time: ~3-5 minutes for all 1839 val_unseen episodes at 16 concurrency.

Input:  val_unseen.json.gz (GT dataset, 1839 episodes)
Output: outputs/datasets/val_unseen_vision_direct_v240.json.gz
"""
import asyncio
import base64
import gzip
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
GT_PATH   = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz"
RF_DIR    = ROOT / "outputs" / "rendered_frames"       # frame_0000/0001/0002_rgb.jpg per episode
MID_DIR   = Path("/mnt/nvme0/vln_habitat/habitat_data/midpoint_frames")  # ts_turn/apv_turn/mid frames
CKPT_PATH = ROOT / "outputs" / "gate4_v240_checkpoint.json"
OUTPUT    = ROOT / "outputs" / "datasets" / "val_unseen_vision_direct_v240.json.gz"

# ── Gemma API ─────────────────────────────────────────────────────────────────

VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL    = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
CONCURRENCY   = 16
MAX_TOKENS    = 96
TEMPERATURE   = 0.25

# ── Prompt ────────────────────────────────────────────────────────────────────

VISION_PROMPT_TEMPLATE = """\
Write a navigation instruction for this indoor path.

Image 1 = starting view (where you begin)
Image 2 = mid-path view (at a waypoint or turn)
Image 3 = goal view (where you must stop)

Path: {path_desc}

Rules:
- 1-3 sentences, 10-40 words
- Name specific visual landmarks you can see (furniture, colors, doors, floor materials, wall features)
- End with a stop condition referencing something visible in Image 3
- Write ONLY the instruction, no labels or preamble"""

# ── Geometry helpers ──────────────────────────────────────────────────────────

def dist3d(p1, p2) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def heading_xz(p1, p2) -> float:
    dx, dz = p2[0] - p1[0], p2[2] - p1[2]
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, -dz))


def path_description(ep: dict) -> str:
    path = ep["reference_path"]
    total_dist = sum(dist3d(path[i], path[i + 1]) for i in range(len(path) - 1))

    turns = []
    if len(path) >= 3:
        prev_h = heading_xz(path[0], path[1])
        for i in range(1, len(path) - 1):
            h = heading_xz(path[i], path[i + 1])
            diff = (h - prev_h + 180) % 360 - 180
            if abs(diff) > 25:
                turns.append("left" if diff < 0 else "right")
            prev_h = h

    if not turns:
        return f"{total_dist:.1f}m straight path"
    elif len(turns) == 1:
        return f"{total_dist:.1f}m path, one {turns[0]} turn"
    else:
        turn_str = ", ".join(turns)
        return f"{total_dist:.1f}m path, {len(turns)} turns ({turn_str})"


# ── Frame selection ───────────────────────────────────────────────────────────

def get_frame_paths(eid: int) -> List[str]:
    ep_str = f"episode_{eid:06d}"
    frames = []

    # Start frame
    start = RF_DIR / ep_str / "frame_0000_rgb.jpg"
    if start.exists():
        frames.append(str(start))

    # Mid-path frame: prefer turn-side view > approach view > mid frame > fallback
    mid_found = False
    for mf_name in ["ts_turn_1_rgb.jpg", "apv_turn_1_rgb.jpg", "mid_0002_rgb.jpg",
                     "mid_0003_rgb.jpg", "mid_0001_rgb.jpg"]:
        p = MID_DIR / ep_str / mf_name
        if p.exists():
            frames.append(str(p))
            mid_found = True
            break
    if not mid_found:
        p = RF_DIR / ep_str / "frame_0001_rgb.jpg"
        if p.exists():
            frames.append(str(p))

    # Goal frame
    goal = RF_DIR / ep_str / "frame_0002_rgb.jpg"
    if goal.exists():
        frames.append(str(goal))

    return frames


def load_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ── Generation ────────────────────────────────────────────────────────────────

def build_messages(ep: dict) -> List[dict]:
    eid = ep["episode_id"]
    frames = get_frame_paths(eid)

    content = []
    for fp in frames[:3]:
        b64 = load_b64(fp)
        ext = "jpeg" if fp.endswith(".jpg") else "png"
        content.append({"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}})

    prompt = VISION_PROMPT_TEMPLATE.format(path_desc=path_description(ep))
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def is_valid(text: str) -> bool:
    if not text or len(text.split()) < 8:
        return False
    stop_words = {"stop", "wait", "halt", "stand", "pause"}
    if not any(w in text.lower() for w in stop_words):
        return False
    return True


async def generate_one(
    client: AsyncOpenAI,
    ep: dict,
    sem: asyncio.Semaphore,
) -> Tuple[int, str]:
    eid = ep["episode_id"]
    async with sem:
        for attempt in range(3):
            temp = TEMPERATURE + 0.1 * attempt
            try:
                messages = build_messages(ep)
                resp = await client.chat.completions.create(
                    model=VLLM_MODEL,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=temp,
                )
                text = resp.choices[0].message.content.strip()
                if is_valid(text):
                    return eid, text
                # retry with higher temp
            except Exception as e:
                if attempt == 2:
                    print(f"  ERROR ep {eid}: {e}")
                    return eid, ""
        return eid, text  # return last attempt even if invalid


async def generate_all(episodes: List[dict], checkpoint: Dict[int, str]) -> Dict[int, str]:
    """
    Generate one instruction per unique trajectory_id (613 unique paths × 3 annotations = 1839 eps).
    This is 3x faster than per-episode generation; annotations of the same path share the same VIS
    instruction (consistent and unambiguous for the same visual content).
    Checkpoint keys are trajectory_id (str) → instruction text.
    """
    # Group by trajectory_id
    from collections import defaultdict
    traj_groups: Dict[int, List[dict]] = defaultdict(list)
    for ep in episodes:
        traj_groups[ep["trajectory_id"]].append(ep)

    # Remaining trajectories not yet in checkpoint
    remaining_trajs = [
        (tid, eps_list)
        for tid, eps_list in traj_groups.items()
        if f"traj_{tid}" not in checkpoint
    ]
    done_trajs = len(traj_groups) - len(remaining_trajs)
    print(f"Unique trajectories: {len(traj_groups)} | Done: {done_trajs} | Remaining: {len(remaining_trajs)}")

    if not remaining_trajs:
        # Expand checkpoint to all episodes
        results: Dict[int, str] = {}
        for ep in episodes:
            key = f"traj_{ep['trajectory_id']}"
            if key in checkpoint:
                results[ep["episode_id"]] = checkpoint[key]
        return results

    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
    sem = asyncio.Semaphore(CONCURRENCY)

    traj_results: Dict[str, str] = dict(checkpoint)
    t0 = time.time()
    errors = 0
    batch_size = 50

    # Use first episode of each trajectory for generation (all share same frames)
    async def gen_traj(tid: int, eps_list: List[dict]) -> Tuple[int, str]:
        rep_ep = eps_list[0]
        _, text = await generate_one(client, rep_ep, sem)
        return tid, text

    coros = [gen_traj(tid, eps_list) for tid, eps_list in remaining_trajs]

    done_count = 0
    for coro in asyncio.as_completed(coros):
        tid, text = await coro
        key = f"traj_{tid}"
        if text:
            traj_results[key] = text
        else:
            errors += 1
        done_count += 1

        if done_count % batch_size == 0:
            elapsed = time.time() - t0
            rate = done_count / elapsed
            eta = (len(remaining_trajs) - done_count) / rate if rate > 0 else 0
            print(f"  [{done_count}/{len(remaining_trajs)}] errors={errors} "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}m")
            save_checkpoint(traj_results)

    elapsed = time.time() - t0
    print(f"Done. {len(remaining_trajs)} trajectories in {elapsed:.1f}s ({len(remaining_trajs)/elapsed:.1f}/s). Errors: {errors}")

    # Expand to per-episode results
    results: Dict[int, str] = {}
    for ep in episodes:
        key = f"traj_{ep['trajectory_id']}"
        if key in traj_results:
            results[ep["episode_id"]] = traj_results[key]
    return results


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> Dict[int, str]:
    if CKPT_PATH.exists():
        data = json.loads(CKPT_PATH.read_text())
        print(f"Loaded checkpoint: {len(data)} episodes done")
        return {int(k): v for k, v in data.items()}
    return {}


def save_checkpoint(results: Dict[int, str]) -> None:
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CKPT_PATH.write_text(json.dumps({str(k): v for k, v in results.items()}, ensure_ascii=False))


# ── Dataset assembly ──────────────────────────────────────────────────────────

def assemble_dataset(gt_episodes: List[dict], instructions: Dict[int, str]) -> dict:
    episodes = []
    replaced = 0
    fallback = 0
    no_frames = 0
    for ep in gt_episodes:
        eid = ep["episode_id"]
        ep_out = dict(ep)
        text = instructions.get(eid, "")
        if text:
            ep_out["instruction"] = {
                "instruction_text": text,
                "instruction_tokens": None,
            }
            replaced += 1
        else:
            fallback += 1
            # Check if frames exist
            if not get_frame_paths(eid):
                no_frames += 1
        episodes.append(ep_out)
    print(f"Assembly: {replaced} vision-direct, {fallback} GT fallbacks ({no_frames} no-frames)")
    return {"episodes": episodes}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"=== Gate 4 Vision-Direct v240 ===")
    print(f"Model: {VLLM_MODEL}")
    print(f"Concurrency: {CONCURRENCY}")
    print()

    # Load GT dataset
    print("Loading GT dataset...")
    with gzip.open(GT_PATH, "rt") as f:
        gt_data = json.load(f)
    episodes = gt_data["episodes"]
    print(f"Loaded {len(episodes)} val_unseen episodes")

    # Check frame coverage
    missing = sum(1 for ep in episodes if len(get_frame_paths(ep["episode_id"])) < 2)
    print(f"Frame coverage: {len(episodes) - missing}/{len(episodes)} have ≥2 frames")
    print()

    # Load checkpoint
    checkpoint = load_checkpoint()

    # Generate
    instructions = asyncio.run(generate_all(episodes, checkpoint))

    # Save final checkpoint
    save_checkpoint(instructions)
    print(f"Total instructions generated: {len(instructions)}")

    # Spot-check quality
    print("\n=== Quality spot-check (first 5) ===")
    gt_map = {ep["episode_id"]: ep for ep in episodes}
    for eid in sorted(instructions.keys())[:5]:
        gt_instr = gt_map[eid]["instruction"]["instruction_text"]
        vis_instr = instructions[eid]
        print(f"  ep{eid:5d} GT:  {gt_instr[:80]}")
        print(f"  ep{eid:5d} VIS: {vis_instr[:80]}")
        print()

    # Assemble dataset
    print("Assembling dataset...")
    dataset = assemble_dataset(episodes, instructions)

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUTPUT, "wt", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)
    print(f"Saved: {OUTPUT}")

    # Copy to habitat data path
    habitat_out = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v240.json.gz")
    import shutil
    shutil.copy(OUTPUT, habitat_out)
    print(f"Copied to: {habitat_out}")


if __name__ == "__main__":
    main()
