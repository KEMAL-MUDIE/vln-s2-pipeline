#!/usr/bin/env python3
"""
Phase 1 Direction-Aware Turn Descriptions — v150

Regenerates ONLY the turn_N descriptions from Phase C checkpoint with direction-aware prompts.
The VLM now knows whether the robot will turn LEFT or RIGHT at each waypoint, enabling it to
describe the most distinctive landmark in the correct direction.

Key improvement over v143 Phase 1:
  - v143: "Turn at the large, light-brown wooden support pillar." (no direction context)
  - v150: "Turn LEFT at the green vintage stove." (direction-aware, more specific)

Process:
  1. Load v143 Phase 1 checkpoint (start + goal descriptions — keep these as-is)
  2. For each episode, compute turn directions from path primitives
  3. Re-generate turn_N descriptions using direction-specific prompts
  4. Merge into new v150 Phase 1 checkpoint

Output: outputs/gate4_v150_phase1_checkpoint.json
"""

import asyncio
import base64
import gzip
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ── Config ────────────────────────────────────────────────────────────────────

VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL    = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
VLLM_API_KEY  = "token-abc123"

PIPELINE_ROOT   = Path(__file__).parent
RF_DIR          = PIPELINE_ROOT / "outputs" / "rendered_frames"
GT_PATH         = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
V143_P1_CKPT    = PIPELINE_ROOT / "outputs" / "gate4_v143_phase1_checkpoint.json"
V150_P1_CKPT    = PIPELINE_ROOT / "outputs" / "gate4_v150_phase1_checkpoint.json"

sys.path.insert(0, str(PIPELINE_ROOT))
try:
    from gate2_path.path_analyzer import analyze_path
except ImportError:
    print("WARNING: gate2_path not found — using straight-line direction fallback")
    def analyze_path(path, rot=None, **kw):
        return {"primitives": [{"type": "stop"}], "summary": {}}


# ── Direction-aware vision prompts ────────────────────────────────────────────

VISION_PROMPT_TURN_LEFT = (
    "A robot is navigating indoors and is about to turn LEFT at this location. "
    "In 1-2 sentences: (1) name the most distinctive landmark to the LEFT "
    "or slightly ahead-left that a person would use to remember 'turn here' — "
    "use a SPECIFIC object type + color (e.g. 'green stove', 'grey stone pillar', "
    "'brown leather sofa') — avoid using 'wooden' as the ONLY descriptor since many "
    "indoor objects are wooden. (2) Briefly describe what's visible straight ahead "
    "after turning left (what the robot will walk toward). "
    "Examples: 'Turn left at the green vintage stove with ceramic knobs. Ahead, a "
    "stone-floored dining area with rustic beams opens up.' or 'The grey stone "
    "pillar on the left marks the corner. A narrow hallway with white walls continues left.' "
    "Reply with ONLY these 1-2 sentences, nothing else."
)

VISION_PROMPT_TURN_RIGHT = (
    "A robot is navigating indoors and is about to turn RIGHT at this location. "
    "In 1-2 sentences: (1) name the most distinctive landmark to the RIGHT "
    "or slightly ahead-right that a person would use to remember 'turn here' — "
    "use a SPECIFIC object type + color (e.g. 'white marble island', 'blue sectional sofa', "
    "'dark oak bookshelf') — avoid using 'wooden' as the ONLY descriptor since many "
    "indoor objects are wooden. (2) Briefly describe what's visible straight ahead "
    "after turning right (what the robot will walk toward). "
    "Examples: 'Turn right past the white marble kitchen island with bar stools. "
    "Ahead, a sunlit hallway leads to the bedrooms.' or 'The brown oak bookshelf "
    "on the right marks the corner. A living room with a blue sofa opens ahead.' "
    "Reply with ONLY these 1-2 sentences, nothing else."
)

VISION_PROMPT_TURN_GENERIC = (
    "A robot is navigating indoors and is about to turn at this location. "
    "In 1-2 sentences: (1) describe the most prominent landmark at this turning point "
    "(color, material, shape), and (2) briefly note what's visible in the direction the "
    "robot will go after turning. Focus on navigation-useful details a person would remember. "
    "Examples: 'Turn at the white rectangular dining table with dark chairs. Ahead, a sunlit "
    "living room with hardwood floors opens up.' or 'The grey stone pillar marks the corner. "
    "A hallway with wooden panels continues to the left.' "
    "Reply with ONLY these 1-2 sentences, nothing else."
)


def get_turn_prompt(direction: Optional[str]) -> str:
    if direction == "left":
        return VISION_PROMPT_TURN_LEFT
    elif direction == "right":
        return VISION_PROMPT_TURN_RIGHT
    else:
        return VISION_PROMPT_TURN_GENERIC


# ── Path analysis helpers ─────────────────────────────────────────────────────

def heading_xz(p1: List, p2: List) -> float:
    """Compass heading in degrees from p1→p2 on XZ plane."""
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2]
    return math.degrees(math.atan2(dx, -dz))


def signed_diff(a: float, b: float) -> float:
    """Signed angular difference a→b in (-180, 180]."""
    return (b - a + 180) % 360 - 180


def get_turn_direction_at_waypoint(path: List, waypoint_idx: int) -> Optional[str]:
    """Compute left/right turn direction at a specific waypoint in the reference path.
    Uses path geometry directly from poses.json waypoint_idx — no path_analyzer needed.
    This avoids count mismatches between renderer and path_analyzer.
    """
    n = len(path)
    i = waypoint_idx
    if i <= 0 or i >= n - 1:
        return None
    heading_before = heading_xz(path[i - 1], path[i])
    heading_after = heading_xz(path[i], path[i + 1])
    delta = signed_diff(heading_before, heading_after)
    if delta < -15:
        return "left"
    elif delta > 15:
        return "right"
    else:
        return None


# ── Image utilities ───────────────────────────────────────────────────────────

def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_vision_desc(raw: str) -> Optional[str]:
    raw = raw.strip().strip('"\'').strip()
    for prefix in ["Description:", "The landmark is", "I see", "I can see",
                   "In this image", "This image shows", "The image shows"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].strip(" :-").strip()
    sents = re.split(r'(?<=[.!?])\s+', raw)
    raw = " ".join(sents[:2]).strip()
    raw = re.sub(r"\s+", " ", raw).strip()
    words = raw.split()
    if len(words) < 3 or len(words) > 60:
        return None
    return raw


# ── Async VLM call ────────────────────────────────────────────────────────────

async def vision_describe_one(client, eid: str, image_path: Path, label: str,
                               direction: Optional[str], sem: asyncio.Semaphore,
                               done: List, total: int, t0: float) -> Dict:
    """Single async vision call for one turn frame."""
    async with sem:
        prompt_text = get_turn_prompt(direction)
        try:
            b64 = image_to_base64(image_path)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt_text},
                ]
            }]
            resp = await client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                max_tokens=100,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            desc = clean_vision_desc(raw)
            result = {"eid": eid, "label": label, "desc": desc, "ok": True, "direction": direction}
        except Exception as e:
            result = {"eid": eid, "label": label, "desc": None, "ok": False, "error": str(e), "direction": direction}

    done[0] += 1
    if done[0] % 200 == 0 or done[0] == total:
        elapsed = time.time() - t0
        r = done[0] / elapsed if elapsed > 0 else 0.001
        eta = (total - done[0]) / r if r > 0 else 0
        print(f"  [Phase1-v150 {done[0]}/{total}] {r:.1f}/s  ETA={eta/60:.1f}m", flush=True)
    return result


async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-episodes", type=int, default=None)
    ap.add_argument("--force-regen", action="store_true", help="Regenerate even if already in v150 checkpoint")
    args = ap.parse_args()

    print("=== Phase 1 v150: Direction-Aware Turn Descriptions ===")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  RF_DIR: {RF_DIR}")
    print(f"  v143 checkpoint: {V143_P1_CKPT}")
    print(f"  v150 output: {V150_P1_CKPT}")

    # Load GT episodes
    with gzip.open(GT_PATH, "rt") as f:
        gt_data = json.load(f)
    episodes = gt_data["episodes"]
    if args.n_episodes:
        episodes = episodes[:args.n_episodes]
    ep_map = {ep["episode_id"]: ep for ep in episodes}
    print(f"  Episodes: {len(episodes)}")

    # Load v143 Phase 1 checkpoint (start + goal descriptions — keep as-is)
    v143_p1 = {}
    if V143_P1_CKPT.exists():
        v143_p1 = json.load(open(V143_P1_CKPT))
        print(f"  v143 checkpoint: {len(v143_p1)} episodes")
    else:
        print("  WARNING: v143 checkpoint not found! Starting fresh.")

    # Load existing v150 checkpoint (for resuming)
    v150_p1 = {}
    if V150_P1_CKPT.exists() and not args.force_regen:
        v150_p1 = json.load(open(V150_P1_CKPT))
        print(f"  v150 checkpoint (existing): {len(v150_p1)} episodes")

    # Initialize v150 checkpoint from v143 (inherit start + goal)
    # For each episode: start with v143's start/goal, then override turn_N with new direction-aware
    merged = {}
    for eid_str, v143_descs in v143_p1.items():
        merged[eid_str] = {}
        # Keep start and goal from v143
        if "start" in v143_descs:
            merged[eid_str]["start"] = v143_descs["start"]
        if "goal" in v143_descs:
            merged[eid_str]["goal"] = v143_descs["goal"]
        # Copy existing v150 turn descriptions if already computed
        if eid_str in v150_p1:
            for k, v in v150_p1[eid_str].items():
                if k.startswith("turn_"):
                    merged[eid_str][k] = v

    print(f"  Merged checkpoint initialized: {len(merged)} episodes")

    # Build tasks — only turn_N frames that haven't been re-generated
    tasks = []
    direction_stats = {"left": 0, "right": 0, "unknown": 0}

    for ep in episodes:
        eid = ep["episode_id"]
        eid_str = str(eid)

        ep_dir = RF_DIR / f"episode_{eid:06d}"
        poses_f = ep_dir / "poses.json"
        if not poses_f.exists():
            continue

        try:
            poses = json.load(open(poses_f))
        except Exception:
            continue

        ref_path = ep.get("reference_path", [])

        for frame in poses.get("frames", []):
            label = frame["label"]
            if not label.startswith("turn_"):
                continue

            # Skip if already computed in v150
            if not args.force_regen and label in merged.get(eid_str, {}):
                continue

            img_path = ep_dir / frame["path"]
            if not img_path.exists():
                continue

            # Compute direction directly from path geometry at this waypoint
            # Uses frame['waypoint_idx'] — no path_analyzer count mismatch issues
            waypoint_idx = frame.get("waypoint_idx", 0)
            direction = get_turn_direction_at_waypoint(ref_path, waypoint_idx)
            if direction:
                direction_stats[direction] += 1
            else:
                direction_stats["unknown"] += 1

            tasks.append({
                "eid": eid_str,
                "label": label,
                "image_path": img_path,
                "direction": direction,
            })

    print(f"\n  Turn tasks to generate: {len(tasks)}")
    print(f"  Direction breakdown: left={direction_stats['left']}, "
          f"right={direction_stats['right']}, unknown={direction_stats['unknown']}")

    if not tasks:
        print("  All turn descriptions already computed!")
    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        sem = asyncio.Semaphore(args.concurrency)
        done = [0]
        t0 = time.time()
        total = len(tasks)

        coros = [
            vision_describe_one(
                client, t["eid"], t["image_path"], t["label"],
                t["direction"], sem, done, total, t0
            )
            for t in tasks
        ]

        # Process results
        new_results = {}
        for result in await asyncio.gather(*coros):
            eid_str = result["eid"]
            label = result["label"]
            if eid_str not in new_results:
                new_results[eid_str] = {}
            new_results[eid_str][label] = result["desc"]

        elapsed = time.time() - t0
        ok = sum(1 for fd in new_results.values() for d in fd.values() if d)
        fail = sum(1 for fd in new_results.values() for d in fd.values() if not d)
        print(f"\n  Phase 1 v150 done in {elapsed/60:.1f}m: "
              f"{ok} good descriptions, {fail} unusable/failed")

        # Merge new turn descriptions into the checkpoint
        for eid_str, frame_descs in new_results.items():
            if eid_str not in merged:
                merged[eid_str] = {}
            for label, desc in frame_descs.items():
                merged[eid_str][label] = desc

    # Save v150 Phase 1 checkpoint
    V150_P1_CKPT.parent.mkdir(parents=True, exist_ok=True)
    with open(V150_P1_CKPT, "w") as f:
        json.dump(merged, f)
    print(f"\n  Saved: {V150_P1_CKPT}")

    # Stats
    total_eps = len(merged)
    turn_descs = sum(
        1 for fd in merged.values()
        for k, v in fd.items()
        if k.startswith("turn_") and v
    )
    print(f"  Episodes: {total_eps}, Turn descriptions: {turn_descs}")

    # Sample quality check — show a few turn descriptions
    print("\n  Sample direction-aware turn descriptions:")
    count = 0
    for eid_str, descs in list(merged.items())[:200]:
        ep = ep_map.get(int(eid_str))
        if not ep:
            continue
        ref_path = ep.get("reference_path", [])
        ep_dir = RF_DIR / f"episode_{int(eid_str):06d}"
        poses_f = ep_dir / "poses.json"
        if not poses_f.exists():
            continue
        try:
            poses = json.load(open(poses_f))
        except Exception:
            continue
        for frame in poses.get("frames", []):
            label = frame["label"]
            if not label.startswith("turn_"):
                continue
            desc = descs.get(label)
            if desc:
                waypoint_idx = frame.get("waypoint_idx", 0)
                direction = get_turn_direction_at_waypoint(ref_path, waypoint_idx) or "?"
                print(f"  ep {eid_str} {label} [wpt={waypoint_idx}, {direction}]: {desc[:115]}")
                count += 1
                if count >= 10:
                    break
        if count >= 10:
            break


if __name__ == "__main__":
    asyncio.run(main())
