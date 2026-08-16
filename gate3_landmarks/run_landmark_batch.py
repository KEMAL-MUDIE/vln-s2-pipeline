#!/usr/bin/env python3
"""
Gate 3: Batch Landmark Detector
Uses Gemma 4 31B AWQ vision API to detect room types and landmarks from rendered key frames.

For each episode with rendered frames, sends the key frame images to Gemma and extracts:
  - room_type: "bedroom", "living room", "kitchen", "hallway", etc.
  - landmarks: ["gray couch", "dining table", "stairs", ...]
  - direction_hint: "near the window", "past the doorway", etc.

Output: outputs/gate3_landmarks/episode_{id:06d}.json for each episode
        outputs/gate3_landmarks/all_landmarks.json (merged summary)

Usage:
  python3 gate3_landmarks/run_landmark_batch.py \
      --frames-dir outputs/rendered_frames \
      --n-episodes 50           # optional: limit
"""
import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY

FRAMES_DIR = ROOT / "outputs" / "rendered_frames"
OUT_DIR = ROOT / "outputs" / "gate3_landmarks"
CONCURRENCY = 12   # lower than gate4 (vision requests are heavier)

LANDMARK_PROMPT = """You are a precise visual scene analyzer for indoor navigation.
Look at the provided image(s) from a robot navigating indoors.

Respond in EXACTLY this JSON format (no markdown, no extra text):
{
  "room_type": "single room type (e.g. bedroom, living room, hallway, kitchen, dining room, bathroom, office, stairs)",
  "landmarks": ["list", "of", "specific", "visible", "objects"],
  "stop_landmark": "the most prominent object a person would stop near at this location",
  "direction_hint": "brief spatial description (e.g. near the window, past the doorway)",
  "confidence": "high/medium/low"
}

Be specific: name actual objects (gray sofa, wooden dining table, marble countertop), not generic terms."""

GOAL_LANDMARK_PROMPT = """You are a precise visual scene analyzer for indoor navigation.
Look at the provided image. This is the GOAL/DESTINATION of a navigation path.

Respond in EXACTLY this JSON format (no markdown, no extra text):
{
  "room_type": "room type at the destination",
  "stop_landmark": "the most specific and prominent object to stop near (this is what the navigator aims for)",
  "landmarks": ["other", "visible", "objects", "nearby"],
  "direction_hint": "brief description of where exactly to stop"
}

Be specific and actionable — this landmark description guides where to stop."""


def encode_image_b64(img_path: Path) -> str:
    """Encode image file to base64 string."""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_episode_frames(episode_dir: Path) -> Optional[Dict]:
    """Load poses.json and return frames metadata."""
    poses_file = episode_dir / "poses.json"
    if not poses_file.exists():
        return None
    with open(poses_file) as f:
        return json.load(f)


def build_landmark_messages(episode_dir: Path, frames_meta: Dict) -> List[Dict]:
    """
    Build Gemma messages for landmark detection.
    Sends start + goal frames (+ any turn frames) with targeted prompts.
    """
    frames = frames_meta.get("frames", [])
    if not frames:
        return []

    # Split into start frames, turn frames, goal frame
    start_frames = [f for f in frames if f["label"] == "start"]
    turn_frames = [f for f in frames if f["label"].startswith("turn_")]
    goal_frames = [f for f in frames if f["label"] == "goal"]

    messages = []

    # Start + turns → general scene context
    context_frames = (start_frames + turn_frames)[:3]  # max 3 context frames
    if context_frames:
        content = []
        for frame in context_frames:
            img_path = episode_dir / frame["path"]
            if img_path.exists():
                b64 = encode_image_b64(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
        content.append({"type": "text", "text": LANDMARK_PROMPT})
        messages.append({"role": "user", "content": content})

    # Goal frame → stop landmark
    if goal_frames:
        goal_path = episode_dir / goal_frames[0]["path"]
        if goal_path.exists():
            b64 = encode_image_b64(goal_path)
            goal_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": GOAL_LANDMARK_PROMPT},
            ]
            messages.append({"role": "user", "content": goal_content})

    return messages


async def detect_landmarks_for_episode(
    client,
    episode_id: int,
    episode_dir: Path,
    frames_meta: Dict,
    out_dir: Path,
) -> Optional[Dict]:
    """Run landmark detection for one episode. Returns combined result."""
    out_file = out_dir / f"episode_{episode_id:06d}.json"
    if out_file.exists():
        with open(out_file) as f:
            return json.load(f)

    frames = frames_meta.get("frames", [])
    if not frames:
        return None

    # Build two separate calls: scene context + goal landmark
    start_frames = [f for f in frames if f["label"] == "start"]
    turn_frames = [f for f in frames if f["label"].startswith("turn_")]
    goal_frames = [f for f in frames if f["label"] == "goal"]

    result = {
        "episode_id": episode_id,
        "scene_id": frames_meta.get("scene_id", ""),
        "n_frames": len(frames),
        "scene_context": None,
        "goal_landmark": None,
    }

    # ── Scene context detection (start + turns) ─────────────────────────────
    context_frames = (start_frames + turn_frames)[:3]
    if context_frames:
        content = []
        for frame in context_frames:
            img_path = episode_dir / frame["path"]
            if img_path.exists():
                b64 = encode_image_b64(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
        content.append({"type": "text", "text": LANDMARK_PROMPT})

        try:
            resp = await client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=256,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            # Parse JSON (handle markdown code blocks)
            if "```" in raw:
                raw = raw.split("```")[1].strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            result["scene_context"] = json.loads(raw)
        except Exception as e:
            result["scene_context"] = {"error": str(e)}

    # ── Goal landmark detection ──────────────────────────────────────────────
    if goal_frames:
        goal_path = episode_dir / goal_frames[0]["path"]
        if goal_path.exists():
            b64 = encode_image_b64(goal_path)
            goal_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": GOAL_LANDMARK_PROMPT},
            ]
            try:
                resp = await client.chat.completions.create(
                    model=VLLM_MODEL,
                    messages=[{"role": "user", "content": goal_content}],
                    max_tokens=128,
                    temperature=0.1,
                )
                raw = resp.choices[0].message.content.strip()
                if "```" in raw:
                    raw = raw.split("```")[1].strip()
                    if raw.startswith("json"):
                        raw = raw[4:].strip()
                result["goal_landmark"] = json.loads(raw)
            except Exception as e:
                result["goal_landmark"] = {"error": str(e)}

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


async def run_batch(frames_dir: Path, out_dir: Path, n_episodes: Optional[int] = None):
    """Run landmark detection on all rendered episodes."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("ERROR: openai not installed. Run: pip install openai")
        sys.exit(1)

    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

    # Find all rendered episodes
    episode_dirs = sorted(frames_dir.glob("episode_*"))
    if n_episodes:
        episode_dirs = episode_dirs[:n_episodes]

    # Filter to those with renders
    valid = []
    for ep_dir in episode_dirs:
        meta = load_episode_frames(ep_dir)
        if meta and meta.get("n_frames", 0) > 0:
            valid.append((meta["episode_id"], ep_dir, meta))

    print(f"=== Gate 3: Landmark Detector ===")
    print(f"  Model:    Gemma 4 31B AWQ (vision) @ {VLLM_BASE_URL}")
    print(f"  Episodes: {len(valid)} rendered")
    print(f"  Output:   {out_dir}")
    print()

    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0
    errors = 0
    t0 = time.time()

    async def throttled(ep_id, ep_dir, meta):
        async with sem:
            return await detect_landmarks_for_episode(client, ep_id, ep_dir, meta, out_dir)

    tasks = [throttled(ep_id, ep_dir, meta) for ep_id, ep_dir, meta in valid]
    total = len(tasks)

    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            has_error = (
                isinstance(result.get("scene_context"), dict) and "error" in result.get("scene_context", {})
            ) or (
                isinstance(result.get("goal_landmark"), dict) and "error" in result.get("goal_landmark", {})
            )
            if has_error:
                errors += 1
            done += 1
        else:
            errors += 1
            done += 1

        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else 0
        if done % 50 == 0 or done == total:
            print(f"  [{done}/{total}] ok={done-errors} err={errors} rate={rate:.1f}/s ETA={eta/60:.1f}m")

    # Save merged summary
    all_results = []
    for f in sorted(out_dir.glob("episode_*.json")):
        with open(f) as fh:
            all_results.append(json.load(fh))

    summary_path = out_dir / "all_landmarks.json"
    with open(summary_path, "w") as f:
        json.dump({"n_episodes": len(all_results), "results": all_results}, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n=== Done: {done}/{total} in {elapsed:.1f}s ({done/elapsed:.1f}/s) ===")
    print(f"Saved: {out_dir}/all_landmarks.json ({len(all_results)} entries)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", default=str(FRAMES_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--n-episodes", type=int, default=None)
    args = p.parse_args()
    asyncio.run(run_batch(Path(args.frames_dir), Path(args.out_dir), args.n_episodes))


if __name__ == "__main__":
    main()
