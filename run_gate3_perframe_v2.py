#!/usr/bin/env python3
"""
Gate 3 v2 — Per-Frame Turn Landmark Detection

ROOT CAUSE FIX for 25pp SR gap:
  Gate 3 v1 sends start + turn frames together → gets generic scene description
  Gate 3 v2 detects EACH turn frame SEPARATELY → gets specific "turn cue" landmark

This gives Gate 4 v11 the per-turn landmark vocab to generate GT-style instructions:
  GT pattern: "turn left at the refrigerator" / "walk past the television"
  Gen v2:     "turn left" (no visual anchor → model can't verify position)
  Gen v11:    "turn left at the [TURN_LANDMARK]" (matches GT training pattern)

Output: outputs/gate3_perframe/episode_XXXXXX.json
Format:
  {episode_id, scene_id,
   start: {room, landmarks, main_landmark},
   turns: [{label, direction, room, main_landmark, landmarks}...],
   goal: {room, stop_landmark, landmarks}}

Usage:
  source /home/kemal/VLNav/vlnav_env/bin/activate
  python3 run_gate3_perframe_v2.py [--n-episodes N] [--concurrency C]
"""
import argparse
import asyncio
import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY

FRAMES_DIR = ROOT / "outputs" / "rendered_frames"
OUT_DIR    = ROOT / "outputs" / "gate3_perframe"
CONCURRENCY = 10  # vision calls are heavy; 10 is stable

# ── Prompts ───────────────────────────────────────────────────────────────────

START_PROMPT = """\
You are analyzing a starting position for indoor robot navigation.
Look at this image. The robot begins its path here.

Respond in EXACTLY this JSON format (no markdown, no extra text):
{
  "room": "room type (bedroom/living room/kitchen/hallway/dining room/bathroom/office/stairs/patio/garage)",
  "main_landmark": "the single most distinctive object visible — the one a person would say 'start near the [X]'",
  "landmarks": ["2-4 other visible objects that could serve as navigation references"]
}

Rules: Be specific (\"gray sectional sofa\" not \"sofa\", \"wooden dining table\" not \"table\").\
 Avoid generic terms like wall/floor/ceiling/window."""

TURN_PROMPT = """\
You are identifying a visual navigation cue for an indoor robot turn decision.
Look at this image. The robot will turn at THIS exact location.

The critical question: What single object would tell you "this is where to turn"?

Respond in EXACTLY this JSON format (no markdown, no extra text):
{
  "room": "room type at this turn point",
  "main_landmark": "THE most distinctive single object visible at this turn — this is the 'turn cue' (e.g. refrigerator, wooden dresser, gray couch, glass dining table)",
  "landmarks": ["1-3 other visible objects that could also serve as turn cues"],
  "room_transition": "describe any room transition visible (e.g. 'entering living room', 'passing through doorway', 'none visible')"
}

Rules:
- main_landmark must be a SPECIFIC object a navigator would describe ("wooden bookcase" not "furniture")
- Prefer objects that are distinctive in the frame, not generic (not wall/floor/ceiling)
- If a doorway or room boundary is visible, include it"""

GOAL_PROMPT = """\
You are identifying the navigation destination for an indoor robot.
Look at this image. This is the GOAL/ENDPOINT of the navigation path.

Respond in EXACTLY this JSON format (no markdown, no extra text):
{
  "room": "room type at the destination",
  "stop_landmark": "the most prominent and specific object to stop near — this is what the instruction says 'stop near the [X]'",
  "landmarks": ["2-3 other visible objects nearby"],
  "stop_description": "brief spatial description of the stop location (e.g. 'in front of the fireplace', 'near the window')"
}

Rules: Be very specific — this landmark is what the navigator aims for at the end."""


def encode_b64(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_json_response(raw: str) -> Dict:
    """Parse JSON from LLM response, handling markdown fences."""
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                raw = p
                break
    # Find first { and last }
    start = raw.find("{")
    end   = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end+1]
    try:
        return json.loads(raw)
    except Exception:
        # Fallback: try to extract fields
        result = {}
        for field in ["room", "main_landmark", "stop_landmark"]:
            m = re.search(rf'"{field}"\s*:\s*"([^"]+)"', raw)
            if m:
                result[field] = m.group(1)
        m = re.search(r'"landmarks"\s*:\s*\[([^\]]+)\]', raw)
        if m:
            items = re.findall(r'"([^"]+)"', m.group(1))
            result["landmarks"] = items
        return result if result else {"error": "parse_failed", "raw": raw[:200]}


async def detect_frame(client, episode_dir: Path, frame_path: str, prompt: str) -> Dict:
    """Run a single vision detection call on one frame."""
    img_path = episode_dir / frame_path
    if not img_path.exists():
        return {"error": f"image not found: frame_path"}
    b64 = encode_b64(img_path)
    ext = img_path.suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=256,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            return parse_json_response(raw)
        except Exception as e:
            if attempt == 2:
                return {"error": str(e)}
            await asyncio.sleep(0.5 * (attempt + 1))
    return {"error": "max_retries"}


async def process_episode(
    client,
    sem: asyncio.Semaphore,
    ep_dir: Path,
    out_dir: Path,
) -> Optional[Dict]:
    """Process one episode: detect landmarks at each frame separately."""
    poses_path = ep_dir / "poses.json"
    if not poses_path.exists():
        return None

    poses = json.load(open(poses_path))
    ep_id = poses["episode_id"]
    out_file = out_dir / f"episode_{ep_id:06d}.json"
    if out_file.exists():
        return json.load(open(out_file))  # already done

    frames = poses.get("frames", [])
    if not frames:
        return None

    start_frames = [f for f in frames if f["label"] == "start"]
    turn_frames  = sorted([f for f in frames if f["label"].startswith("turn_")],
                           key=lambda x: x["label"])
    goal_frames  = [f for f in frames if f["label"] == "goal"]

    result = {
        "episode_id": ep_id,
        "scene_id":   poses.get("scene_id", ""),
        "n_frames":   len(frames),
        "start":      None,
        "turns":      [],
        "goal":       None,
    }

    async with sem:
        # Process start frame
        if start_frames:
            det = await detect_frame(client, ep_dir, start_frames[0]["path"], START_PROMPT)
            result["start"] = det

        # Process each turn frame SEPARATELY (the key innovation)
        for tf in turn_frames:
            det = await detect_frame(client, ep_dir, tf["path"], TURN_PROMPT)
            result["turns"].append({
                "label":     tf["label"],
                "frame_idx": tf["frame_idx"],
                **det,
            })

        # Process goal frame
        if goal_frames:
            det = await detect_frame(client, ep_dir, goal_frames[0]["path"], GOAL_PROMPT)
            result["goal"] = det

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    return result


async def run_batch(frames_dir: Path, out_dir: Path, n_episodes: Optional[int], concurrency: int):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    sem = asyncio.Semaphore(concurrency)

    ep_dirs = sorted(frames_dir.glob("episode_*"))
    if n_episodes:
        ep_dirs = ep_dirs[:n_episodes]

    # Skip already completed
    pending = []
    already_done = 0
    for ep_dir in ep_dirs:
        poses_path = ep_dir / "poses.json"
        if not poses_path.exists():
            continue
        ep_id = json.load(open(poses_path)).get("episode_id", 0)
        out_file = out_dir / f"episode_{ep_id:06d}.json"
        if out_file.exists():
            already_done += 1
        else:
            pending.append(ep_dir)

    total = len(pending) + already_done
    print(f"=== Gate 3 v2: Per-Frame Turn Landmark Detection ===")
    print(f"  Model:     {VLLM_MODEL}")
    print(f"  Server:    {VLLM_BASE_URL}")
    print(f"  Episodes:  {total} total, {already_done} cached, {len(pending)} pending")
    print(f"  Output:    {out_dir}")
    print(f"  Conc:      {concurrency}")
    print()

    if not pending:
        print("All episodes already processed!")
        return

    t0 = time.time()
    done = [0]
    errors = [0]

    async def run_one(ep_dir):
        try:
            r = await process_episode(client, sem, ep_dir, out_dir)
            if r and "error" not in r.get("start", {}):
                done[0] += 1
            else:
                errors[0] += 1
                done[0] += 1
        except Exception as e:
            errors[0] += 1
            done[0] += 1
            print(f"  ERROR {ep_dir.name}: {e}", flush=True)
        if (done[0] + already_done) % 100 == 0 or done[0] == len(pending):
            elapsed = time.time() - t0
            rate = done[0] / elapsed if elapsed > 0 else 0.001
            eta = (len(pending) - done[0]) / rate if rate > 0 else 0
            print(f"  [{done[0]+already_done}/{total}] done={done[0]} err={errors[0]} "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}m", flush=True)

    await asyncio.gather(*(run_one(ep_dir) for ep_dir in pending))

    elapsed = time.time() - t0
    print(f"\n=== Done: {done[0]+already_done}/{total} in {elapsed:.1f}s ===")
    print(f"Errors: {errors[0]}")

    # Quick quality check on sample
    sample_files = sorted(out_dir.glob("episode_*.json"))[:5]
    print("\nSamples:")
    for sf in sample_files:
        d = json.load(open(sf))
        turns_info = [(t.get("label"), t.get("main_landmark", "?")) for t in d.get("turns", [])]
        print(f"  ep {d['episode_id']:6d}: start={d.get('start',{}).get('main_landmark','?')!r:25s} "
              f"turns={turns_info} goal={d.get('goal',{}).get('stop_landmark','?')!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--frames-dir", default=str(FRAMES_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    asyncio.run(run_batch(
        Path(args.frames_dir), Path(args.out_dir),
        args.n_episodes, args.concurrency
    ))


if __name__ == "__main__":
    main()
