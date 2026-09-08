#!/usr/bin/env python3
"""
Phase 1 Goal Approach — v74 stop improvement.

Key idea: The last midpoint frame (mid_N, from path[-2] facing path[-1]) shows
what's visible when approaching the goal. Use VLM to describe the distinctive
landmark at the goal, enabling better stop phrases for ALL 1839 episodes.

Currently:
  - 224 episodes: generic stop → fixed with perframe gate3 data (v68+)
  - ~724 episodes: stop phrase lacks color/material specificity
  - ~891 episodes: already have vision-grounded stop phrases

v74 goal: improve stop quality for all by using the last midpoint frame.

Output: outputs/gate4_v74_goal_approach_p1_checkpoint.json
  {episode_id: {desc: "...", raw: "..."}}
"""

import asyncio
import base64
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent
MIDPOINTS_DIR = Path("/mnt/nvme0/vln_habitat/habitat_data/midpoint_frames")
GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
P1_GOAL_CKPT = ROOT / "outputs" / "gate4_v74_goal_approach_p1_checkpoint.json"

sys.path.insert(0, str(ROOT))
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY

VISION_PROMPT_GOAL = (
    "A robot is about to stop at its destination. "
    "This image shows what the robot sees just before stopping. "
    "Identify the single most distinctive object or landmark at this destination — "
    "what makes this location unique and recognizable?\n\n"
    "Priority order:\n"
    "1. ARTWORK/DECOR: painting, poster, mirror, clock, plant, lamp — very distinctive\n"
    "2. DISTINCTIVE FURNITURE: sofa, armchair, bed, desk, dining table — include color + material\n"
    "3. ARCHITECTURE: fireplace, staircase, archway, window — only if clearly distinctive\n"
    "4. ROOM/AREA: sink, toilet, bathtub, kitchen counter — only for specific rooms\n\n"
    "Reply with a 2-5 word noun phrase including color/material. Examples:\n"
    "'large oil painting', 'white marble fireplace', 'brown leather sofa', 'porcelain pedestal sink'\n"
    "Reply with ONLY the noun phrase. No explanations."
)

_REJECT_RE = re.compile(
    r'^(open\s+(?:corridor|space|hallway|area)|plain|featureless|empty|nothing|no\s+landmark|'
    r'no\s+(?:visible|clear|distinctive)\s+\w+|nothing\s+distinctive|'
    r'a\s+room|the\s+room|floor|ceiling|wall(?:s)?|bare\s|neutral|uniform|hallway$|corridor$|'
    r'dark\s+hallway|light\s+hallway|white\s+wall$|beige\s+wall$|grey\s+wall$)',
    re.IGNORECASE
)
_GENERIC_STOP = {'door', 'doorway', 'room', 'area', 'space', 'hallway', 'corridor', 'wall', 'ceiling', 'floor'}


def clean_goal_desc(raw: str) -> Optional[str]:
    raw = raw.strip().strip('"\'').strip()
    for prefix in ["The object is", "I see", "I can see", "The most prominent",
                   "Based on", "In this image", "The item is", "There is",
                   "The destination", "The landmark is", "At the destination",
                   "Looking at", "The robot should", "Priority 1:", "Priority 2:",
                   "Priority 3:", "Priority 4:", "1.", "2.", "3.", "4."]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].strip(" :-").strip()
    raw = re.sub(r'^(a|an|the)\s+', '', raw, flags=re.IGNORECASE).strip()
    raw = re.split(r'[,;.\n]', raw)[0].strip()
    words = raw.split()
    if len(words) < 2 or len(words) > 8:
        return None
    raw = ' '.join(words[:7])
    if _REJECT_RE.match(raw):
        return None
    last_word = words[-1].lower().rstrip('s')
    if last_word in _GENERIC_STOP and len(words) <= 2:
        return None
    return raw


def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def describe_goal_frame(client, eid: str, image_path: Path,
                               sem: asyncio.Semaphore,
                               done_counter: List, total: int, t0: float) -> Dict:
    async with sem:
        try:
            b64 = image_to_base64(image_path)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": VISION_PROMPT_GOAL},
                ]
            }]
            resp = await client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                max_tokens=40,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            desc = clean_goal_desc(raw)
            result = {"eid": eid, "desc": desc, "raw": raw, "ok": True}
        except Exception as e:
            result = {"eid": eid, "desc": None, "raw": "", "ok": False, "error": str(e)}

    done_counter[0] += 1
    if done_counter[0] % 200 == 0 or done_counter[0] == total:
        elapsed = time.time() - t0
        r = done_counter[0] / max(elapsed, 0.001)
        eta = (total - done_counter[0]) / r
        print(f"  [goal-p1 {done_counter[0]}/{total}] {r:.1f}/s  ETA={eta/60:.1f}m", flush=True)
    return result


async def run_goal_phase1(episodes: List[Dict], existing: Dict,
                           concurrency: int = 16) -> Dict:
    print(f"\n=== Phase 1 Goal Approach ===")
    print(f"  concurrency={concurrency}, vLLM: {VLLM_BASE_URL}")

    tasks = []
    skipped = 0
    for ep in episodes:
        eid = str(ep["episode_id"])
        ep_dir = MIDPOINTS_DIR / f"episode_{int(eid):06d}"
        mid_f = ep_dir / "midpoints.json"
        if not mid_f.exists():
            continue
        if eid in existing:
            skipped += 1
            continue
        try:
            mid_data = json.load(open(mid_f))
        except Exception:
            continue
        frames = mid_data.get("frames", [])
        if not frames:
            continue
        # Use the LAST midpoint frame (approach to goal)
        last_frame = frames[-1]
        img_path = ep_dir / last_frame["path"]
        if not img_path.exists():
            continue
        tasks.append({"eid": eid, "image_path": img_path})

    print(f"  Episode tasks: {len(tasks)}  Skipped (checkpoint): {skipped}")
    if not tasks:
        print("  Nothing to process.")
        return existing

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    sem = asyncio.Semaphore(concurrency)
    done_counter = [0]
    t0 = time.time()

    coros = [
        describe_goal_frame(client, t["eid"], t["image_path"], sem,
                            done_counter, len(tasks), t0)
        for t in tasks
    ]

    new_results: Dict[str, Dict] = {}
    for result in await asyncio.gather(*coros):
        new_results[result["eid"]] = {"desc": result["desc"], "raw": result["raw"]}

    elapsed = time.time() - t0
    ok = sum(1 for v in new_results.values() if v["desc"])
    fail = sum(1 for v in new_results.values() if not v["desc"])
    print(f"\nDone in {elapsed/60:.1f}m: {ok} good ({100*ok/(ok+fail):.1f}%), {fail} rejected/failed")

    merged = dict(existing)
    merged.update(new_results)

    P1_GOAL_CKPT.parent.mkdir(parents=True, exist_ok=True)
    with open(P1_GOAL_CKPT, "w") as f:
        json.dump(merged, f)
    print(f"Checkpoint: {P1_GOAL_CKPT}")
    return merged


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with gzip.open(GT_PATH, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]

    existing = {}
    if args.resume and P1_GOAL_CKPT.exists():
        existing = json.load(open(P1_GOAL_CKPT))
    print(f"Existing: {len(existing)} episodes")

    results = asyncio.run(run_goal_phase1(episodes, existing, args.concurrency))

    total_ep = len(results)
    good = sum(1 for v in results.values() if v.get("desc"))
    print(f"\nFinal: {total_ep} episodes, {good} with goal descriptions ({100*good/total_ep:.1f}%)")

    for eid in list(results.keys())[:5]:
        print(f"  EP{eid}: {results[eid].get('desc')!r}")


if __name__ == "__main__":
    main()
