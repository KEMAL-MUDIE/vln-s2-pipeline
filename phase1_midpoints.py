#!/usr/bin/env python3
"""
Phase 1 Midpoints — Visual description of intermediate GT waypoints.

Reads midpoint_frames/episode_XXXXXX/midpoints.json (rendered by render_midpoints.py)
and sends each frame to vLLM for a brief "walking landmark" description.

Output: outputs/gate4_v22_midpoint_p1_checkpoint.json
  {
    "1": {"mid_1": "dark wooden dining table with chairs", "mid_2": "grey sofa with cushions", ...},
    "2": {"mid_1": "...", ...},
    ...
  }

Usage:
  python3 phase1_midpoints.py [--concurrency 8]
"""

import argparse
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
P1_MID_CKPT = ROOT / "outputs" / "gate4_v22_midpoint_p1_checkpoint.json"

sys.path.insert(0, str(ROOT))
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY

VISION_PROMPT_MIDPOINT = (
    "A robot navigating indoors has just stepped to this position. "
    "Identify the single most prominent furniture item or object visible directly ahead. "
    "Reply with a noun phrase only (2-5 words), e.g.: 'dark wooden dining table', "
    "'grey upholstered sofa', 'white marble fireplace', 'tall wooden bookshelf'. "
    "If only walls or floors are visible with no distinct furniture, reply: 'open space'. "
    "Reply with ONLY the noun phrase, nothing else."
)

# Generic/useless responses to reject
_REJECT_RE = re.compile(
    r'^(open\s+space|plain|featureless|empty|nothing|no\s+furniture|a\s+room|the\s+room|'
    r'floor|ceiling|wall|walls|corridor|hallway without|bare\s)',
    re.IGNORECASE
)


def clean_midpoint_desc(raw: str) -> Optional[str]:
    raw = raw.strip().strip('"\'').strip()
    for prefix in ["The object is", "I see", "I can see", "The most prominent",
                   "Based on", "In this image", "The item is", "There is"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].strip(" :-").strip()
    # Remove article at start
    raw = re.sub(r'^(a|an|the)\s+', '', raw, flags=re.IGNORECASE).strip()
    # Take only first phrase (before comma/semicolon)
    raw = re.split(r'[,;]', raw)[0].strip()
    # Limit words
    words = raw.split()
    if len(words) < 2 or len(words) > 8:
        return None
    raw = ' '.join(words[:6])
    if _REJECT_RE.match(raw):
        return None
    return raw


def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def describe_midpoint(client, eid: str, label: str, image_path: Path,
                             sem: asyncio.Semaphore, done_counter: List, total: int, t0: float) -> Dict:
    async with sem:
        try:
            b64 = image_to_base64(image_path)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": VISION_PROMPT_MIDPOINT},
                ]
            }]
            resp = await client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                max_tokens=30,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            desc = clean_midpoint_desc(raw)
            result = {"eid": eid, "label": label, "desc": desc, "raw": raw, "ok": True}
        except Exception as e:
            result = {"eid": eid, "label": label, "desc": None, "raw": "", "ok": False, "error": str(e)}

    done_counter[0] += 1
    if done_counter[0] % 500 == 0 or done_counter[0] == total:
        elapsed = time.time() - t0
        r = done_counter[0] / max(elapsed, 0.001)
        eta = (total - done_counter[0]) / r
        print(f"  [Phase1-Mid {done_counter[0]}/{total}] {r:.1f}/s  ETA={eta/60:.1f}m", flush=True)
    return result


async def run_phase1_midpoints(episodes: List[Dict], existing: Dict, concurrency: int = 8) -> Dict:
    print(f"\n=== Phase 1 Midpoints ===")
    print(f"  concurrency={concurrency}, vLLM: {VLLM_BASE_URL}")

    tasks = []
    skipped = 0
    for ep in episodes:
        eid = str(ep["episode_id"])
        ep_dir = MIDPOINTS_DIR / f"episode_{int(eid):06d}"
        midpoints_f = ep_dir / "midpoints.json"
        if not midpoints_f.exists():
            continue
        if eid in existing:
            skipped += 1
            continue
        try:
            mid_data = json.load(open(midpoints_f))
        except Exception:
            continue
        for frame in mid_data.get("frames", []):
            img_path = ep_dir / frame["path"]
            if not img_path.exists():
                continue
            tasks.append({"eid": eid, "label": frame["label"], "image_path": img_path})

    print(f"  Frame tasks: {len(tasks)}  Skipped (checkpoint): {skipped}")
    if not tasks:
        print("  Nothing to process.")
        return existing

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    sem = asyncio.Semaphore(concurrency)
    done_counter = [0]
    t0 = time.time()

    coros = [
        describe_midpoint(client, t["eid"], t["label"], t["image_path"],
                          sem, done_counter, len(tasks), t0)
        for t in tasks
    ]

    new_results: Dict[str, Dict[str, Optional[str]]] = {}
    for result in await asyncio.gather(*coros):
        eid, label = result["eid"], result["label"]
        if eid not in new_results:
            new_results[eid] = {}
        new_results[eid][label] = result["desc"]

    merged = dict(existing)
    for eid, frame_descs in new_results.items():
        merged[eid] = frame_descs

    elapsed = time.time() - t0
    ok = sum(1 for fd in new_results.values() for d in fd.values() if d)
    fail = sum(1 for fd in new_results.values() for d in fd.values() if not d)
    print(f"\nPhase 1 Midpoints done in {elapsed/60:.1f}m: "
          f"{ok} good, {fail} rejected/failed, {skipped} skipped")

    P1_MID_CKPT.parent.mkdir(parents=True, exist_ok=True)
    with open(P1_MID_CKPT, "w") as f:
        json.dump(merged, f)
    print(f"Checkpoint: {P1_MID_CKPT}")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    # Load GT
    with gzip.open(GT_PATH, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]

    # Load existing checkpoint
    existing = {}
    if P1_MID_CKPT.exists():
        existing = json.load(open(P1_MID_CKPT))
    print(f"Existing checkpoint: {len(existing)} episodes")

    results = asyncio.run(run_phase1_midpoints(episodes, existing, args.concurrency))

    # Summary
    total_ep = len(results)
    total_frames = sum(len(v) for v in results.values())
    good = sum(1 for v in results.values() for d in v.values() if d)
    print(f"\nFinal: {total_ep} episodes, {total_frames} frames, {good} useful descriptions")
    print(f"Good rate: {100*good/max(total_frames,1):.1f}%")

    # Show 3 sample episodes
    for eid in list(results.keys())[:3]:
        print(f"\nEP{eid}:")
        for label, desc in results[eid].items():
            print(f"  {label}: {desc!r}")


if __name__ == "__main__":
    main()
