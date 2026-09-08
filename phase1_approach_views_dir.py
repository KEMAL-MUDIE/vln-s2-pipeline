#!/usr/bin/env python3
"""
Phase 1 Approach Views WITH DIRECTION — v72 improvement.

Key change from phase1_approach_views.py:
  The VLM prompt now tells the model WHICH WAY the robot will turn (left/right).
  This focuses the VLM on the correct side of the image.

Motivation (from v62 partial analysis):
  - Approach-view renders from path[i-1] facing path[i]: landmark is ahead-left or ahead-right
  - Without knowing the turn direction, VLM sometimes picks the wrong side
  - Direction-aware prompt: "Robot will turn LEFT — look for landmark ahead and SLIGHTLY LEFT"

Output: outputs/gate4_v72_approach_views_dir_p1_checkpoint.json
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
P1_AV_DIR_CKPT = ROOT / "outputs" / "gate4_v72_approach_views_dir_p1_checkpoint.json"

sys.path.insert(0, str(ROOT))
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY

VISION_PROMPT_TURN_DIR = {
    "left": (
        "A robot is navigating indoors and will turn LEFT at a nearby landmark. "
        "This image shows what the robot sees JUST BEFORE the left turn — "
        "the turning landmark should be visible ahead and slightly to the LEFT. "
        "Identify the single most distinctive object or feature on the LEFT SIDE of the image "
        "that the robot should use to know WHEN to turn left. "
        "Focus on: furniture (sofa, table, chair, bookshelf), "
        "architecture (archway, fireplace, pillar, staircase), "
        "or distinctive decor (clock, mirror, painting, plant). "
        "Reply with a noun phrase (2-5 words) including color/material: "
        "e.g. 'dark wooden bookshelf', 'marble fireplace', 'large framed painting'. "
        "If nothing distinctive on the left, describe the most prominent object ahead. "
        "Reply with ONLY the noun phrase."
    ),
    "right": (
        "A robot is navigating indoors and will turn RIGHT at a nearby landmark. "
        "This image shows what the robot sees JUST BEFORE the right turn — "
        "the turning landmark should be visible ahead and slightly to the RIGHT. "
        "Identify the single most distinctive object or feature on the RIGHT SIDE of the image "
        "that the robot should use to know WHEN to turn right. "
        "Focus on: furniture (sofa, table, chair, bookshelf), "
        "architecture (archway, fireplace, pillar, staircase), "
        "or distinctive decor (clock, mirror, painting, plant). "
        "Reply with a noun phrase (2-5 words) including color/material: "
        "e.g. 'dark wooden bookshelf', 'marble fireplace', 'large framed painting'. "
        "If nothing distinctive on the right, describe the most prominent object ahead. "
        "Reply with ONLY the noun phrase."
    ),
}
VISION_PROMPT_DEFAULT = VISION_PROMPT_TURN_DIR["left"]

_REJECT_RE = re.compile(
    r'^(open\s+(?:corridor|space|hallway|area)|plain|featureless|empty|nothing|no\s+furniture|'
    r'a\s+room|the\s+room|floor|ceiling|wall(?:s)?|bare\s|neutral|uniform|hallway$|corridor$)',
    re.IGNORECASE
)
_GENERIC_LANDMARKS = {
    'door', 'doorway', 'opening', 'passage', 'wall', 'floor', 'ceiling',
    'room', 'area', 'space', 'corner', 'hallway', 'corridor',
}


def clean_desc(raw: str) -> Optional[str]:
    raw = raw.strip().strip('"\'').strip()
    for prefix in ["The object is", "I see", "I can see", "The most prominent",
                   "Based on", "In this image", "The item is", "There is",
                   "The landmark is", "The turning landmark is", "On the left",
                   "On the right", "Looking at", "The robot should"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].strip(" :-").strip()
    raw = re.sub(r'^(a|an|the)\s+', '', raw, flags=re.IGNORECASE).strip()
    raw = re.split(r'[,;]', raw)[0].strip()
    words = raw.split()
    if len(words) < 2 or len(words) > 7:
        return None
    raw = ' '.join(words[:6])
    if _REJECT_RE.match(raw):
        return None
    last_word = words[-1].lower().rstrip('s')
    if last_word in _GENERIC_LANDMARKS and len(words) <= 2:
        return None
    return raw


def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def describe_approach_frame(client, eid: str, label: str, image_path: Path,
                                   turn_dir: str, sem: asyncio.Semaphore,
                                   done_counter: List, total: int, t0: float) -> Dict:
    async with sem:
        try:
            b64 = image_to_base64(image_path)
            prompt = VISION_PROMPT_TURN_DIR.get(turn_dir, VISION_PROMPT_DEFAULT)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ]
            }]
            resp = await client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                max_tokens=30,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            desc = clean_desc(raw)
            result = {"eid": eid, "label": label, "desc": desc, "raw": raw, "turn_dir": turn_dir, "ok": True}
        except Exception as e:
            result = {"eid": eid, "label": label, "desc": None, "raw": "", "turn_dir": turn_dir, "ok": False, "error": str(e)}

    done_counter[0] += 1
    if done_counter[0] % 500 == 0 or done_counter[0] == total:
        elapsed = time.time() - t0
        r = done_counter[0] / max(elapsed, 0.001)
        eta = (total - done_counter[0]) / r
        print(f"  [Phase1-AV-dir {done_counter[0]}/{total}] {r:.1f}/s  ETA={eta/60:.1f}m", flush=True)
    return result


async def run_phase1_av_dir(episodes: List[Dict], existing: Dict, concurrency: int = 16) -> Dict:
    print(f"\n=== Phase 1 Approach Views (Direction-Aware) ===")
    print(f"  concurrency={concurrency}, vLLM: {VLLM_BASE_URL}")

    tasks = []
    skipped = 0
    for ep in episodes:
        eid = str(ep["episode_id"])
        ep_dir = MIDPOINTS_DIR / f"episode_{int(eid):06d}"
        av_f = ep_dir / "approach_views.json"
        if not av_f.exists():
            continue
        if eid in existing:
            skipped += 1
            continue
        try:
            av_data = json.load(open(av_f))
        except Exception:
            continue
        for frame in av_data.get("frames", []):
            img_path = ep_dir / frame["path"]
            if not img_path.exists():
                continue
            turn_dir = frame.get("turn_dir", "left")
            tasks.append({"eid": eid, "label": frame["label"], "image_path": img_path, "turn_dir": turn_dir})

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
        describe_approach_frame(client, t["eid"], t["label"], t["image_path"],
                                 t["turn_dir"], sem, done_counter, len(tasks), t0)
        for t in tasks
    ]

    new_results: Dict[str, Dict[str, Optional[str]]] = {}
    for result in await asyncio.gather(*coros):
        eid, label = result["eid"], result["label"]
        if eid not in new_results:
            new_results[eid] = {}
        new_results[eid][label] = result["desc"]

    merged = dict(existing)
    merged.update(new_results)

    elapsed = time.time() - t0
    ok = sum(1 for fd in new_results.values() for d in fd.values() if d)
    fail = sum(1 for fd in new_results.values() for d in fd.values() if not d)
    print(f"\nDone in {elapsed/60:.1f}m: {ok} good, {fail} rejected/failed, {skipped} skipped")

    P1_AV_DIR_CKPT.parent.mkdir(parents=True, exist_ok=True)
    with open(P1_AV_DIR_CKPT, "w") as f:
        json.dump(merged, f)
    print(f"Checkpoint: {P1_AV_DIR_CKPT}")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with gzip.open(GT_PATH, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]

    existing = {}
    if args.resume and P1_AV_DIR_CKPT.exists():
        existing = json.load(open(P1_AV_DIR_CKPT))
    print(f"Existing checkpoint: {len(existing)} episodes")

    results = asyncio.run(run_phase1_av_dir(episodes, existing, args.concurrency))

    total_ep = len(results)
    total_frames = sum(len(v) for v in results.values())
    good = sum(1 for v in results.values() for d in v.values() if d)
    print(f"\nFinal: {total_ep} episodes, {total_frames} frames, {good} useful descriptions")
    print(f"Good rate: {100*good/max(total_frames,1):.1f}%")

    # Compare with non-directional checkpoint
    old_path = ROOT / "outputs" / "gate4_v70_approach_views_p1_checkpoint.json"
    if old_path.exists():
        old = json.load(open(old_path))
        improve, regress, same = 0, 0, 0
        for eid in set(results.keys()) & set(old.keys()):
            for label in results[eid]:
                new_d = results[eid].get(label)
                old_d = old[eid].get(label) if eid in old else None
                new_good = bool(new_d and len(new_d.split()) >= 2)
                old_good = bool(old_d and len(old_d.split()) >= 2)
                if new_good and not old_good: improve += 1
                elif old_good and not new_good: regress += 1
                else: same += 1
        print(f"\nVs non-directional (v70): improve={improve}, regress={regress}, same={same}")

    for eid in list(results.keys())[:3]:
        print(f"\nEP{eid}:")
        for label, desc in results[eid].items():
            print(f"  {label}: {desc!r}")


if __name__ == "__main__":
    main()
