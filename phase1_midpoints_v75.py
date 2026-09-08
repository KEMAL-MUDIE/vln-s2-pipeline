#!/usr/bin/env python3
"""
Phase 1 Midpoints v75 — Ranked-priority descriptions for walk-past segments.

Current v22 midpoint checkpoint: 18.5% door/doorframe (too high), 4.3% artwork.
v75 fix: Apply same ranked-priority prompt as v73 for approach-view turn landmarks.
  - Prefer: artwork/decor > distinctive furniture > architecture > doors (last resort)
  - Falls back to v22 checkpoint when v75 fails/rejects

Output: outputs/gate4_v75_midpoints_p1_checkpoint.json
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
P1_V75_CKPT = ROOT / "outputs" / "gate4_v75_midpoints_p1_checkpoint.json"
P1_V22_CKPT = ROOT / "outputs" / "gate4_v22_midpoint_p1_checkpoint.json"

sys.path.insert(0, str(ROOT))
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY

VISION_PROMPT_MID = (
    "A robot is walking past an interesting landmark in this indoor scene. "
    "Identify the single most distinctive object visible in the image.\n\n"
    "Priority order (use the FIRST type you can see clearly):\n"
    "1. ARTWORK/DECOR: painting, poster, mirror, clock, plant, lamp, vase, sculpture\n"
    "2. DISTINCTIVE FURNITURE: sofa, armchair, bookshelf, bed, desk, dining table — include color + material\n"
    "3. ARCHITECTURE: fireplace, staircase, archway, large window — only if clearly distinctive\n"
    "4. APPLIANCE/FIXTURE: refrigerator, stove, sink, bathtub — for kitchens/bathrooms\n"
    "5. LAST RESORT: door or passage feature — only if nothing else clearly visible\n\n"
    "Reply with a 2-5 word noun phrase including color/material. Examples:\n"
    "'large oil painting', 'brown leather sofa', 'marble fireplace', 'stainless refrigerator'\n"
    "Reply with ONLY the noun phrase. No explanations."
)

_REJECT_RE = re.compile(
    r'^(open\s+(?:corridor|space|hallway|area)|plain|featureless|empty|nothing|no\s+landmark|'
    r'no\s+(?:visible|clear|distinctive)\s+\w+|nothing\s+distinctive|'
    r'a\s+room|the\s+room|floor|ceiling|wall(?:s)?|bare\s|neutral|uniform|hallway$|corridor$)',
    re.IGNORECASE
)
_GENERIC = {'room', 'area', 'space', 'corner', 'hallway', 'corridor', 'ceiling', 'floor'}


def clean_desc(raw: str) -> Optional[str]:
    raw = raw.strip().strip('"\'').strip()
    for prefix in ["The object is", "I see", "I can see", "The most prominent",
                   "Based on", "In this image", "The item is", "There is",
                   "The landmark is", "Looking at", "Priority 1:", "Priority 2:",
                   "Priority 3:", "Priority 4:", "Priority 5:", "1.", "2.", "3.", "4.", "5."]:
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
    if last_word in _GENERIC and len(words) <= 2:
        return None
    return raw


def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def describe_midpoint_frame(client, eid: str, label: str, image_path: Path,
                                   sem: asyncio.Semaphore,
                                   done_counter: List, total: int, t0: float) -> Dict:
    async with sem:
        try:
            b64 = image_to_base64(image_path)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": VISION_PROMPT_MID},
                ]
            }]
            resp = await client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                max_tokens=40,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            desc = clean_desc(raw)
            result = {"eid": eid, "label": label, "desc": desc, "raw": raw, "ok": True}
        except Exception as e:
            result = {"eid": eid, "label": label, "desc": None, "raw": "", "ok": False, "error": str(e)}

    done_counter[0] += 1
    if done_counter[0] % 500 == 0 or done_counter[0] == total:
        elapsed = time.time() - t0
        r = done_counter[0] / max(elapsed, 0.001)
        eta = (total - done_counter[0]) / r
        print(f"  [mid-v75 {done_counter[0]}/{total}] {r:.1f}/s  ETA={eta/60:.1f}m", flush=True)
    return result


async def run_midpoint_phase1(episodes: List[Dict], existing: Dict, v22_ck: Dict,
                               concurrency: int = 16) -> Dict:
    print(f"\n=== Phase 1 Midpoints v75 (Ranked Priority) ===")
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
        describe_midpoint_frame(client, t["eid"], t["label"], t["image_path"],
                                 sem, done_counter, len(tasks), t0)
        for t in tasks
    ]

    new_results: Dict[str, Dict[str, Optional[str]]] = {}
    for result in await asyncio.gather(*coros):
        eid, label = result["eid"], result["label"]
        if eid not in new_results:
            new_results[eid] = {}
        new_results[eid][label] = result["desc"]

    elapsed = time.time() - t0
    ok = sum(1 for fd in new_results.values() for d in fd.values() if d)
    fail = sum(1 for fd in new_results.values() for d in fd.values() if not d)
    print(f"\nDone in {elapsed/60:.1f}m: {ok} good ({100*ok/(ok+fail):.1f}%), {fail} rejected")

    # Merge: v75 wins, fall back to v22 for failures
    merged = dict(existing)
    for eid, ep_results in new_results.items():
        if eid not in merged:
            merged[eid] = {}
        v22_ep = v22_ck.get(eid, {})
        for label, desc in ep_results.items():
            v22_d = v22_ep.get(label)
            merged[eid][label] = desc if desc else v22_d

    P1_V75_CKPT.parent.mkdir(parents=True, exist_ok=True)
    with open(P1_V75_CKPT, "w") as f:
        json.dump(merged, f)
    print(f"Checkpoint: {P1_V75_CKPT}")
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
    if args.resume and P1_V75_CKPT.exists():
        existing = json.load(open(P1_V75_CKPT))
    print(f"Existing: {len(existing)} episodes")

    v22_ck = json.load(open(P1_V22_CKPT)) if P1_V22_CKPT.exists() else {}
    print(f"v22 fallback: {sum(len(v) for v in v22_ck.values())} frames")

    results = asyncio.run(run_midpoint_phase1(episodes, existing, v22_ck, args.concurrency))

    total_frames = sum(len(v) for v in results.values())
    good = sum(1 for v in results.values() for d in v.values() if d)
    print(f"\nFinal: {total_frames} frames, {good} good ({100*good/total_frames:.1f}%)")

    # Compare with v22
    if P1_V22_CKPT.exists():
        old = json.load(open(P1_V22_CKPT))
        improve, regress, door_v75, door_v22, art_v75, art_v22 = 0, 0, 0, 0, 0, 0
        for eid in set(results.keys()) & set(old.keys()):
            for label in results[eid]:
                new_d = results[eid].get(label) or ""
                old_d = old[eid].get(label) or ""
                new_good = bool(new_d and len(new_d.split()) >= 2)
                old_good = bool(old_d and len(old_d.split()) >= 2)
                if new_good and not old_good: improve += 1
                elif old_good and not new_good: regress += 1
                if new_d and any(k in new_d.lower() for k in ["door", "doorway", "doorframe"]): door_v75 += 1
                if old_d and any(k in old_d.lower() for k in ["door", "doorway", "doorframe"]): door_v22 += 1
                if new_d and any(k in new_d.lower() for k in ["painting", "picture", "poster", "mirror", "clock", "art"]): art_v75 += 1
                if old_d and any(k in old_d.lower() for k in ["painting", "picture", "poster", "mirror", "clock", "art"]): art_v22 += 1
        print(f"\nVs v22: improve={improve}, regress={regress}")
        total_d = sum(1 for v in results.values() for d in v.values() if d)
        print(f"Door rate: v75={100*door_v75/total_d:.1f}% vs v22={100*door_v22/max(sum(1 for v in old.values() for d in v.values() if d),1):.1f}%")
        print(f"Artwork rate: v75={100*art_v75/total_d:.1f}% vs v22={100*art_v22/max(sum(1 for v in old.values() for d in v.values() if d),1):.1f}%")

    # Sample
    for eid in list(results.keys())[:3]:
        print(f"\nEP{eid}:")
        v22_ep = v22_ck.get(eid, {})
        for label, desc in results[eid].items():
            print(f"  {label}: v75={desc!r} | v22={v22_ep.get(label)!r}")


if __name__ == "__main__":
    main()
