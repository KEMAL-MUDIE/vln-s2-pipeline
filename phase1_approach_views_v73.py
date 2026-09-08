#!/usr/bin/env python3
"""
Phase 1 Approach Views v73 — Improved landmark specificity.

Key improvements over v72 (phase1_approach_views_dir.py):
  1. Ranked priority prompt: prefers distinctive objects over generic architecture
  2. Relaxed word limit: 2-8 words (was 2-7) to reduce false rejections
  3. Stricter generic rejection: doors/walls/floors/ceilings auto-rejected
  4. Falls back to v72 checkpoint for anything v73 can't improve

Output: outputs/gate4_v73_approach_views_p1_checkpoint.json
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
P1_V73_CKPT = ROOT / "outputs" / "gate4_v73_approach_views_p1_checkpoint.json"
P1_V72_CKPT = ROOT / "outputs" / "gate4_v72_approach_views_dir_p1_checkpoint.json"

sys.path.insert(0, str(ROOT))
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY

# Ranked priority: ask VLM to prefer distinctive objects over generic architecture
VISION_PROMPT_V73 = {
    "left": (
        "A robot navigates indoors and will turn LEFT at a nearby landmark. "
        "This image shows the view JUST BEFORE the left turn. "
        "Identify the best landmark on the LEFT SIDE to mark the turn point.\n\n"
        "Priority order (use the FIRST type you can see clearly):\n"
        "1. ARTWORK/DECOR: painting, poster, mirror, clock, plant, lamp, vase\n"
        "2. DISTINCTIVE FURNITURE: sofa, armchair, bookshelf, dining table, cabinet — include color + material\n"
        "3. ARCHITECTURE: fireplace, staircase, archway, column — only if clearly distinctive\n"
        "4. LAST RESORT: door or wall feature — only if nothing else visible on the left\n\n"
        "Reply with a 2-5 word noun phrase including color/material. Examples:\n"
        "'large framed oil painting', 'red velvet armchair', 'marble fireplace', 'dark wooden staircase'\n"
        "Reply with ONLY the noun phrase. No explanations."
    ),
    "right": (
        "A robot navigates indoors and will turn RIGHT at a nearby landmark. "
        "This image shows the view JUST BEFORE the right turn. "
        "Identify the best landmark on the RIGHT SIDE to mark the turn point.\n\n"
        "Priority order (use the FIRST type you can see clearly):\n"
        "1. ARTWORK/DECOR: painting, poster, mirror, clock, plant, lamp, vase\n"
        "2. DISTINCTIVE FURNITURE: sofa, armchair, bookshelf, dining table, cabinet — include color + material\n"
        "3. ARCHITECTURE: fireplace, staircase, archway, column — only if clearly distinctive\n"
        "4. LAST RESORT: door or wall feature — only if nothing else visible on the right\n\n"
        "Reply with a 2-5 word noun phrase including color/material. Examples:\n"
        "'large framed oil painting', 'red velvet armchair', 'marble fireplace', 'dark wooden staircase'\n"
        "Reply with ONLY the noun phrase. No explanations."
    ),
}

# Generic architectural features to downgrade (accept but flag)
_GENERIC_ARCH_RE = re.compile(
    r'^(door|doorway|doorframe|door frame|wall|window frame|opening|passage|entryway|entrance)\b',
    re.IGNORECASE
)

_REJECT_RE = re.compile(
    r'^(open\s+(?:corridor|space|hallway|area)|plain|featureless|empty|nothing|no\s+furniture|'
    r'no\s+landmark|no\s+(?:visible|clear|distinctive)\s+\w+|'
    r'a\s+room|the\s+room|floor|ceiling|wall(?:s)?|bare\s|neutral|uniform|hallway$|corridor$|'
    r'dark\s+hallway|light\s+hallway|white\s+wall$|beige\s+wall$|grey\s+wall$|'
    r'nothing\s+distinctive)',
    re.IGNORECASE
)

_GENERIC_LANDMARKS = {
    'room', 'area', 'space', 'corner', 'hallway', 'corridor',
    'ceiling', 'floor',
}


def clean_desc(raw: str) -> Optional[str]:
    raw = raw.strip().strip('"\'').strip()
    for prefix in ["The object is", "I see", "I can see", "The most prominent",
                   "Based on", "In this image", "The item is", "There is",
                   "The landmark is", "The turning landmark is", "On the left side",
                   "On the right side", "On the left", "On the right",
                   "Looking at", "The robot should", "Priority 1:", "Priority 2:",
                   "Priority 3:", "Priority 4:", "1.", "2.", "3.", "4."]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].strip(" :-").strip()
    raw = re.sub(r'^(a|an|the)\s+', '', raw, flags=re.IGNORECASE).strip()
    raw = re.split(r'[,;.\n]', raw)[0].strip()
    words = raw.split()
    # Relaxed limit: 2-8 words (was 2-7)
    if len(words) < 2 or len(words) > 8:
        return None
    raw = ' '.join(words[:7])
    if _REJECT_RE.match(raw):
        return None
    last_word = words[-1].lower().rstrip('s')
    if last_word in _GENERIC_LANDMARKS and len(words) <= 2:
        return None
    return raw


def is_generic_arch(desc: str) -> bool:
    return bool(desc and _GENERIC_ARCH_RE.match(desc.strip()))


def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def describe_approach_frame(client, eid: str, label: str, image_path: Path,
                                   turn_dir: str, sem: asyncio.Semaphore,
                                   done_counter: List, total: int, t0: float) -> Dict:
    async with sem:
        try:
            b64 = image_to_base64(image_path)
            prompt = VISION_PROMPT_V73.get(turn_dir, VISION_PROMPT_V73["left"])
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
                max_tokens=40,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            desc = clean_desc(raw)
            result = {
                "eid": eid, "label": label, "desc": desc, "raw": raw,
                "turn_dir": turn_dir, "ok": True,
                "generic_arch": is_generic_arch(desc) if desc else False,
            }
        except Exception as e:
            result = {
                "eid": eid, "label": label, "desc": None, "raw": "",
                "turn_dir": turn_dir, "ok": False, "error": str(e),
                "generic_arch": False,
            }

    done_counter[0] += 1
    if done_counter[0] % 500 == 0 or done_counter[0] == total:
        elapsed = time.time() - t0
        r = done_counter[0] / max(elapsed, 0.001)
        eta = (total - done_counter[0]) / r
        print(f"  [v73 {done_counter[0]}/{total}] {r:.1f}/s  ETA={eta/60:.1f}m", flush=True)
    return result


async def run_phase1_v73(episodes: List[Dict], existing: Dict, v72_ck: Dict,
                          concurrency: int = 16) -> Dict:
    print(f"\n=== Phase 1 v73 (Ranked Priority Prompt) ===")
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
            tasks.append({"eid": eid, "label": frame["label"],
                          "image_path": img_path, "turn_dir": turn_dir})

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
    generic_arch_count = 0
    for result in await asyncio.gather(*coros):
        eid, label = result["eid"], result["label"]
        if eid not in new_results:
            new_results[eid] = {}
        new_results[eid][label] = result["desc"]
        if result.get("generic_arch"):
            generic_arch_count += 1

    elapsed = time.time() - t0
    ok = sum(1 for fd in new_results.values() for d in fd.values() if d)
    fail = sum(1 for fd in new_results.values() for d in fd.values() if not d)
    print(f"\nDone in {elapsed/60:.1f}m: {ok} good ({ok/(ok+fail)*100:.1f}%), {fail} rejected, {generic_arch_count} generic-arch")

    # Merge: v73 wins, fall back to v72 for failures
    merged = dict(existing)
    for eid, ep_results in new_results.items():
        if eid not in merged:
            merged[eid] = {}
        for label, desc in ep_results.items():
            v72_desc = v72_ck.get(eid, {}).get(label)
            if desc:
                merged[eid][label] = desc
            elif v72_desc:
                merged[eid][label] = v72_desc  # fall back to v72
            else:
                merged[eid][label] = None

    P1_V73_CKPT.parent.mkdir(parents=True, exist_ok=True)
    with open(P1_V73_CKPT, "w") as f:
        json.dump(merged, f)
    print(f"Checkpoint: {P1_V73_CKPT}")
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
    if args.resume and P1_V73_CKPT.exists():
        existing = json.load(open(P1_V73_CKPT))
    print(f"Existing checkpoint: {len(existing)} episodes")

    v72_ck = json.load(open(P1_V72_CKPT)) if P1_V72_CKPT.exists() else {}
    print(f"v72 fallback checkpoint: {sum(len(v) for v in v72_ck.values())} frames")

    results = asyncio.run(run_phase1_v73(episodes, existing, v72_ck, args.concurrency))

    total_ep = len(results)
    total_frames = sum(len(v) for v in results.values())
    good = sum(1 for v in results.values() for d in v.values() if d)
    print(f"\nFinal: {total_ep} episodes, {total_frames} frames, {good} useful descriptions")
    print(f"Good rate: {100*good/max(total_frames,1):.1f}%")

    # Compare with v72
    if P1_V72_CKPT.exists():
        old = json.load(open(P1_V72_CKPT))
        improve, regress, same_both, same_none = 0, 0, 0, 0
        for eid in set(results.keys()) & set(old.keys()):
            for label in results[eid]:
                new_d = results[eid].get(label)
                old_d = old[eid].get(label) if eid in old else None
                new_good = bool(new_d and len(new_d.split()) >= 2)
                old_good = bool(old_d and len(old_d.split()) >= 2)
                if new_good and not old_good: improve += 1
                elif old_good and not new_good: regress += 1
                elif new_good and old_good: same_both += 1
                else: same_none += 1
        print(f"\nVs v72: improve={improve}, regress={regress}, both_good={same_both}, both_bad={same_none}")

    # Show sample descriptions
    for eid in list(results.keys())[:3]:
        print(f"\nEP{eid}:")
        for label, desc in results[eid].items():
            v72_d = v72_ck.get(eid, {}).get(label, "N/A")
            print(f"  {label}: v73={desc!r} | v72={v72_d!r}")


if __name__ == "__main__":
    main()
