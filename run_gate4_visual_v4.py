#!/usr/bin/env python3
"""
Gate 4 Visual v4 — Guaranteed GT Distribution via Pre-assigned Starting Verbs

Root problem diagnosed in v3:
  - v3 few-shots had Head(10%) and Leave(10%) — Gemma amplifies rare verbs
  - Result: leave=22% (GT=4%), head=14% (GT=2%), go=2% (GT=19%), turn=6% (GT=16%)
  - Distribution mismatch is NOT fixed by few-shot diversity alone

v4 Fix — Pre-assigned starting verb (guaranteed distribution):
  1. Sample start verb from exact GT distribution for each episode (seeded by ep_id)
  2. Inject verb into prompt: "Start instruction with: [verb]"
  3. Show only 4 examples (one per major verb category: walk/go/turn/exit)
  4. Remove Head and Leave from few-shots entirely
  5. Target 22-28 words (closer to GT's 26.8)

Expected output distribution (matches GT exactly):
  walk(34%), go(19%), turn(16%), exit(11%), leave(4%), come(4%),
  enter(3%), proceed(3%), move(3%), head(2%), face(1%)

Output: outputs/datasets/val_unseen_generated_gemma_visual_v4.json.gz
"""
import asyncio
import gzip
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
LANDMARKS_DIR = ROOT / "outputs" / "gate3_landmarks"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v4.json.gz"
CHECKPOINT_PATH = ROOT / "outputs" / "gate4_visual_v4_checkpoint.json"

from gate2_path.path_analyzer import analyze_path, primitives_to_text
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset

# GT val_unseen starting verb distribution (from R2R annotation analysis)
# Source: counted from 3 × 1839 = 5517 GT instructions
_GT_VERB_DISTRIBUTION = [
    ("Walk",    34),
    ("Go",      19),
    ("Turn",    16),
    ("Exit",    11),
    ("Leave",    4),
    ("Come",     4),
    ("Enter",    3),
    ("Proceed",  3),
    ("Move",     3),
    ("Head",     2),
    ("Face",     1),
]

def _sample_start_verb(episode_id: int) -> str:
    """Deterministically sample start verb from GT distribution using ep_id as seed."""
    rng = random.Random(episode_id * 7919 + 42)  # fixed seed per episode
    verbs = []
    for verb, pct in _GT_VERB_DISTRIBUTION:
        verbs.extend([verb] * pct)
    return rng.choice(verbs)


# Minimal 4-example prompt — no Head or Leave examples
# Focus on Walk/Go/Turn/Exit (the dominant GT patterns)
_PROMPT_TEMPLATE = """You are writing navigation instructions for an indoor robot.

EXAMPLES (these show the correct style and format):
1. "Walk straight down the hallway and turn right into the bedroom. Stop near the window."
2. "Go past the dining table and turn left into the kitchen. Stop near the refrigerator."
3. "Turn left at the top of the stairs and walk into the living room. Stop in front of the couch."
4. "Exit the bedroom and turn left. Walk straight past the gray couch. Stop near the rug."

TASK: Write ONE instruction for:
Starting location: {start_room}
Path: {motion_description}
Stop near: {stop_landmark}

MANDATORY RULES:
- START the instruction with the word: "{start_verb}"
- 20-30 words total
- End with "stop near/at/in front of the [stop landmark]"
- Imperative sentences only
- Write ONLY the instruction text

Instruction: {start_verb}"""


def _primitives_to_motion(primitives: list) -> str:
    parts = []
    for p in primitives:
        if p["type"] == "straight":
            d = p.get("distance_m", 0)
            if d > 4.0:
                parts.append("walk straight")
            elif d > 1.5:
                parts.append("continue forward")
        elif p["type"] == "left_turn":
            parts.append("turn sharp left" if p.get("sharp") else "turn left")
        elif p["type"] == "right_turn":
            parts.append("turn sharp right" if p.get("sharp") else "turn right")
        elif p["type"] == "elevation":
            dir_ = p.get("direction")
            parts.append(f"go {'up' if dir_=='up' else 'down'} the stairs")
    return ", then ".join(parts) if parts else "walk forward"


def build_visual_task_v4(episode: Dict, path_analysis: Dict, landmark_data: Optional[Dict]) -> Dict:
    ep_id = episode["episode_id"]
    start_verb = _sample_start_verb(ep_id)

    start_room = "the room"
    stop_landmark = "the end of the path"

    if landmark_data:
        sc = landmark_data.get("scene_context") or {}
        gl = landmark_data.get("goal_landmark") or {}
        if isinstance(sc, dict) and "error" not in sc:
            start_room = sc.get("room_type", start_room)
        if isinstance(gl, dict) and "error" not in gl:
            stop_landmark = gl.get("stop_landmark", stop_landmark)

    motion_desc = _primitives_to_motion(path_analysis["primitives"])
    elev = (path_analysis["summary"].get("elevation_change_m") or 0)
    if elev > 0.5 and "stairs" not in motion_desc:
        motion_desc = "go up the stairs, then " + motion_desc
    elif elev < -0.5 and "stairs" not in motion_desc:
        motion_desc = "go down the stairs, then " + motion_desc

    prompt = _PROMPT_TEMPLATE.format(
        start_room=start_room,
        motion_description=motion_desc,
        stop_landmark=stop_landmark,
        start_verb=start_verb,
    )
    return {
        "episode_id": ep_id,
        "start_verb": start_verb,
        "motion_sequence": motion_desc,
        "scene_context": start_room,
        "_visual_prompt": prompt,
    }


def quality_check_v4(text: str, start_verb: str) -> tuple:
    words = text.split()
    failures = []
    if len(words) < 10:
        failures.append(f"short({len(words)}w)")
    if len(words) > 60:
        failures.append(f"long({len(words)}w)")
    if not any(w in text.lower() for w in ["stop", "wait", "halt", "stand"]):
        failures.append("no_stop")
    # Verify starts with the assigned verb (case-insensitive)
    if not text.strip().lower().startswith(start_verb.lower()):
        failures.append(f"wrong_start(expected={start_verb})")
    return len(failures) == 0, failures


def clean_output_v4(raw: str, start_verb: str) -> str:
    import re
    # Strip common preambles
    for prefix in ["Instruction:", "Navigation:", "Here is", "Here's", "Answer:", "Response:", start_verb + ":"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].lstrip(":").strip()
    # If missing start_verb, prepend it
    if not raw.lower().startswith(start_verb.lower()):
        raw = start_verb + " " + raw[0].lower() + raw[1:]
    # Take first 3 sentences
    sentences = re.split(r'(?<=[.!?])\s+', raw.strip())
    clean = " ".join(sentences[:3]).strip()
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean


async def generate_v4_instructions(tasks: list, checkpoint: dict, concurrency: int = 12):
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("ERROR: openai not installed.")
        sys.exit(1)

    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    sem = asyncio.Semaphore(concurrency)
    results = {}
    total = len(tasks)
    done = 0
    t0 = time.time()

    async def one(task):
        eid = task["episode_id"]
        start_verb = task["start_verb"]
        if str(eid) in checkpoint:
            return eid, checkpoint[str(eid)], start_verb
        prompt = task.get("_visual_prompt")
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=VLLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=120,
                    temperature=0.55,  # slightly more diversity than v3's 0.5
                )
                raw = resp.choices[0].message.content.strip()
                # Prompt ended with "Instruction: {verb}" — model continues from there.
                # If model output already starts with the verb, just capitalize it.
                # If not, prepend it.
                if raw and raw.lower().startswith(start_verb.lower()):
                    full = start_verb + raw[len(start_verb):]
                else:
                    full = start_verb + " " + raw[0].lower() + raw[1:] if raw else start_verb
                return eid, full, start_verb
            except Exception as e:
                return eid, f"ERROR: {e}", start_verb

    progress = [0]

    async def wrapped(task):
        result = await one(task)
        progress[0] += 1
        d = progress[0]
        if d % 200 == 0 or d == total:
            elapsed = time.time() - t0
            rate = d / elapsed if elapsed > 0 else 0
            eta = (total - d) / rate if rate > 0 else 0
            print(f"  [{d}/{total}] {rate:.1f}/s  ETA={eta/60:.1f}m")
        return result

    coros = [wrapped(t) for t in tasks]
    for coro in asyncio.as_completed(coros):
        eid, text, sv = await coro
        results[eid] = (text, sv)

    return results


async def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=12)
    args = p.parse_args()

    print("=== Gate 4 Visual v4: Guaranteed GT Verb Distribution ===")
    print(f"GT verb targets: walk(34%) go(19%) turn(16%) exit(11%) leave(4%) come(4%) ...")
    print(f"Method: pre-assign start verb per episode (seeded by ep_id)")

    # Load GT dataset
    print("Loading GT dataset...")
    with gzip.open(GT_PATH, "rt") as f:
        gt_data = json.load(f)
    gt_episodes = gt_data["episodes"]
    if args.n_episodes:
        gt_episodes = gt_episodes[:args.n_episodes]
    print(f"Episodes to process: {len(gt_episodes)}")

    # Load landmarks
    print("Loading landmarks...")
    landmark_map = {}
    if LANDMARKS_DIR.exists():
        for fp in LANDMARKS_DIR.glob("*.json"):
            try:
                with open(fp) as f:
                    d = json.load(f)
                ep_id = d.get("episode_id") or int(fp.stem)
                landmark_map[ep_id] = d
            except Exception:
                pass
    print(f"Loaded {len(landmark_map)} landmark files")

    # Load checkpoint
    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH) as f:
                checkpoint = json.load(f)
            print(f"Resuming from checkpoint: {len(checkpoint)} done")
        except Exception:
            pass

    # Build tasks
    tokenizer = VLNTokenizer(GT_PATH)
    tasks = []
    path_analyses = {}
    for ep in gt_episodes:
        try:
            pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
            path_analyses[ep["episode_id"]] = pa
            lm = landmark_map.get(ep["episode_id"])
            task = build_visual_task_v4(ep, pa, lm)
            tasks.append(task)
        except Exception as e:
            print(f"  [ep {ep['episode_id']}] build error: {e}")

    # Verify verb distribution before generation
    from collections import Counter
    assigned_verbs = Counter(t["start_verb"] for t in tasks)
    print(f"\nAssigned verb distribution ({len(tasks)} eps):")
    for verb, cnt in sorted(assigned_verbs.items(), key=lambda x: -x[1]):
        print(f"  {verb:<12} {cnt:4d} ({100*cnt/len(tasks):.1f}%)")

    # Generate
    print(f"\nGenerating with concurrency={args.concurrency}...")
    results = await generate_v4_instructions(tasks, checkpoint, args.concurrency)

    # Assemble episodes
    print("\nAssembling dataset...")
    episodes_out = []
    quality_pass, quality_fail = 0, 0
    fail_reasons = Counter()
    for ep in gt_episodes:
        eid = ep["episode_id"]
        if eid not in results:
            continue
        text_sv = results[eid]
        if isinstance(text_sv, tuple):
            text, start_verb = text_sv
        else:
            text, start_verb = text_sv, "Walk"

        text = clean_output_v4(text, start_verb)
        ok, failures = quality_check_v4(text, start_verb)
        if ok:
            quality_pass += 1
        else:
            quality_fail += 1
            for f in failures:
                fail_reasons[f] += 1
            # Don't discard — just log

        ep_assembled = assemble_episode(ep, text, tokenizer)
        episodes_out.append(ep_assembled)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = {
        "episodes": episodes_out,
        "instruction_vocab": gt_data.get("instruction_vocab", {}),
        "_generation_meta": {
            "mode": "visual_gemma_v4",
            "model": VLLM_MODEL,
            "version": "v4_gt_verb_distribution",
            "n_episodes": len(episodes_out),
            "quality_pass": quality_pass,
        },
    }
    save_dataset(dataset, OUTPUT_PATH)

    # Report
    n = len(episodes_out)
    print(f"\n=== v4 Generation Complete ===")
    print(f"Episodes: {n}/1839  Quality pass: {quality_pass}/{n} ({100*quality_pass/n:.1f}%)")
    if fail_reasons:
        print(f"Failures: {dict(fail_reasons)}")

    # Verify output distribution
    final_verbs = Counter()
    wc = []
    for ep in episodes_out:
        instr = ep['instruction']['instruction_text'] if isinstance(ep.get('instruction'), dict) else ep.get('instruction', '')
        words = instr.strip().split()
        if words:
            final_verbs[words[0].lower().rstrip('.,!')] += 1
            wc.append(len(words))

    print(f"\nFinal verb distribution (target vs actual):")
    print(f"  {'verb':<12} {'target%':>8} {'actual%':>8} {'delta':>7}")
    gt_targets = dict(_GT_VERB_DISTRIBUTION)
    for verb in sorted(final_verbs.keys(), key=lambda v: -final_verbs[v]):
        actual_pct = 100 * final_verbs[verb] / n
        target_pct = gt_targets.get(verb.capitalize(), 0)
        delta = actual_pct - target_pct
        flag = '⚠️' if abs(delta) > 5 else '✅'
        print(f"  {verb:<12} {target_pct:>7.1f}% {actual_pct:>7.1f}%  {delta:>+6.1f}pp {flag}")
    print(f"  Avg words: {sum(wc)/len(wc):.1f} (GT=26.8)")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
