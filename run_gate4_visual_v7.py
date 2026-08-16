#!/usr/bin/env python3
"""
Gate 4 Visual v7 — Calibrated Natural Instructions

Root problem with v6:
  - `towards` massively overshoots: 10.11% vs GT's 1.02%
    (example 2 "Walk towards the counter" was too influential)
  - `wait` still 0%: model ignores wait even in examples
  - avg_words 29.2 > GT's 26.8 (too long)
  - spatial still 91.5% (GT=74.6%)

v7 Fix:
  1. Replace "towards" example with a simple "to" directional
  2. Post-process inject "wait" in ~1.5% of episodes (seeded, so reproducible)
  3. Reduce max_tokens: 130 → 115 (fewer words per instruction)
  4. Tighter word count rule: "20-28 words" instead of "22-32"
  5. Remove "towards" from examples entirely (let model use it naturally at GT rate)

Expected v7 vocabulary:
  - spatial: ~80-85% (approaching GT's 74.6%, down from v6's 91.5%)
  - towards: ~0-2% (down from v6's 10.11%, GT=1.02%)
  - to:      ~1-2% (up from v6's 1.20%, GT=2.59%)
  - wait:    ~1-2% (injected post-hoc, GT=1.14%)
  - avg words: ~26-27 (down from v6's 29.2, GT=26.8)
  Expected KL(GT→v7) ≈ 1.7-1.9 (closer to v1's 1.56)

Output: outputs/datasets/val_unseen_generated_gemma_visual_v7.json.gz
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
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v7.json.gz"

from gate2_path.path_analyzer import analyze_path, primitives_to_text
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset

# GT val_unseen starting verb distribution (same as v4/v5/v6)
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
    rng = random.Random(episode_id * 7919 + 42)
    verbs = [v for v, pct in _GT_VERB_DISTRIBUTION for _ in range(pct)]
    return rng.choice(verbs)


def _use_wait_stop(episode_id: int) -> bool:
    """Seed-deterministic: ~1.5% of episodes use 'wait' as stop word (GT=1.14%)."""
    rng = random.Random(episode_id * 3137 + 17)
    return rng.random() < 0.015


# v7 examples: NO "towards" example. Use "to" directional instead.
# 4 examples showing variety: spatial / simple directional / wait / mixed
_PROMPT_TEMPLATE = """You are writing navigation instructions for an indoor robot.

EXAMPLES (show natural variety — spatial prepositions and simple directional phrases):
1. "Walk through the doorway into the living room and go past the couch. Stop in front of the fireplace."
2. "Go to the kitchen and turn right. Walk to the counter near the refrigerator and stop there."
3. "Turn left through the hallway. Walk into the bedroom and wait near the window by the bed."
4. "Exit the bathroom and walk past the wooden door on the right. Turn into the bedroom. Stop at the foot of the bed."

TASK: Write ONE navigation instruction for:
Starting: {start_description}
Path: {motion_description}
Destination: {stop_description}

RULES:
- START with the word: "{start_verb}"
- Use natural language: spatial words (into/through/past/at/in front of) OR "to/at" directional
- 20-28 words total (keep concise — GT average is 26.8 words)
- End with: "stop at/near/in front of the [landmark]" or "wait at/near the [landmark]"
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


def build_visual_task_v7(episode: Dict, path_analysis: Dict, landmark_data: Optional[Dict]) -> Dict:
    ep_id = episode["episode_id"]
    start_verb = _sample_start_verb(ep_id)
    use_wait = _use_wait_stop(ep_id)

    start_room = "the room"
    start_landmark = None
    stop_landmark = "the destination"
    stop_direction = None
    goal_room = None

    if landmark_data:
        sc = landmark_data.get("scene_context") or {}
        gl = landmark_data.get("goal_landmark") or {}

        if isinstance(sc, dict) and "error" not in sc:
            start_room = sc.get("room_type", start_room)
            sc_landmarks = sc.get("landmarks") or []
            if sc_landmarks:
                start_landmark = sc_landmarks[min(1, len(sc_landmarks)-1)]

        if isinstance(gl, dict) and "error" not in gl:
            stop_landmark = gl.get("stop_landmark", stop_landmark)
            stop_direction = gl.get("direction_hint") or None
            goal_room = gl.get("room_type") or None

    if start_landmark:
        start_description = f"{start_room} (near the {start_landmark})"
    else:
        start_description = f"the {start_room}"

    if stop_direction and len(stop_direction) < 60:
        stop_description = f"{stop_landmark} — {stop_direction}"
    elif goal_room and goal_room != "hallway":
        stop_description = f"{stop_landmark} in the {goal_room}"
    else:
        stop_description = stop_landmark

    # For wait episodes: append the stop word hint to the destination description
    if use_wait:
        stop_description = f"{stop_description} [use 'wait' to stop]"

    motion_desc = _primitives_to_motion(path_analysis["primitives"])
    elev = (path_analysis["summary"].get("elevation_change_m") or 0)
    if elev > 0.5 and "stairs" not in motion_desc:
        motion_desc = "go up the stairs, then " + motion_desc
    elif elev < -0.5 and "stairs" not in motion_desc:
        motion_desc = "go down the stairs, then " + motion_desc

    prompt = _PROMPT_TEMPLATE.format(
        start_description=start_description,
        motion_description=motion_desc,
        stop_description=stop_description,
        start_verb=start_verb,
    )
    return {
        "episode_id": ep_id,
        "start_verb": start_verb,
        "use_wait": use_wait,
        "motion_sequence": motion_desc,
        "scene_context": start_room,
        "stop_landmark": stop_landmark,
        "_visual_prompt": prompt,
    }


def _inject_wait(text: str, stop_landmark: str) -> str:
    """Replace stop/halt/stand with wait in final clause."""
    import re
    # Try replacing "stop at/near/in front of" with "wait at/near"
    text_lower = text.lower()
    for phrase in ["stop at", "stop near", "stop in front of", "stop by"]:
        if phrase in text_lower:
            idx = text_lower.rfind(phrase)  # last occurrence
            replacement = phrase.replace("stop", "wait")
            return text[:idx] + replacement + text[idx + len(phrase):]
    # Fallback: append wait phrase
    text = text.rstrip('. ') + f". Wait near the {stop_landmark}."
    return text


def quality_check_v7(text: str, start_verb: str) -> tuple:
    words = text.split()
    failures = []
    if len(words) < 10:
        failures.append(f"short({len(words)}w)")
    if len(words) > 70:
        failures.append(f"long({len(words)}w)")
    if not any(w in text.lower() for w in ["stop", "wait", "halt", "stand"]):
        failures.append("no_stop")
    if not text.strip().lower().startswith(start_verb.lower()):
        failures.append("wrong_start")
    return len(failures) == 0, failures


def clean_output_v7(raw: str, start_verb: str) -> str:
    import re
    for prefix in ["Instruction:", "Navigation:", "Here is", "Here's", "Answer:", "Response:", start_verb + ":"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].lstrip(":").strip()
    if raw and raw.lower().startswith(start_verb.lower()):
        full = start_verb + raw[len(start_verb):]
    else:
        full = start_verb + " " + raw[0].lower() + raw[1:] if raw else start_verb
    sentences = re.split(r'(?<=[.!?])\s+', full.strip())
    clean = " ".join(sentences[:3]).strip()
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean


async def generate_v7_instructions(tasks: list, concurrency: int = 16):
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("ERROR: openai not installed.")
        sys.exit(1)

    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    sem = asyncio.Semaphore(concurrency)
    total = len(tasks)
    t0 = time.time()
    results = {}
    progress = [0]

    async def one(task):
        eid = task["episode_id"]
        start_verb = task["start_verb"]
        prompt = task.get("_visual_prompt")
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=VLLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=115,  # shorter than v6 (130) → fewer words
                    temperature=0.55,  # same as v5 (v6 was 0.60)
                )
                raw = resp.choices[0].message.content.strip()
                if raw and raw.lower().startswith(start_verb.lower()):
                    full = start_verb + raw[len(start_verb):]
                else:
                    full = start_verb + " " + raw[0].lower() + raw[1:] if raw else start_verb
                return eid, full, start_verb
            except Exception as e:
                return eid, f"ERROR: {e}", start_verb

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

    for coro in asyncio.as_completed([wrapped(t) for t in tasks]):
        eid, text, sv = await coro
        results[eid] = (text, sv)
    return results


async def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=16)
    args = p.parse_args()

    print("=== Gate 4 Visual v7: Calibrated Natural Instructions ===")
    print("Fixes: remove 'towards' example, inject wait ~1.5%, shorter tokens")
    print("GT verb targets: walk(34%) go(19%) turn(16%) exit(11%) ...")

    with gzip.open(GT_PATH, "rt") as f:
        gt_data = json.load(f)
    gt_episodes = gt_data["episodes"][:args.n_episodes] if args.n_episodes else gt_data["episodes"]
    print(f"Episodes: {len(gt_episodes)}")

    # Load landmarks
    landmark_map = {}
    if LANDMARKS_DIR.exists():
        for fp in LANDMARKS_DIR.glob("*.json"):
            try:
                with open(fp) as f:
                    d = json.load(f)
                ep_id = d.get("episode_id") or int(fp.stem.replace("episode_", ""))
                landmark_map[ep_id] = d
            except Exception:
                pass
    print(f"Loaded {len(landmark_map)} landmark files")

    # Build tasks
    tokenizer = VLNTokenizer(GT_PATH)
    tasks = []
    wait_count_planned = 0
    for ep in gt_episodes:
        try:
            pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
            lm = landmark_map.get(ep["episode_id"])
            task = build_visual_task_v7(ep, pa, lm)
            tasks.append(task)
            if task.get("use_wait"):
                wait_count_planned += 1
        except Exception as e:
            print(f"  [ep {ep['episode_id']}] build error: {e}")

    from collections import Counter
    assigned_verbs = Counter(t["start_verb"] for t in tasks)
    print(f"\nAssigned verb distribution ({len(tasks)} eps):")
    for verb, cnt in sorted(assigned_verbs.items(), key=lambda x: -x[1]):
        print(f"  {verb:<12} {cnt:4d} ({100*cnt/len(tasks):.1f}%)")
    print(f"Wait injection planned: {wait_count_planned} eps ({100*wait_count_planned/len(tasks):.1f}%)")

    print(f"\nGenerating v7 with concurrency={args.concurrency}...")
    results = await generate_v7_instructions(tasks, args.concurrency)

    # Assemble + post-process wait injection
    print("\nAssembling dataset...")
    episodes_out = []
    quality_pass, quality_fail = 0, 0
    fail_reasons = Counter()
    wait_injected = 0

    for ep in gt_episodes:
        eid = ep["episode_id"]
        if eid not in results:
            continue
        text_sv = results[eid]
        text, start_verb = text_sv if isinstance(text_sv, tuple) else (text_sv, "Walk")

        text = clean_output_v7(text, start_verb)

        # Post-process: inject "wait" for designated episodes
        task_meta = next((t for t in tasks if t["episode_id"] == eid), None)
        if task_meta and task_meta.get("use_wait"):
            lm = landmark_map.get(eid, {})
            gl = lm.get("goal_landmark") or {}
            stop_lm = gl.get("stop_landmark", "the destination") if isinstance(gl, dict) else "the destination"
            text = _inject_wait(text, stop_lm)
            wait_injected += 1

        ok, failures = quality_check_v7(text, start_verb)
        if ok:
            quality_pass += 1
        else:
            quality_fail += 1
            for f in failures:
                fail_reasons[f] += 1

        ep_assembled = assemble_episode(ep, text, tokenizer)
        episodes_out.append(ep_assembled)

    # Post-process no_stop failures
    no_stop_fixed = 0
    for ep_out in episodes_out:
        instr = ep_out['instruction']['instruction_text'] if isinstance(ep_out.get('instruction'), dict) else ep_out.get('instruction', '')
        if not any(w in instr.lower() for w in ['stop', 'wait', 'halt', 'stand']):
            ep_id_match = ep_out.get('episode_id')
            lm = landmark_map.get(ep_id_match, {})
            gl = lm.get('goal_landmark') or {}
            stop_lm = gl.get('stop_landmark', 'the destination') if isinstance(gl, dict) else 'the destination'
            instr = instr.rstrip('. ') + f". Stop near the {stop_lm}."
            if isinstance(ep_out.get('instruction'), dict):
                ep_out['instruction']['instruction_text'] = instr
            else:
                ep_out['instruction'] = instr
            no_stop_fixed += 1

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = {
        "episodes": episodes_out,
        "instruction_vocab": gt_data.get("instruction_vocab", {}),
        "_generation_meta": {
            "mode": "visual_gemma_v7",
            "model": VLLM_MODEL,
            "version": "v7_calibrated_natural",
            "n_episodes": len(episodes_out),
            "quality_pass": quality_pass,
            "no_stop_fixed": no_stop_fixed,
            "wait_injected": wait_injected,
        },
    }
    save_dataset(dataset, OUTPUT_PATH)

    n = len(episodes_out)
    print(f"\n=== v7 Generation Complete ===")
    print(f"Episodes: {n}/1839  Quality pass: {quality_pass}/{n} ({100*quality_pass/n:.1f}%)")
    print(f"Wait injected: {wait_injected} eps ({100*wait_injected/n:.2f}%)")
    print(f"No-stop fixed: {no_stop_fixed}")
    if fail_reasons:
        print(f"Failures: {dict(fail_reasons)}")

    # Vocabulary analysis
    final_verbs = Counter()
    wc = []
    spatial_count, wait_count, to_count, towards_count = 0, 0, 0, 0
    spatial_words = ['into', 'through', 'past', 'in front', 'at the', 'onto']
    for ep in episodes_out:
        instr = ep['instruction']['instruction_text'] if isinstance(ep.get('instruction'), dict) else ep.get('instruction', '')
        words = instr.strip().split()
        if words:
            final_verbs[words[0].lower().rstrip('.,!')] += 1
            wc.append(len(words))
        instr_l = instr.lower()
        if any(sw in instr_l for sw in spatial_words):
            spatial_count += 1
        if 'wait' in instr_l:
            wait_count += 1
        tokens = instr_l.split()
        for i, tok in enumerate(tokens):
            if tok in ('to', 'to.', 'to,') and i + 1 < len(tokens) and tokens[i+1] not in ('stop', 'the'):
                to_count += 1
                break
        if 'towards' in instr_l or 'toward' in instr_l:
            towards_count += 1

    print(f"\nVocabulary vs GT targets:")
    print(f"  Spatial prepositions: {100*spatial_count/n:.1f}%  (GT=74.6%  v6=91.5%  v5=99.5%)")
    print(f"  wait:    {100*wait_count/n:.2f}%  (GT=1.14%  v6=0.00%  injected={100*wait_injected/n:.2f}%)")
    print(f"  to:      {100*to_count/n:.2f}%   (GT=2.59%  v6=1.20%)")
    print(f"  towards: {100*towards_count/n:.2f}%  (GT=1.02%  v6=10.11%)")
    print(f"  Avg words: {sum(wc)/len(wc):.1f}  (GT=26.8  v6=29.2)")

    print(f"\nVerb distribution (top 8):")
    gt_targets = dict(_GT_VERB_DISTRIBUTION)
    for verb in sorted(final_verbs.keys(), key=lambda v: -final_verbs[v])[:8]:
        actual_pct = 100 * final_verbs[verb] / n
        target_pct = gt_targets.get(verb.capitalize(), 0)
        delta = actual_pct - target_pct
        flag = '⚠️' if abs(delta) > 5 else '✅'
        print(f"  {verb:<12} target={target_pct:.0f}%  actual={actual_pct:.1f}%  {delta:+.1f}pp {flag}")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
