#!/usr/bin/env python3
"""
Gate 4 Visual v5 — Rich Spatial Instruction Generation

Root problem diagnosed in v4 vocabulary analysis:
  - v4 KL(GT→v4)=2.59 vs V1 KL(GT→v1)=1.56 (v1 is 1.7× closer to GT)
  - v4 uses 0% spatial prepositions: "to", "into", "through", "past", "at", "on"
  - GT uses these heavily: "to"=2.59%, "into"=1.49%, "through"=1.44%, "past"=1.00%
  - v4 only uses stop_landmark from landmarks, ignoring rich scene_context and path_landmarks

v5 Fix — Use full landmark data for spatial richness:
  1. Use scene_context.landmarks[0] as intermediate "walk past [X]" anchor
  2. Use goal_landmark.direction_hint for exact stop position ("stop in front of", "stop at")
  3. Inject spatial prepositions explicitly in prompt examples and rules
  4. 4 rich few-shots demonstrating: into/past/through/at prepositions
  5. Same GT verb distribution as v4 (pre-assigned via ep_id seeding)

Expected improvement over v4:
  - KL(GT→v5) ≈ 1.5-1.8 (matching or beating v1)
  - "past" usage: 0% → ~8-12% (GT=1.0%, but richer sentences use more)
  - "into" usage: 0% → ~15-25%
  - "through" usage: 0.55% → ~10-15%
  - Projected SR: ~58-63% (vs v4's ~50-57% est.)

Output: outputs/datasets/val_unseen_generated_gemma_visual_v5.json.gz
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
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v5.json.gz"

from gate2_path.path_analyzer import analyze_path, primitives_to_text
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset

# GT val_unseen starting verb distribution (same as v4)
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


# Rich spatial examples with INTO / PAST / THROUGH / AT / FRONT prepositions
_PROMPT_TEMPLATE = """You are writing navigation instructions for an indoor robot.

EXAMPLES (these show the correct style with spatial prepositions):
1. "Walk through the doorway into the living room and go past the couch. Stop in front of the fireplace."
2. "Go into the kitchen and turn right past the dining table. Stop at the counter near the refrigerator."
3. "Turn left through the hallway into the bedroom. Walk to the window. Stop near the bed."
4. "Exit the bathroom and walk past the wooden door into the bedroom. Stop at the foot of the bed."

TASK: Write ONE navigation instruction for:
Starting: {start_description}
Path: {motion_description}
Destination: {stop_description}

MANDATORY RULES:
- START with the word: "{start_verb}"
- Use spatial prepositions: "into", "through", "past", "at", "in front of", "near"
- 22-32 words total (match GT's 26.8 word average)
- End with a clear stop phrase: "stop near/at/in front of the [landmark]"
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


def build_visual_task_v5(episode: Dict, path_analysis: Dict, landmark_data: Optional[Dict]) -> Dict:
    ep_id = episode["episode_id"]
    start_verb = _sample_start_verb(ep_id)

    # Default values
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
            # Pick a notable intermediate landmark from the start context
            sc_landmarks = sc.get("landmarks") or []
            if sc_landmarks:
                # Pick 2nd landmark (1st often matches start room type)
                start_landmark = sc_landmarks[min(1, len(sc_landmarks)-1)]

        if isinstance(gl, dict) and "error" not in gl:
            stop_landmark = gl.get("stop_landmark", stop_landmark)
            stop_direction = gl.get("direction_hint") or None
            goal_room = gl.get("room_type") or None

    # Build rich start description
    if start_landmark:
        start_description = f"{start_room} (near the {start_landmark})"
    else:
        start_description = f"the {start_room}"

    # Build rich stop description with direction hint
    if stop_direction and len(stop_direction) < 60:
        stop_description = f"{stop_landmark} — {stop_direction}"
    elif goal_room and goal_room != "hallway":
        stop_description = f"{stop_landmark} in the {goal_room}"
    else:
        stop_description = stop_landmark

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
        "motion_sequence": motion_desc,
        "scene_context": start_room,
        "stop_landmark": stop_landmark,
        "_visual_prompt": prompt,
    }


def quality_check_v5(text: str, start_verb: str) -> tuple:
    words = text.split()
    failures = []
    if len(words) < 10:
        failures.append(f"short({len(words)}w)")
    if len(words) > 70:
        failures.append(f"long({len(words)}w)")
    if not any(w in text.lower() for w in ["stop", "wait", "halt", "stand"]):
        failures.append("no_stop")
    if not text.strip().lower().startswith(start_verb.lower()):
        failures.append(f"wrong_start")
    return len(failures) == 0, failures


def clean_output_v5(raw: str, start_verb: str) -> str:
    import re
    for prefix in ["Instruction:", "Navigation:", "Here is", "Here's", "Answer:", "Response:", start_verb + ":"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].lstrip(":").strip()
    # If model output already starts with the verb (prompt ended with "Instruction: {verb}")
    if raw and raw.lower().startswith(start_verb.lower()):
        full = start_verb + raw[len(start_verb):]
    else:
        if not raw.lower().startswith(start_verb.lower()):
            full = start_verb + " " + raw[0].lower() + raw[1:] if raw else start_verb
        else:
            full = raw
    sentences = re.split(r'(?<=[.!?])\s+', full.strip())
    clean = " ".join(sentences[:3]).strip()
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean


async def generate_v5_instructions(tasks: list, concurrency: int = 16):
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
                    max_tokens=130,
                    temperature=0.55,
                )
                raw = resp.choices[0].message.content.strip()
                # Prompt ended with "Instruction: {verb}" — model continues from there
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

    print("=== Gate 4 Visual v5: Rich Spatial Instructions ===")
    print("Fix: use scene_context.landmarks + goal_landmark.direction_hint + spatial prepositions")
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
    for ep in gt_episodes:
        try:
            pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
            lm = landmark_map.get(ep["episode_id"])
            task = build_visual_task_v5(ep, pa, lm)
            tasks.append(task)
        except Exception as e:
            print(f"  [ep {ep['episode_id']}] build error: {e}")

    # Verify verb distribution
    from collections import Counter
    assigned_verbs = Counter(t["start_verb"] for t in tasks)
    print(f"\nAssigned verb distribution ({len(tasks)} eps):")
    for verb, cnt in sorted(assigned_verbs.items(), key=lambda x: -x[1]):
        print(f"  {verb:<12} {cnt:4d} ({100*cnt/len(tasks):.1f}%)")

    # Generate
    print(f"\nGenerating v5 with concurrency={args.concurrency}...")
    results = await generate_v5_instructions(tasks, args.concurrency)

    # Assemble
    print("\nAssembling dataset...")
    episodes_out = []
    quality_pass, quality_fail = 0, 0
    fail_reasons = Counter()
    for ep in gt_episodes:
        eid = ep["episode_id"]
        if eid not in results:
            continue
        text_sv = results[eid]
        text, start_verb = text_sv if isinstance(text_sv, tuple) else (text_sv, "Walk")

        text = clean_output_v5(text, start_verb)
        ok, failures = quality_check_v5(text, start_verb)
        if ok:
            quality_pass += 1
        else:
            quality_fail += 1
            for f in failures:
                fail_reasons[f] += 1

        ep_assembled = assemble_episode(ep, text, tokenizer)
        episodes_out.append(ep_assembled)

    # Post-process no_stop failures using direction_hint
    no_stop_fixed = 0
    for ep_out in episodes_out:
        instr = ep_out['instruction']['instruction_text'] if isinstance(ep_out.get('instruction'), dict) else ep_out.get('instruction', '')
        if not any(w in instr.lower() for w in ['stop', 'wait', 'halt', 'stand']):
            # Look up stop landmark from landmark data
            ep_id_match = ep_out.get('episode_id')
            lm = landmark_map.get(ep_id_match, {})
            gl = lm.get('goal_landmark') or {}
            stop_lm = gl.get('stop_landmark', 'the destination') if isinstance(gl, dict) else 'the destination'
            stop_dir = gl.get('direction_hint', '') if isinstance(gl, dict) else ''
            if stop_dir and len(stop_dir) < 50:
                phrase = f". Stop near the {stop_lm}."
            else:
                phrase = f". Stop near the {stop_lm}."
            instr = instr.rstrip('. ') + phrase
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
            "mode": "visual_gemma_v5",
            "model": VLLM_MODEL,
            "version": "v5_rich_spatial_instructions",
            "n_episodes": len(episodes_out),
            "quality_pass": quality_pass,
            "no_stop_fixed": no_stop_fixed,
        },
    }
    save_dataset(dataset, OUTPUT_PATH)

    n = len(episodes_out)
    print(f"\n=== v5 Generation Complete ===")
    print(f"Episodes: {n}/1839  Quality pass: {quality_pass}/{n} ({100*quality_pass/n:.1f}%)")
    print(f"No-stop fixed by landmark append: {no_stop_fixed}")
    if fail_reasons:
        print(f"Failures: {dict(fail_reasons)}")

    # Verify distribution and vocabulary
    final_verbs = Counter()
    wc, spatial_count = [], 0
    spatial_words = ['into', 'through', 'past', 'in front', 'at the', 'onto']
    for ep in episodes_out:
        instr = ep['instruction']['instruction_text'] if isinstance(ep.get('instruction'), dict) else ep.get('instruction', '')
        words = instr.strip().split()
        if words:
            final_verbs[words[0].lower().rstrip('.,!')] += 1
            wc.append(len(words))
        if any(sw in instr.lower() for sw in spatial_words):
            spatial_count += 1

    print(f"\nFinal verb distribution (target vs actual):")
    gt_targets = dict(_GT_VERB_DISTRIBUTION)
    for verb in sorted(final_verbs.keys(), key=lambda v: -final_verbs[v])[:8]:
        actual_pct = 100 * final_verbs[verb] / n
        target_pct = gt_targets.get(verb.capitalize(), 0)
        delta = actual_pct - target_pct
        flag = '⚠️' if abs(delta) > 5 else '✅'
        print(f"  {verb:<12} target={target_pct:.0f}%  actual={actual_pct:.1f}%  {delta:+.1f}pp {flag}")
    print(f"  Avg words: {sum(wc)/len(wc):.1f} (GT=26.8)")
    print(f"  Spatial preposition coverage: {100*spatial_count/n:.1f}% (GT uses into/through/past/at heavily)")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
