#!/usr/bin/env python3
"""
Gate 4 Visual v2 — GT-style instruction generation

Key improvements over v1:
  1. Shorter instructions (20-30 words): GT averages 26.8 words, v1 averages 31.6
  2. GT-matched style: "Exit [room], turn [dir]. Walk past [landmark]. Stop near [stop]."
  3. Start room explicitly named: "Leave the bedroom" vs generic path start
  4. Few-shot GT examples in prompt: teaches Gemma the exact style to use
  5. Stop condition made MORE prominent: mentioned first in requirements

The -0.71pp gap (visual 63.18% vs GT 63.89%) is partly due to instruction length
and style mismatch. v2 targets the GT style more precisely.

Output: outputs/datasets/val_unseen_generated_gemma_visual_v2.json.gz
"""
import asyncio
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
LANDMARKS_DIR = ROOT / "outputs" / "gate3_landmarks"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v2.json.gz"
CHECKPOINT_PATH = ROOT / "outputs" / "gate4_visual_v2_checkpoint.json"

from gate2_path.path_analyzer import analyze_path, primitives_to_text
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset


VISUAL_INSTRUCTION_PROMPT_V2 = """You are writing navigation instructions for a robot that navigates indoors.
Your goal: write a SHORT, NATURAL instruction that closely matches these human-written examples.

HUMAN EXAMPLE INSTRUCTIONS (match this style exactly):
- "Exit the bedroom and turn left. Walk straight passing the gray couch and stop near the rug."
- "Go up the stairs to the top. Turn right and walk into the bedroom. Stop at the bed."
- "Walk out of the kitchen and turn left. Walk down the hallway past the door and stop near the bathroom entrance."
- "Walk straight past the dining table and turn right. Walk into the living room and stop in front of the couch."
- "Leave the bedroom and walk straight into the hallway. Turn left and stop in front of the window."

NOW WRITE A NEW INSTRUCTION FOR THIS SPECIFIC NAVIGATION:
Starting room: {start_room}
Navigation: {motion_description}
Stop at: {stop_landmark} ({stop_hint})

REQUIREMENTS (critical):
- 15 to 30 words total (match example length above)
- Start with: "Exit/Leave the {start_room}" or "Walk out of the {start_room}"
- End with: "stop near/at/in front of the {stop_landmark}"
- Natural English, second person, imperative verbs
- NO preamble, NO explanation — write ONLY the instruction

Instruction:"""


def primitives_to_motion_description(primitives: list, start_room: str) -> str:
    """Convert path primitives to a human-friendly motion description."""
    parts = []
    for p in primitives:
        if p["type"] == "straight":
            d = p.get("distance_m", 0)
            if d > 3.0:
                parts.append("walk straight")
            elif d > 1.0:
                parts.append("continue forward")
        elif p["type"] == "left_turn":
            if p.get("sharp"):
                parts.append("turn sharp left")
            else:
                parts.append("turn left")
        elif p["type"] == "right_turn":
            if p.get("sharp"):
                parts.append("turn sharp right")
            else:
                parts.append("turn right")
        elif p["type"] == "elevation":
            if p.get("direction") == "up":
                parts.append("go up the stairs")
            elif p.get("direction") == "down":
                parts.append("go down the stairs")

    if not parts:
        return "walk forward"
    return ", then ".join(parts)


def build_visual_task_v2(episode: Dict, path_analysis: Dict, landmark_data: Optional[Dict]) -> Dict:
    """Build v2 task with GT-matching prompt style."""
    summary = path_analysis["summary"]

    start_room = "indoor space"
    stop_landmark = "the end of the path"
    stop_hint = ""

    if landmark_data:
        sc = landmark_data.get("scene_context") or {}
        gl = landmark_data.get("goal_landmark") or {}

        if isinstance(sc, dict) and "error" not in sc:
            start_room = sc.get("room_type", start_room)

        if isinstance(gl, dict) and "error" not in gl:
            stop_landmark = gl.get("stop_landmark", stop_landmark)
            stop_hint = gl.get("direction_hint", "")

    motion_desc = primitives_to_motion_description(path_analysis["primitives"], start_room)

    # Elevation from summary
    elev = summary.get("elevation_change_m", 0) or 0
    if elev > 0.5 and "stairs" not in motion_desc:
        motion_desc = "go up the stairs, " + motion_desc
    elif elev < -0.5 and "stairs" not in motion_desc:
        motion_desc = "go down the stairs, " + motion_desc

    prompt = VISUAL_INSTRUCTION_PROMPT_V2.format(
        start_room=start_room,
        motion_description=motion_desc,
        stop_landmark=stop_landmark,
        stop_hint=stop_hint or f"near the {stop_landmark}",
    )

    return {
        "episode_id": episode["episode_id"],
        "motion_sequence": motion_desc,
        "scene_context": "",
        "_visual_prompt": prompt,
    }


def quality_check_v2(text: str) -> tuple:
    words = text.split()
    failures = []
    if len(words) < 10:
        failures.append(f"short({len(words)}w)")
    if len(words) > 45:
        failures.append(f"long({len(words)}w)")
    stop_words = ["stop", "wait", "halt", "stand", "pause"]
    if not any(w in text.lower() for w in stop_words):
        failures.append("no_stop")
    return len(failures) == 0, failures


def clean_output(raw: str) -> str:
    import re
    for prefix in ["Instruction:", "Navigation:", "Here is", "Here's", "Answer:", "Sure", "Result:"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].lstrip(":").strip()
    sentences = re.split(r'(?<=[.!?])\s+', raw.strip())
    clean = " ".join(sentences[:3]).strip()
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean


async def generate_v2_instructions(tasks: list, checkpoint: dict, concurrency: int = 12):
    """Generate GT-style instructions via Gemma."""
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
        if str(eid) in checkpoint:
            return eid, checkpoint[str(eid)]
        prompt = task.get("_visual_prompt")
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=VLLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,     # Keep short for GT-style
                    temperature=0.4,    # Slightly higher creativity
                )
                return eid, resp.choices[0].message.content.strip()
            except Exception as e:
                return eid, f"ERROR: {e}"

    coros = [one(t) for t in tasks]
    for coro in asyncio.as_completed(coros):
        eid, text = await coro
        results[eid] = text
        done += 1
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        if done % 200 == 0 or done == total:
            print(f"  [{done}/{total}] {rate:.1f}/s  ETA={eta/60:.1f}m")

    return results


async def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--output", default=str(OUTPUT_PATH))
    p.add_argument("--compare", action="store_true", help="Compare v1 vs v2 side-by-side")
    args = p.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Gate 4 Visual v2 — GT-Style Instructions ===")
    print(f"  Model:      Gemma 4 31B AWQ @ {VLLM_BASE_URL}")
    print(f"  Target:     20-30 words, GT-matching style")
    print(f"  Output:     {output_path}")
    print()

    # Load GT episodes
    with gzip.open(GT_PATH, "rt") as f:
        gt_data = json.load(f)
    gt_eps = gt_data["episodes"]
    if args.n_episodes:
        gt_eps = gt_eps[:args.n_episodes]

    gt_ep_dict = {ep["episode_id"]: ep for ep in gt_eps}
    print(f"  Episodes:   {len(gt_eps)}")

    # Load v1 for comparison if requested
    v1_path = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual.json.gz"
    v1_dict = {}
    if args.compare and v1_path.exists():
        with gzip.open(v1_path, "rt") as f:
            v1_dict = {ep["episode_id"]: ep for ep in json.load(f)["episodes"]}

    # Load landmarks
    lm_map = {}
    for lf in LANDMARKS_DIR.glob("episode_*.json"):
        with open(lf) as f:
            ld = json.load(f)
        lm_map[ld["episode_id"]] = ld
    print(f"  Landmarks:  {len(lm_map)}")

    # Load checkpoint
    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        print(f"  Checkpoint: {len(checkpoint)} cached")

    # Build tasks
    print("\n[Gate 2] Analyzing paths...")
    tasks = []
    for ep in gt_eps:
        eid = ep["episode_id"]
        if str(eid) in checkpoint:
            continue
        analysis = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        task = build_visual_task_v2(ep, analysis, lm_map.get(eid))
        tasks.append(task)

    print(f"  {len(tasks)} to generate ({len(checkpoint)} cached)")

    # Generate
    if tasks:
        print(f"\n[Gate 4 v2] Generating (concurrency={args.concurrency})...")
        t0 = time.time()
        new_results = await generate_v2_instructions(tasks, checkpoint, args.concurrency)
        checkpoint.update({str(k): v for k, v in new_results.items()})
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f)
        elapsed = time.time() - t0
        print(f"  Done: {len(new_results)} in {elapsed:.1f}s ({len(new_results)/elapsed:.1f}/s)")

    # Quality check
    all_gen = {int(k): v for k, v in checkpoint.items()}
    ok = sum(1 for v in all_gen.values() if quality_check_v2(clean_output(v))[0])
    words = [len(clean_output(v).split()) for v in all_gen.values()]
    import statistics
    avg_words = statistics.mean(words) if words else 0
    print(f"\n[Quality] {ok}/{len(all_gen)} pass ({100*ok/max(1,len(all_gen)):.1f}%)")
    print(f"  Avg words: {avg_words:.1f} (GT=26.8, v1=31.6, target=20-30)")

    # Show comparisons
    print("\n[Samples] v1 vs v2 vs GT (first 5 episodes):")
    for ep in gt_eps[:5]:
        eid = ep["episode_id"]
        raw = all_gen.get(eid, "MISSING")
        v2_text = clean_output(raw) if raw != "MISSING" else "MISSING"
        v1_text = v1_dict.get(eid, {}).get("instruction", {}).get("instruction_text", "N/A") if v1_dict else "N/A"
        gt_text = ep["instruction"]["instruction_text"].strip()
        analysis = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        motion = primitives_to_text(analysis["primitives"])
        print(f"\n  Ep {eid}:")
        print(f"    Path:  {motion}")
        print(f"    GT:    {gt_text[:85]}")
        print(f"    v1:    {v1_text[:85]}")
        print(f"    v2:    {v2_text[:85]}")

    # Assemble
    print("\n[Gate 5/6] Assembling dataset...")
    tok = VLNTokenizer(GT_PATH)
    assembled = []
    for ep in gt_eps:
        eid = ep["episode_id"]
        raw = all_gen.get(eid, "")
        text = clean_output(raw) if raw else ""
        if not text or text.startswith("ERROR"):
            continue
        assembled.append(assemble_episode(ep, text, tok))

    dataset = {
        "episodes": assembled,
        "instruction_vocab": gt_data.get("instruction_vocab", {}),
        "_generation_meta": {
            "mode": "visual_gemma_v2",
            "model": VLLM_MODEL,
            "version": "v2_gt_style",
            "n_episodes": len(assembled),
            "avg_words": avg_words,
            "quality_ok": ok,
            "target_style": "GT-matched: 20-30 words, exit/stop pattern",
        },
    }
    save_dataset(dataset, output_path)

    print(f"\n=== Done ===")
    print(f"  v2 dataset: {output_path}")
    print(f"  Episodes:   {len(assembled)}/{len(gt_eps)}")
    print(f"  Avg words:  {avg_words:.1f} (target: 20-30)")
    print(f"\nTo evaluate v2 instructions:")
    print(f"  bash habitat_eval/scripts/run_eval_gate7_visual_v2.sh")


if __name__ == "__main__":
    asyncio.run(main())
