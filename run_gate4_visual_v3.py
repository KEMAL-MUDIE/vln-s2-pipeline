#!/usr/bin/env python3
"""
Gate 4 Visual v3 — Diverse GT-style instruction generation

Key improvements over v2:
  1. 10 diverse few-shot GT examples (vs v2's 5) covering all starting patterns
  2. Starting verb distribution matches GT: Walk/Go (55%), Turn (16%), Exit (11%)
  3. Temperature 0.5 (vs v2's 0.4) for more natural language diversity
  4. No forced "Exit [room]" pattern — lets Gemma choose naturally from the path

v2 diagnosed issue:
  - 98.7% "Exit/Leave" start (GT is only 28.1%)
  - Over-fitting to few-shot prompt pattern

v3 fix:
  - 10 diverse starting patterns in few-shot: Walk, Go, Turn, Head, Exit, Leave
  - Expected: start verb distribution matching GT Walk(34%) + Go(19%) + Turn(16%) + Exit(11%)

Output: outputs/datasets/val_unseen_generated_gemma_visual_v3.json.gz
"""
import asyncio
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
LANDMARKS_DIR = ROOT / "outputs" / "gate3_landmarks"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v3.json.gz"
CHECKPOINT_PATH = ROOT / "outputs" / "gate4_visual_v3_checkpoint.json"

from gate2_path.path_analyzer import analyze_path, primitives_to_text
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset

# 10 diverse few-shot examples matching GT distribution
# GT start verb distribution: Walk(34%), Go(19%), Turn(16%), Exit(11%), Leave(4%), Head(2%)
VISUAL_INSTRUCTION_PROMPT_V3 = """You are writing navigation instructions for a robot navigating indoors.
Match the style of these human-written examples EXACTLY.

EXAMPLES (use varied starting words like the examples below):
1. "Walk straight down the hallway and turn right into the bedroom. Stop near the window."
2. "Go past the dining table and turn left into the kitchen. Wait near the refrigerator."
3. "Turn left at the top of the stairs and walk into the living room. Stop in front of the couch."
4. "Exit the bedroom and turn left. Walk straight past the gray couch and stop near the rug."
5. "Head down the hallway to the right. Walk through the door and stop near the bathroom sink."
6. "Walk up the stairs to the landing. Turn left and stop near the railing."
7. "Go into the living room and walk past the chairs. Stop near the television."
8. "Turn right and walk down the long hallway. Enter the bedroom on the left and stop near the bed."
9. "Leave the kitchen and walk into the dining area. Stop near the table."
10. "Walk straight through the door and into the study. Stop near the desk by the window."

NOW WRITE A SINGLE INSTRUCTION FOR:
Starting room: {start_room}
Navigation: {motion_description}
Stop near: {stop_landmark}

RULES:
- 15-30 words total (match example lengths)
- Use VARIED starting words: Walk/Go/Turn/Head/Exit/Leave (do NOT always start with "Exit")
- End with "stop near/at/in front of the [landmark]"
- Simple, direct imperative sentences
- Write ONLY the instruction, no explanation

Instruction:"""


def primitives_to_motion_description(primitives: list) -> str:
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
            if p.get("direction") == "up":
                parts.append("go up the stairs")
            elif p.get("direction") == "down":
                parts.append("go down the stairs")
    return ", then ".join(parts) if parts else "walk forward"


def build_visual_task_v3(episode: Dict, path_analysis: Dict, landmark_data: Optional[Dict]) -> Dict:
    start_room = "the room"
    stop_landmark = "the end of the path"

    if landmark_data:
        sc = landmark_data.get("scene_context") or {}
        gl = landmark_data.get("goal_landmark") or {}
        if isinstance(sc, dict) and "error" not in sc:
            start_room = sc.get("room_type", start_room)
        if isinstance(gl, dict) and "error" not in gl:
            stop_landmark = gl.get("stop_landmark", stop_landmark)

    motion_desc = primitives_to_motion_description(path_analysis["primitives"])
    elev = (path_analysis["summary"].get("elevation_change_m") or 0)
    if elev > 0.5 and "stairs" not in motion_desc:
        motion_desc = "go up the stairs, then " + motion_desc
    elif elev < -0.5 and "stairs" not in motion_desc:
        motion_desc = "go down the stairs, then " + motion_desc

    prompt = VISUAL_INSTRUCTION_PROMPT_V3.format(
        start_room=start_room,
        motion_description=motion_desc,
        stop_landmark=stop_landmark,
    )
    return {
        "episode_id": episode["episode_id"],
        "motion_sequence": motion_desc,
        "scene_context": start_room,
        "_visual_prompt": prompt,
    }


def quality_check_v3(text: str) -> tuple:
    words = text.split()
    failures = []
    if len(words) < 10:
        failures.append(f"short({len(words)}w)")
    if len(words) > 50:
        failures.append(f"long({len(words)}w)")
    if not any(w in text.lower() for w in ["stop", "wait", "halt", "stand", "pause"]):
        failures.append("no_stop")
    return len(failures) == 0, failures


def clean_output(raw: str) -> str:
    import re
    for prefix in ["Instruction:", "Navigation:", "Here is", "Here's", "Answer:", "Sure", "Result:", "Response:"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].lstrip(":").strip()
    sentences = re.split(r'(?<=[.!?])\s+', raw.strip())
    clean = " ".join(sentences[:3]).strip()
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean


async def generate_v3_instructions(tasks: list, checkpoint: dict, concurrency: int = 12):
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
                    max_tokens=150,
                    temperature=0.5,
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
    p.add_argument("--analyze", action="store_true", help="Print starting verb distribution")
    args = p.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Gate 4 Visual v3 — Diverse GT-Style Instructions ===")
    print(f"  Model:  Gemma 4 31B AWQ @ {VLLM_BASE_URL}")
    print(f"  Fix:    v2 over-fit to 'Exit' (98.7% vs GT 28.1%) — 10 diverse examples")
    print(f"  Target: Walk(34%) + Go(19%) + Turn(16%) + Exit(11%) + others")
    print(f"  Output: {output_path}")
    print()

    with gzip.open(GT_PATH, "rt") as f:
        gt_data = json.load(f)
    gt_eps = gt_data["episodes"]
    if args.n_episodes:
        gt_eps = gt_eps[:args.n_episodes]
    print(f"  Episodes: {len(gt_eps)}")

    lm_map = {}
    for lf in LANDMARKS_DIR.glob("episode_*.json"):
        with open(lf) as f:
            ld = json.load(f)
        lm_map[ld["episode_id"]] = ld
    print(f"  Landmarks: {len(lm_map)}")

    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        print(f"  Checkpoint: {len(checkpoint)} cached")

    print("\n[Gate 2] Analyzing paths...")
    tasks = []
    for ep in gt_eps:
        eid = ep["episode_id"]
        if str(eid) in checkpoint:
            continue
        analysis = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        task = build_visual_task_v3(ep, analysis, lm_map.get(eid))
        tasks.append(task)
    print(f"  {len(tasks)} to generate ({len(checkpoint)} cached)")

    if tasks:
        print(f"\n[Gate 4 v3] Generating (concurrency={args.concurrency}, temp=0.5)...")
        t0 = time.time()
        new_results = await generate_v3_instructions(tasks, checkpoint, args.concurrency)
        checkpoint.update({str(k): v for k, v in new_results.items()})
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f)
        elapsed = time.time() - t0
        print(f"  Done: {len(new_results)} in {elapsed:.1f}s ({len(new_results)/elapsed:.1f}/s)")

    all_gen = {int(k): v for k, v in checkpoint.items()}
    ok = sum(1 for v in all_gen.values() if quality_check_v3(clean_output(v))[0])
    words = [len(clean_output(v).split()) for v in all_gen.values()]
    import statistics
    avg_words = statistics.mean(words) if words else 0
    print(f"\n[Quality] {ok}/{len(all_gen)} pass  avg_words={avg_words:.1f} (GT=26.8, v2=23.6)")

    if args.analyze:
        from collections import Counter
        starts = [clean_output(v).split()[0].lower() for v in all_gen.values() if clean_output(v)]
        cnt = Counter(starts).most_common(12)
        print(f"[Starting verbs] {cnt}")
        print(f"  GT: walk(34%) go(19%) turn(16%) exit(11%) leave(4%) head(2%)")

    print("\n[Samples] First 5:")
    for ep in gt_eps[:5]:
        eid = ep["episode_id"]
        raw = all_gen.get(eid, "MISSING")
        v3_text = clean_output(raw) if raw != "MISSING" else "MISSING"
        gt_text = ep["instruction"]["instruction_text"].strip()
        print(f"\n  Ep {eid}:")
        print(f"    GT: {gt_text[:85]}")
        print(f"    v3: {v3_text[:85]}")

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
            "mode": "visual_gemma_v3",
            "model": VLLM_MODEL,
            "version": "v3_diverse_starts",
            "n_episodes": len(assembled),
            "avg_words": avg_words,
            "quality_ok": ok,
            "target_style": "Diverse starts matching GT: walk/go/turn/exit/head distribution",
        },
    }
    save_dataset(dataset, output_path)

    print(f"\n=== Done ===")
    print(f"  v3 dataset: {output_path}")
    print(f"  Episodes:   {len(assembled)}/{len(gt_eps)}")
    print(f"  Avg words:  {avg_words:.1f}")
    print(f"\nRun --analyze flag to check starting verb distribution.")
    print(f"\nTo evaluate v3 instructions:")
    print(f"  bash habitat_eval/scripts/run_eval_gate7_visual_v3.sh")


if __name__ == "__main__":
    asyncio.run(main())
