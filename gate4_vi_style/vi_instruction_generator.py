#!/usr/bin/env python3
"""
Gate 4 VI-Style: Visually Impaired Navigation Instruction Generator

Converts visual (sighted) navigation instructions into VI-friendly style:
  - Explicit step counts ("Walk 10 steps" instead of "Walk straight")
  - Turn precision ("Turn 90° to your left" instead of "Turn left")
  - Tactile cues (wall guidance, floor texture changes, door frames)
  - Proximity/audio cues (counting up stairs, sensing open spaces)
  - No/minimal visual landmark dependence

This prepares instructions for Task 4: fine-tuning the navigation model
to generate VI-accessible instructions from the same visual input.

Usage:
  python3 gate4_vi_style/vi_instruction_generator.py \
      --visual-dataset outputs/datasets/val_unseen_generated_gemma_visual.json.gz \
      --landmarks-dir outputs/gate3_landmarks \
      --output outputs/datasets/val_unseen_generated_vi_style.json.gz \
      --n-episodes 100  # optional limit for testing
"""
import argparse
import asyncio
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY

VI_STYLE_RULES = """VISUALLY IMPAIRED NAVIGATION INSTRUCTION RULES:
1. Count approximate steps (assume 0.7m per step): "Walk 4 steps" not "Walk 2.8m"
2. Turn angles explicitly: "Turn 90° left" or "Turn left at a right angle"
3. Include tactile cues: "Feel the wall on your right", "When you reach the doorframe"
4. Describe floor changes: "When the floor changes from tile to carpet, stop"
5. Count stairs: "Climb 8 steps up the stairs" not "Go up the stairs"
6. Reference near-body landmarks: "furniture at waist height on your left"
7. No color references (avoid "the grey couch") — use "a soft seating piece"
8. Include door frame detection: "Pass through the doorway ahead"
9. Cardinal/clock directions if applicable: "at your 2 o'clock"
10. Clear stop condition: "Stop when you feel/hear/sense [specific cue]"
"""

VI_CONVERSION_PROMPT = """You are a navigation assistant for visually impaired people.
Convert the following navigation instruction and path data into a VI-accessible version.

ORIGINAL VISUAL INSTRUCTION:
{visual_instruction}

PATH DATA:
- Total distance: {distance_m:.1f}m (~{distance_steps} steps)
- Motion sequence: {motion_sequence}
- Start room: {start_room}
- Destination room: {goal_room}
- Has stairs: {has_stairs}

{rules}

Write ONLY the VI-accessible instruction text (2-4 sentences). No preamble:"""


def estimate_steps(distance_m: float, step_length_m: float = 0.7) -> int:
    """Convert meters to approximate step count."""
    return max(1, round(distance_m / step_length_m))


def motion_to_vi_sequence(primitives: List[Dict]) -> str:
    """Convert path primitives to VI-friendly sequence description."""
    parts = []
    total_dist = 0.0
    stair_ups = 0
    stair_downs = 0

    for p in primitives:
        if p["type"] == "straight":
            d = p.get("distance_m", 0)
            total_dist += d
            steps = estimate_steps(d)
            parts.append(f"walk {steps} steps straight")
        elif p["type"] == "left_turn":
            angle = p.get("angle_deg", 90)
            if p.get("sharp"):
                parts.append(f"turn {angle:.0f}° to your left (sharp)")
            else:
                parts.append(f"turn {angle:.0f}° to your left")
        elif p["type"] == "right_turn":
            angle = p.get("angle_deg", 90)
            if p.get("sharp"):
                parts.append(f"turn {angle:.0f}° to your right (sharp)")
            else:
                parts.append(f"turn {angle:.0f}° to your right")
        elif p["type"] == "elevation":
            if p.get("direction") == "up":
                stair_ups += 1
                h = p.get("change_m", 0)
                n_stairs = max(1, round(h / 0.18))  # typical stair rise = 18cm
                parts.append(f"climb {n_stairs} steps up")
            elif p.get("direction") == "down":
                stair_downs += 1
                h = p.get("change_m", 0)
                n_stairs = max(1, round(h / 0.18))
                parts.append(f"descend {n_stairs} steps down")

    return " → ".join(parts)


def build_vi_task(
    episode: Dict,
    visual_instruction: str,
    path_analysis: Dict,
    landmark_data: Optional[Dict],
) -> Dict:
    """Build task for VI instruction conversion."""
    summary = path_analysis["summary"]
    primitives = path_analysis["primitives"]

    total_dist = summary.get("total_distance_m", 0) or 0
    total_steps = estimate_steps(total_dist)
    has_stairs = any(p["type"] == "elevation" for p in primitives)
    motion_vi = motion_to_vi_sequence(primitives)

    start_room = "indoor space"
    goal_room = "destination area"

    if landmark_data:
        sc = landmark_data.get("scene_context") or {}
        gl = landmark_data.get("goal_landmark") or {}
        if isinstance(sc, dict) and "error" not in sc:
            start_room = sc.get("room_type", start_room)
        if isinstance(gl, dict) and "error" not in gl:
            goal_room = gl.get("room_type", goal_room)

    prompt = VI_CONVERSION_PROMPT.format(
        visual_instruction=visual_instruction,
        distance_m=total_dist,
        distance_steps=total_steps,
        motion_sequence=motion_vi,
        start_room=start_room,
        goal_room=goal_room,
        has_stairs="YES — count stairs carefully" if has_stairs else "NO",
        rules=VI_STYLE_RULES,
    )

    return {
        "episode_id": episode["episode_id"],
        "_vi_prompt": prompt,
        "motion_sequence": motion_vi,
    }


def quality_check_vi(text: str) -> tuple:
    """Check if instruction has VI-style characteristics."""
    words = text.split()
    failures = []
    if len(words) < 8:
        failures.append(f"too_short({len(words)}w)")
    if len(words) > 80:
        failures.append(f"too_long({len(words)}w)")

    # Should have tactile/count cues
    vi_cues = ["step", "steps", "stair", "stairs", "feel", "floor", "wall",
               "doorway", "door", "turn", "°", "degree", "stop", "wait", "pause"]
    if not any(c in text.lower() for c in vi_cues):
        failures.append("no_vi_cues")

    return len(failures) == 0, failures


async def generate_vi_instructions(
    tasks: List[Dict], checkpoint: Dict, concurrency: int = 10
) -> Dict[int, str]:
    """Generate VI-style instructions via Gemma."""
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
        prompt = task["_vi_prompt"]
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=VLLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.3,
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
        if done % 100 == 0 or done == total:
            print(f"  [{done}/{total}] rate={rate:.1f}/s ETA={eta/60:.1f}m")

    return results


async def main():
    from gate2_path.path_analyzer import analyze_path
    from gate5_tokenizer.tokenizer import VLNTokenizer
    from gate6_assembler.assembler import assemble_episode, save_dataset

    p = argparse.ArgumentParser()
    p.add_argument("--visual-dataset",
                   default=str(ROOT / "outputs/datasets/val_unseen_generated_gemma_visual.json.gz"))
    p.add_argument("--gt-path",
                   default="/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
    p.add_argument("--landmarks-dir",
                   default=str(ROOT / "outputs/gate3_landmarks"))
    p.add_argument("--output",
                   default=str(ROOT / "outputs/datasets/val_unseen_generated_vi_style.json.gz"))
    p.add_argument("--checkpoint",
                   default=str(ROOT / "outputs/gate4_vi_checkpoint.json"))
    p.add_argument("--n-episodes", type=int, default=None)
    args = p.parse_args()

    print("=== Gate 4 VI-Style: Visually Impaired Navigation Instructions ===")
    print(f"  Input:  {args.visual_dataset}")
    print(f"  Output: {args.output}")
    print()

    # Load visual dataset (the instructions to convert)
    with gzip.open(args.visual_dataset, "rt") as f:
        visual_data = json.load(f)
    visual_eps = visual_data["episodes"]

    # Load GT for structural data + path analyzer
    with gzip.open(args.gt_path, "rt") as f:
        gt_data = json.load(f)
    gt_eps = {ep["episode_id"]: ep for ep in gt_data["episodes"]}

    # Load landmarks
    landmarks_dir = Path(args.landmarks_dir)
    landmark_map = {}
    for lf in landmarks_dir.glob("episode_*.json"):
        with open(lf) as f:
            ld = json.load(f)
        landmark_map[ld["episode_id"]] = ld
    print(f"  {len(landmark_map)} episodes have landmark data")

    # Load checkpoint
    checkpoint = {}
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        print(f"  Checkpoint: {len(checkpoint)} already generated")

    # Build tasks
    print("\n[Building VI tasks...]")
    episodes_to_process = visual_eps[:args.n_episodes] if args.n_episodes else visual_eps
    tasks = []
    for ep in episodes_to_process:
        eid = ep["episode_id"]
        eid_str = str(eid)
        if eid_str in checkpoint:
            continue

        visual_instr = ep["instruction"]["instruction_text"]
        gt_ep = gt_eps.get(eid, {})
        analysis = analyze_path(gt_ep.get("reference_path", []), gt_ep.get("start_rotation"))
        landmark_data = landmark_map.get(eid)
        task = build_vi_task(ep, visual_instr, analysis, landmark_data)
        tasks.append(task)

    print(f"  {len(tasks)} to generate ({len(checkpoint)} cached)")

    # Generate VI instructions
    if tasks:
        print(f"\n[Gate 4 VI] Generating instructions (concurrency=10)...")
        new_results = await generate_vi_instructions(tasks, checkpoint, concurrency=10)
        checkpoint.update({str(k): v for k, v in new_results.items()})
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f)
        print(f"  Checkpoint saved: {len(checkpoint)}")

    # Quality check + show samples
    all_generated = {int(k): v for k, v in checkpoint.items()}
    ok_count = sum(1 for v in all_generated.values() if quality_check_vi(v.strip())[0])
    print(f"\n[Quality] {ok_count}/{len(all_generated)} pass VI quality check")

    print("\n[Samples] VI instructions (first 3):")
    for ep in episodes_to_process[:3]:
        eid = ep["episode_id"]
        visual = ep["instruction"]["instruction_text"]
        vi = all_generated.get(eid, "MISSING").strip()
        gt_ep = gt_eps.get(eid, {})
        gt_instr = gt_ep.get("instruction", {}).get("instruction_text", "")
        print(f"\n  Episode {eid}:")
        print(f"    Visual:   {visual}")
        print(f"    VI-style: {vi}")
        print(f"    GT:       {gt_instr[:80]}")

    # Assemble VI dataset
    print("\n[Assembling VI dataset...]")
    tok = VLNTokenizer(args.gt_path)
    assembled = []
    for ep in episodes_to_process:
        eid = ep["episode_id"]
        vi_text = all_generated.get(eid, "").strip()
        if not vi_text or vi_text.startswith("ERROR"):
            vi_text = ep["instruction"]["instruction_text"]  # fallback to visual
        gt_ep = gt_eps.get(eid, ep)
        assembled.append(assemble_episode(gt_ep, vi_text, tok))

    dataset = {
        "episodes": assembled,
        "instruction_vocab": gt_data.get("instruction_vocab", {}),
        "_generation_meta": {
            "mode": "vi_style_gemma",
            "model": VLLM_MODEL,
            "n_episodes": len(assembled),
            "quality_ok": ok_count,
            "style": "visually_impaired_indoor_navigation",
        },
    }
    save_dataset(dataset, Path(args.output))

    print(f"\n=== Done ===")
    print(f"VI dataset: {args.output}")
    print(f"Episodes:   {len(assembled)}")
    print(f"Quality OK: {ok_count}/{len(all_generated)} ({100*ok_count/max(1,len(all_generated)):.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
