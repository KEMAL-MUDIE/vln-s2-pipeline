#!/usr/bin/env python3
"""
Gate 4 Visual Batch Runner — visual mode
Generates VLN instructions using Gemma 4 31B AWQ with:
  - Gate 2: path geometry (motion primitives)
  - Gate 3: landmark context (room type, visible landmarks, stop landmark from images)
  - Gate 4: Gemma instruction generation with both inputs

This produces higher-quality instructions vs text-only mode because:
  - Knows the room type (bedroom vs hallway vs kitchen)
  - Knows specific landmarks (gray couch, wooden stairs, glass door)
  - Knows exactly what to stop near (stop landmark from goal frame)

Output: outputs/datasets/val_unseen_generated_gemma_visual.json.gz
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
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual.json.gz"
CHECKPOINT_PATH = ROOT / "outputs" / "gate4_visual_checkpoint.json"

from gate2_path.path_analyzer import analyze_path, primitives_to_text
from gate4_instructions.gemma_vllm_backend import (
    VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY, generate_batch_async
)
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset


VISUAL_INSTRUCTION_PROMPT = """You are a navigation instruction writer for an indoor robot assistant.
Write a single natural navigation instruction based on the path geometry and visual scene context below.

PATH GEOMETRY (motion sequence from start to goal):
{motion_sequence}

SCENE CONTEXT (visual analysis of key frames):
Starting room: {start_room}
Visible landmarks at start: {start_landmarks}
Navigation via: {turn_context}
Destination room: {goal_room}
Stop landmark (goal): {stop_landmark}
Stop location hint: {stop_hint}

REQUIREMENTS:
- Write 2-4 sentences in second person ("Walk into...", "Turn left at...", "Stop near...")
- Name SPECIFIC landmarks from the scene context (use the exact objects provided)
- Follow the path geometry precisely (turns, distances, elevation changes)
- End with a clear stop instruction using the stop landmark
- Keep it natural and concise (10-50 words)
- Style: "Exit the [room], turn [direction]. Walk past the [landmark]. Stop near the [stop_landmark]."

Write ONLY the instruction text, no preamble:"""


def build_visual_task(episode: Dict, path_analysis: Dict, landmark_data: Optional[Dict]) -> Dict:
    """Construct task dict for visual instruction generation."""

    motion = primitives_to_text(path_analysis["primitives"])
    summary = path_analysis["summary"]

    # Default fallbacks if no landmark data
    start_room = "indoor space"
    start_landmarks = "furniture"
    turn_context = "through the building"
    goal_room = "destination area"
    stop_landmark = "the end of the path"
    stop_hint = ""

    if landmark_data:
        sc = landmark_data.get("scene_context") or {}
        gl = landmark_data.get("goal_landmark") or {}

        if isinstance(sc, dict) and "error" not in sc:
            start_room = sc.get("room_type", start_room)
            lms = sc.get("landmarks", [])
            if lms:
                start_landmarks = ", ".join(lms[:4])
            direction_hint = sc.get("direction_hint", "")

        if isinstance(gl, dict) and "error" not in gl:
            goal_room = gl.get("room_type", goal_room)
            stop_landmark = gl.get("stop_landmark", stop_landmark)
            stop_hint = gl.get("direction_hint", "")

    # Build turn context from primitives (types: left_turn, right_turn, elevation)
    turns = []
    has_elevation_up = False
    has_elevation_down = False
    for p in path_analysis["primitives"]:
        if p["type"] == "left_turn":
            label = "sharp left" if p.get("sharp") else "left"
            turns.append(f"turn {label}")
        elif p["type"] == "right_turn":
            label = "sharp right" if p.get("sharp") else "right"
            turns.append(f"turn {label}")
        elif p["type"] == "elevation":
            if p.get("direction") == "up":
                has_elevation_up = True
            elif p.get("direction") == "down":
                has_elevation_down = True

    elev = summary.get("elevation_change_m", 0)
    if has_elevation_up or (elev is not None and elev > 0.5):
        turns.insert(0, "go up stairs")
    elif has_elevation_down or (elev is not None and elev < -0.5):
        turns.insert(0, "go down stairs")

    turn_context = ", ".join(turns) if turns else "straight path"

    prompt = VISUAL_INSTRUCTION_PROMPT.format(
        motion_sequence=motion,
        start_room=start_room,
        start_landmarks=start_landmarks,
        turn_context=turn_context,
        goal_room=goal_room,
        stop_landmark=stop_landmark,
        stop_hint=stop_hint,
    )

    return {
        "episode_id": episode["episode_id"],
        "motion_sequence": motion,  # used by gemma_vllm_backend as main input
        "scene_context": "",         # not used in this path
        "_visual_prompt": prompt,    # full prompt override
    }


def quality_check(text: str) -> tuple:
    words = text.split()
    failures = []
    if len(words) < 10:
        failures.append(f"short({len(words)}w)")
    if len(words) > 70:
        failures.append(f"long({len(words)}w)")
    stop_words = ["stop", "wait", "halt", "stand", "pause"]
    if not any(w in text.lower() for w in stop_words):
        failures.append("no_stop")
    return len(failures) == 0, failures


def clean_output(raw: str) -> str:
    import re
    for prefix in ["Instruction:", "Navigation instruction:", "Here is", "Here's", "Answer:", "Sure"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].lstrip(":").strip()
    sentences = re.split(r'(?<=[.!?])\s+', raw.strip())
    clean = " ".join(sentences[:4]).strip()
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean


async def generate_visual_instructions(tasks: list, checkpoint: dict, concurrency: int = 12):
    """Call Gemma with full visual prompts (overriding the standard backend prompt)."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("ERROR: openai not installed.")
        sys.exit(1)

    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    sem = asyncio.Semaphore(concurrency)

    async def one_task(task):
        eid = task["episode_id"]
        if str(eid) in checkpoint:
            return eid, checkpoint[str(eid)]

        prompt = task.get("_visual_prompt", task["motion_sequence"])
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=VLLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.3,
                )
                text = resp.choices[0].message.content.strip()
                return eid, text
            except Exception as e:
                return eid, f"ERROR: {e}"

    results = {}
    total = len(tasks)
    t0 = time.time()
    done = 0

    coros = [one_task(t) for t in tasks]
    for coro in asyncio.as_completed(coros):
        eid, text = await coro
        results[eid] = text
        done += 1
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else 0
        if done % 100 == 0 or done == total:
            print(f"  [{done}/{total}] rate={rate:.1f}/s ETA={eta/60:.1f}m")

    return results


async def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=== S2 Pipeline: Gate 4 Visual Batch Runner ===")
    print(f"Model:  Gemma 4 31B AWQ (text) @ {VLLM_BASE_URL}")
    print(f"Mode:   visual — uses Gate 3 landmark context + path geometry")
    print(f"Output: {OUTPUT_PATH}")
    print()

    # Load GT episodes
    print("Loading GT episodes...")
    with gzip.open(GT_PATH, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]
    print(f"  {len(episodes)} episodes")

    # Load landmark data
    print("Loading Gate 3 landmark data...")
    landmark_map = {}
    if LANDMARKS_DIR.exists():
        for lf in LANDMARKS_DIR.glob("episode_*.json"):
            with open(lf) as f:
                ld = json.load(f)
            landmark_map[ld["episode_id"]] = ld
    print(f"  {len(landmark_map)} episodes have landmark data")

    # Load checkpoint
    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        print(f"  Checkpoint: {len(checkpoint)} already generated")

    # Gate 2: path analysis + task building
    print("\n[Gate 2+3] Building visual tasks...")
    tasks = []
    for ep in episodes:
        eid = str(ep["episode_id"])
        if eid in checkpoint:
            continue
        analysis = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        landmark_data = landmark_map.get(ep["episode_id"])
        task = build_visual_task(ep, analysis, landmark_data)
        tasks.append(task)

    n_with_landmarks = sum(1 for ep in episodes if ep["episode_id"] in landmark_map)
    print(f"  {len(tasks)} to generate ({len(checkpoint)} cached)")
    print(f"  {n_with_landmarks}/{len(episodes)} have visual landmark context")

    # Gate 4: generate instructions
    if tasks:
        print(f"\n[Gate 4] Generating visual instructions (concurrency=12)...")
        new_results = await generate_visual_instructions(tasks, checkpoint, concurrency=12)

        # Save checkpoint
        checkpoint.update({str(k): v for k, v in new_results.items()})
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f)
        print(f"  Checkpoint saved: {len(checkpoint)} total")

    # Quality check
    all_generated = {int(k): v for k, v in checkpoint.items()}
    ok_count = sum(1 for t in all_generated.values() if quality_check(clean_output(t))[0])
    print(f"\n[Quality] {ok_count}/{len(all_generated)} pass quality check")

    # Show 5 examples comparing text-only vs visual
    print("\n[Samples] Visual vs text-only comparison (first 5 with landmarks):")
    shown = 0
    for ep in episodes:
        eid = ep["episode_id"]
        if eid not in landmark_map:
            continue
        raw = all_generated.get(eid, "MISSING")
        clean = clean_output(raw) if raw != "MISSING" else "MISSING"
        gt = ep["instruction"]["instruction_text"].strip()
        lm = landmark_map[eid]
        sc = lm.get("scene_context") or {}
        gl = lm.get("goal_landmark") or {}

        analysis = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        motion = primitives_to_text(analysis["primitives"])

        print(f"\n  Episode {eid}:")
        print(f"    Path:      {motion}")
        if isinstance(sc, dict) and "error" not in sc:
            print(f"    Room:      {sc.get('room_type', '?')}, Landmarks: {sc.get('landmarks', [])[:3]}")
        if isinstance(gl, dict) and "error" not in gl:
            print(f"    Goal:      {gl.get('stop_landmark', '?')} ({gl.get('room_type', '?')})")
        print(f"    GT:        {gt[:100]}")
        print(f"    Generated: {clean}")
        shown += 1
        if shown >= 5:
            break

    # Gate 5 + 6: tokenize and assemble
    print("\n[Gate 5] Loading tokenizer...")
    tok = VLNTokenizer(GT_PATH)

    print("[Gate 6] Assembling dataset...")
    assembled = []
    for ep in episodes:
        eid = ep["episode_id"]
        raw = all_generated.get(eid, "")
        text = clean_output(raw) if raw else ""
        if not text:
            continue
        assembled.append(assemble_episode(ep, text, tok))

    instruction_vocab = data.get("instruction_vocab", {})
    dataset = {
        "episodes": assembled,
        "instruction_vocab": instruction_vocab,
        "_generation_meta": {
            "mode": "visual_gemma",
            "model": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
            "n_episodes": len(assembled),
            "n_with_landmarks": len(landmark_map),
            "n_skipped": len(episodes) - len(assembled),
            "quality_ok": ok_count,
        },
    }
    save_dataset(dataset, OUTPUT_PATH)

    print(f"\n=== Done ===")
    print(f"Generated dataset: {OUTPUT_PATH}")
    print(f"Episodes:         {len(assembled)}/{len(episodes)}")
    print(f"With landmarks:   {len(landmark_map)}")
    print(f"Quality OK:       {ok_count}/{len(all_generated)} ({100*ok_count/max(1,len(all_generated)):.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
