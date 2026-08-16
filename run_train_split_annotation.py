#!/usr/bin/env python3
"""
R2R Train Split Annotation Pipeline — text-only mode
Annotates all 10,819 R2R train episodes with auto-generated instructions.
No GPU required — uses Gemma 4 31B AWQ API only.

This creates training data labels for Task 3 (InternNav training).
The resulting dataset supplements GT VLN-CE training data.

Output: outputs/datasets/train_generated_gemma_visual.json.gz
        (Well, text-only actually, since no train-split rendering done yet.)
"""
import asyncio
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# R2R train split data path
TRAIN_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/train/train.json.gz"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "train_generated_gemma_textonly.json.gz"
CHECKPOINT_PATH = ROOT / "outputs" / "train_gate4_checkpoint.json"
LANDMARKS_DIR = ROOT / "outputs" / "gate3_train_landmarks"  # not yet generated

from gate2_path.path_analyzer import analyze_path, primitives_to_text
from gate4_instructions.gemma_vllm_backend import generate_batch_async
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset


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


async def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=== R2R Train Split Auto-Annotation (Text-Only Mode) ===")
    print(f"Model:    Gemma 4 31B AWQ @ 10.77.32.231:8000")
    print(f"Mode:     text-only (path geometry → instruction)")
    print(f"Input:    {TRAIN_PATH}")
    print(f"Output:   {OUTPUT_PATH}")
    print()

    # Load train episodes
    print("Loading train episodes...")
    with gzip.open(TRAIN_PATH, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]
    print(f"  {len(episodes)} train episodes loaded")

    # Load checkpoint if exists
    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        print(f"  Checkpoint: {len(checkpoint)} already generated")

    # Gate 2: path analysis for all episodes
    print("\n[Gate 2] Analyzing train paths...")
    tasks = []
    for ep in episodes:
        eid = str(ep["episode_id"])
        if eid in checkpoint:
            continue

        analysis = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        motion = primitives_to_text(analysis["primitives"])
        summary = analysis["summary"]

        turns = []
        if summary["n_left_turns"] > 0:
            turns.append(f"{summary['n_left_turns']} left turn(s)")
        if summary["n_right_turns"] > 0:
            turns.append(f"{summary['n_right_turns']} right turn(s)")
        elev = summary["elevation_change_m"]
        elev_desc = ""
        if elev > 0.5:
            elev_desc = f" Path goes UP {elev:.1f}m (stairs likely)."
        elif elev < -0.5:
            elev_desc = f" Path goes DOWN {elev:.1f}m (stairs likely)."

        scene_context = (
            f"Indoor path: {summary['path_type'].replace('_', ' ')}, "
            f"total ~{summary['total_distance_m']:.1f}m, "
            f"{', '.join(turns) if turns else 'no significant turns'}.{elev_desc}"
        )
        tasks.append({
            "episode_id": ep["episode_id"],
            "motion_sequence": motion,
            "scene_context": scene_context,
        })

    print(f"  {len(tasks)} to generate ({len(checkpoint)} already cached)")

    # Gate 4: async batch generation (concurrency=16 for maximum throughput)
    if tasks:
        print(f"\n[Gate 4] Generating instructions (concurrency=16)...")
        t0 = time.time()
        new_results = await generate_batch_async(tasks, concurrency=16, progress_every=500)
        checkpoint.update({str(k): v for k, v in new_results.items()})
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f)
        elapsed = time.time() - t0
        print(f"  Completed {len(new_results)} in {elapsed:.1f}s ({len(new_results)/elapsed:.1f}/s)")

    # Quality check
    all_generated = {int(k): v for k, v in checkpoint.items()}
    ok = sum(1 for v in all_generated.values() if quality_check(clean_output(v))[0])
    print(f"\n[Quality] {ok}/{len(all_generated)} pass ({100*ok/max(1,len(all_generated)):.1f}%)")

    # Show 3 samples from train
    print("\n[Samples] First 3 train instructions:")
    for ep in episodes[:3]:
        eid = ep["episode_id"]
        raw = all_generated.get(eid, "MISSING")
        clean = clean_output(raw) if raw != "MISSING" else "MISSING"
        analysis = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        motion = primitives_to_text(analysis["primitives"])
        gt_text = ep["instruction"]["instruction_text"].strip()
        print(f"\n  Ep {eid} ({ep['scene_id'].split('/')[1]}):")
        print(f"    Path:     {motion}")
        print(f"    GT:       {gt_text[:80]}")
        print(f"    Generated:{clean}")

    # Gate 5+6: assemble dataset
    print("\n[Gate 5] Loading tokenizer (using val_unseen vocab)...")
    VAL_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
    tok = VLNTokenizer(VAL_PATH)

    print("[Gate 6] Assembling train dataset...")
    assembled = []
    for ep in episodes:
        eid = ep["episode_id"]
        raw = all_generated.get(eid, "")
        text = clean_output(raw) if raw else ""
        if not text:
            continue
        assembled.append(assemble_episode(ep, text, tok))

    dataset = {
        "episodes": assembled,
        "instruction_vocab": data.get("instruction_vocab", {}),
        "_generation_meta": {
            "mode": "text_only_gemma",
            "model": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
            "split": "train",
            "n_episodes": len(assembled),
            "n_skipped": len(episodes) - len(assembled),
            "quality_ok": ok,
        },
    }
    save_dataset(dataset, OUTPUT_PATH)

    print(f"\n=== Done ===")
    print(f"Train annotation: {OUTPUT_PATH}")
    print(f"Episodes: {len(assembled)}/{len(episodes)}")
    print(f"\nNext steps:")
    print(f"  1. Run Gate 1+3 rendering on train split for visual annotations")
    print(f"     bash gate1_renderer/render_batch_train.sh")
    print(f"  2. Combine with GT training trajectories for model training")


if __name__ == "__main__":
    asyncio.run(main())
