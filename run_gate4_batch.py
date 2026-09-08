#!/usr/bin/env python3
"""
Gate 4 Batch Runner — text-only mode
Processes all VLN-CE val_unseen episodes:
  Gate 2 (path analysis) → Gate 4 (Gemma instruction generation) → Gate 5+6 (assemble dataset)

Runtime: ~1839 episodes / 16 concurrent / ~0.8s per call ≈ ~5-10 min
Output: outputs/datasets/val_unseen_generated_gemma.json.gz
"""
import asyncio
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma.json.gz"
CHECKPOINT_PATH = ROOT / "outputs" / "gate4_checkpoint.json"

from gate2_path.path_analyzer import analyze_path, primitives_to_text
from gate4_instructions.gemma_vllm_backend import generate_batch_async
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset


def quality_check(text: str) -> tuple:
    words = text.split()
    failures = []
    if len(words) < 10:
        failures.append(f"short({len(words)}w)")
    if len(words) > 60:
        failures.append(f"long({len(words)}w)")
    stop_words = ["stop", "wait", "halt", "stand", "pause"]
    if not any(w in text.lower() for w in stop_words):
        failures.append("no_stop")
    return len(failures) == 0, failures


def clean_output(raw: str) -> str:
    """Strip common model preambles, take first 4 sentences."""
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

    print("=== S2 Pipeline: Gate 4 Batch Runner ===")
    print(f"Model:  cyankiwi/gemma-4-31B-it-AWQ-4bit @ 10.77.32.231:8000")
    print(f"Mode:   text-only (path geometry → instruction, no rendered frames)")
    print(f"Output: {OUTPUT_PATH}")
    print()

    # Load GT episodes (structural metadata only — instructions NOT used in generation)
    print("Loading GT episodes (structural metadata only)...")
    with gzip.open(GT_PATH, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]
    print(f"  {len(episodes)} episodes loaded")

    # Load checkpoint if exists
    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        print(f"  Checkpoint: {len(checkpoint)} episodes already generated")

    # Gate 2: path analysis for all episodes
    print("\n[Gate 2] Analyzing paths...")
    tasks = []
    for ep in episodes:
        eid = str(ep["episode_id"])
        if eid in checkpoint:
            continue  # already done
        analysis = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        motion = primitives_to_text(analysis["primitives"])
        # Build richer scene context from path characteristics
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
            f"Indoor path: {summary['path_type'].replace('_',' ')}, "
            f"total ~{summary['total_distance_m']:.1f}m, "
            f"{', '.join(turns) if turns else 'no significant turns'}.{elev_desc}"
        )
        tasks.append({
            "episode_id": ep["episode_id"],
            "motion_sequence": motion,
            "scene_context": scene_context,
        })

    print(f"  {len(tasks)} episodes to generate ({len(checkpoint)} already cached)")

    # Gate 4: async batch generation
    if tasks:
        print(f"\n[Gate 4] Generating instructions via Gemma 4 31B AWQ (concurrency=16)...")
        t0 = time.time()
        new_results = await generate_batch_async(tasks, concurrency=16, progress_every=100)

        # Save checkpoint
        checkpoint.update({str(k): v for k, v in new_results.items()})
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f)
        print(f"  Checkpoint saved: {len(checkpoint)} total")
    else:
        new_results = {}
        print("  All episodes already in checkpoint — skipping generation")

    # Quality check
    all_generated = {int(k): v for k, v in checkpoint.items()}
    ok_count = sum(1 for t in all_generated.values() if quality_check(clean_output(t))[0])
    print(f"\n[Quality] {ok_count}/{len(all_generated)} pass quality check")

    # Show 5 examples
    print("\n[Samples] First 5 generated instructions:")
    for ep in episodes[:5]:
        eid = ep["episode_id"]
        raw = all_generated.get(eid, "MISSING")
        clean = clean_output(raw) if raw != "MISSING" else "MISSING"
        ok, failures = quality_check(clean)
        analysis = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        motion = primitives_to_text(analysis["primitives"])
        print(f"\n  Episode {eid}:")
        print(f"    Path:      {motion}")
        print(f"    GT:        {ep['instruction']['instruction_text'].strip()}")
        print(f"    Generated: {clean}")
        print(f"    Quality:   {'OK' if ok else 'FAIL: ' + str(failures)}")

    # Gate 5 + 6: tokenize and assemble dataset
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

    # Load vocab for output
    instruction_vocab = data.get("instruction_vocab", {})
    dataset = {
        "episodes": assembled,
        "instruction_vocab": instruction_vocab,
        "_generation_meta": {
            "mode": "text_only_gemma",
            "model": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
            "n_episodes": len(assembled),
            "n_skipped": len(episodes) - len(assembled),
            "quality_ok": ok_count,
            "quality_fail": len(all_generated) - ok_count,
        },
    }
    save_dataset(dataset, OUTPUT_PATH)

    print(f"\n=== Done ===")
    print(f"Generated dataset: {OUTPUT_PATH}")
    print(f"Episodes: {len(assembled)}/{len(episodes)}")
    print(f"Quality OK: {ok_count}/{len(all_generated)} ({100*ok_count/max(1,len(all_generated)):.1f}%)")
    print(f"\nNext: run Gate 7 evaluation")
    print(f"  bash gate7_eval/run_habitat_eval.sh {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
