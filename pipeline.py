#!/usr/bin/env python3
"""
S2 Pipeline — End-to-End Runner
Processes VLN episodes from images+paths → GT-quality annotations.

Usage:
  # Text-only mode (Gates 2 + 4 + 5 + 6 only — no rendering needed):
  python3 pipeline.py --mode text_only --split val_unseen --n-episodes 100

  # Full mode (Gates 1 + 2 + 3 + 4 + 5 + 6):
  python3 pipeline.py --mode full --split val_unseen --render

  # From rosbag (Gate 8 → 2 + 3 + 4 + 5 + 6):
  python3 pipeline.py --mode rosbag --bag /path/to/recording.bag
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT))

GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
OUTPUT_DIR = PIPELINE_ROOT / "outputs"


def run_text_only(episodes, backend: str, output_path: Path, max_ep: int = None):
    """
    Text-only pipeline: Gates 2 + 4(text) + 5 + 6.
    No rendering needed. Uses path geometry only for instruction generation.
    Fastest path to generating a first dataset for Gate 7 evaluation.
    """
    from gate2_path.path_analyzer import analyze_path, primitives_to_text
    from gate5_tokenizer.tokenizer import VLNTokenizer
    from gate6_assembler.assembler import assemble_episode, save_dataset

    import yaml
    with open(PIPELINE_ROOT / "configs" / "vlm_prompts.yaml") as f:
        prompts = yaml.safe_load(f)
    prompt_template = prompts["instruction_generation_text_only"]

    tok = VLNTokenizer(GT_PATH)
    print(f"Tokenizer loaded: {tok.num_vocab} vocab")

    generated_texts = {}
    quality_stats = {"ok": 0, "fail": 0, "total": 0}
    t0 = time.time()

    episodes_to_process = episodes[:max_ep] if max_ep else episodes
    print(f"Processing {len(episodes_to_process)} episodes in text-only mode (backend={backend})...")

    if backend in ("gemma", "gpt4o"):
        from gate4_instructions.instruction_generator import InstructionGenerator, quality_check
        gen = InstructionGenerator(backend=backend)

        for i, ep in enumerate(episodes_to_process):
            path_analysis = analyze_path(ep["reference_path"], ep.get("start_rotation"))
            motion_text = primitives_to_text(path_analysis["primitives"])
            result = gen.generate(ep, frames_dir=None, path_analysis=path_analysis)
            text = result["instruction_text"]
            ok, _ = quality_check(text)
            generated_texts[ep["episode_id"]] = text
            quality_stats["total"] += 1
            if ok:
                quality_stats["ok"] += 1
            else:
                quality_stats["fail"] += 1
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(episodes_to_process) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(episodes_to_process)}] OK={quality_stats['ok']} "
                      f"FAIL={quality_stats['fail']} rate={rate:.1f} ep/s ETA={eta/60:.1f}m")
    else:
        print(f"ERROR: Unknown backend '{backend}'. Use 'gemma' or 'gpt4o'.")
        sys.exit(1)

    # Assemble dataset
    source_ep_map = {ep["episode_id"]: ep for ep in episodes}
    assembled = []
    for eid, text in generated_texts.items():
        ep = assemble_episode(source_ep_map[eid], text, tok)
        assembled.append(ep)

    # Load vocab
    with gzip.open(GT_PATH, "rt") as f:
        source_data = json.load(f)
    dataset = {
        "episodes": assembled,
        "instruction_vocab": source_data.get("instruction_vocab", {}),
        "_generation_meta": {
            "mode": "text_only",
            "backend": backend,
            "n_episodes": len(assembled),
            "quality_ok": quality_stats["ok"],
            "quality_fail": quality_stats["fail"],
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
    save_dataset(dataset, output_path)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f}m")
    print(f"Quality: {quality_stats['ok']}/{quality_stats['total']} passed ({100*quality_stats['ok']/max(1,quality_stats['total']):.1f}%)")
    print(f"Output: {output_path}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="S2 Pipeline: images+paths → VLN annotations")
    parser.add_argument("--mode", choices=["text_only", "full", "rosbag"], default="text_only")
    parser.add_argument("--backend", choices=["gemma", "gpt4o"], default="gemma",
                        help="VLM backend for instruction generation")
    parser.add_argument("--split", default="val_unseen",
                        help="Dataset split to process")
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="Limit number of episodes (None = all)")
    parser.add_argument("--gt-path", default=GT_PATH,
                        help="Path to GT dataset .json.gz")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .json.gz path")
    parser.add_argument("--bag", type=str, default=None,
                        help="Rosbag path (for --mode rosbag)")
    args = parser.parse_args()

    output_path = args.output or (OUTPUT_DIR / "datasets" / f"{args.split}_generated_{args.backend}.json.gz")

    print("=== S2 Pipeline: Auto-Annotation of VLN Episodes ===")
    print(f"Mode:    {args.mode}")
    print(f"Backend: {args.backend}")
    print(f"Output:  {output_path}")
    print()

    if args.mode == "text_only":
        with gzip.open(args.gt_path, "rt") as f:
            data = json.load(f)
        episodes = data["episodes"]
        print(f"Loaded {len(episodes)} source episodes (structural metadata only — NO instructions used)")
        run_text_only(episodes, args.backend, output_path, args.n_episodes)

    elif args.mode == "full":
        print("Full mode (with rendering) — ensure Habitat-Sim is available")
        print("Run inside vlnav/habitat-eval:rebuilt container")
        # Gate 1 → render → Gate 2+3+4+5+6
        # TODO: implement after Gate 1 rendering is validated
        print("Not yet implemented. Run text_only mode first.")

    elif args.mode == "rosbag":
        if not args.bag:
            print("ERROR: --bag required for rosbag mode")
            sys.exit(1)
        from gate8_adapters.rosbag_adapter import RosbagAdapter
        adapter = RosbagAdapter(args.bag)
        frames_dir = OUTPUT_DIR / "rendered_frames"
        episode = adapter.extract(frames_dir, episode_id=0)
        print(f"Extracted: {episode['n_frames']} frames")
        # Continue with Gates 2-6...
        print("Continuing with Gates 2-6 for the extracted episode...")

    print("\nNext step: run Gate 7 evaluation")
    print(f"  bash gate7_eval/run_habitat_eval.sh {output_path}")


if __name__ == "__main__":
    main()
