#!/usr/bin/env python3
"""
Gate 6: Episode Assembler
Combines path metadata + generated instruction + tokens into a GT-compatible VLN episode JSON.
Outputs .json.gz files that drop into Habitat/Isaac Lab eval configs unchanged.

Inputs:
  - Original episode structural data (start_position, reference_path, goals, scene_id, etc.)
  - Generated instruction_text (from Gate 4)
  - instruction_tokens (from Gate 5)
Output:
  - val_unseen_generated.json.gz in GT format
"""
import gzip
import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from gate5_tokenizer.tokenizer import VLNTokenizer


def compute_geodesic_distance(reference_path: List[List[float]]) -> float:
    """Approximate geodesic distance as sum of Euclidean segment lengths."""
    total = 0.0
    for i in range(len(reference_path) - 1):
        p1, p2 = reference_path[i], reference_path[i + 1]
        total += math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    return round(total, 6)


def assemble_episode(
    source_episode: Dict,
    generated_text: str,
    tokenizer: VLNTokenizer,
    episode_id: Optional[int] = None,
) -> Dict:
    """
    Build one complete episode dict from source metadata + generated instruction.

    Source episode provides all structural fields (position, path, scene, etc.)
    Only instruction_text and instruction_tokens are replaced with generated values.
    """
    tokens = tokenizer.encode(generated_text)

    ep = {
        "episode_id": episode_id if episode_id is not None else source_episode["episode_id"],
        "trajectory_id": source_episode.get("trajectory_id", 0),
        "scene_id": source_episode["scene_id"],
        "start_position": source_episode["start_position"],
        "start_rotation": source_episode["start_rotation"],
        "info": {
            "geodesic_distance": source_episode.get("info", {}).get(
                "geodesic_distance",
                compute_geodesic_distance(source_episode["reference_path"])
            )
        },
        "goals": source_episode["goals"],
        "instruction": {
            "instruction_text": generated_text,
            "instruction_tokens": tokens,
        },
        "reference_path": source_episode["reference_path"],
    }
    return ep


def assemble_dataset(
    source_episodes: List[Dict],
    generated_texts: Dict[int, str],
    tokenizer: VLNTokenizer,
    vocab_source_path: str,
) -> Dict:
    """
    Build complete dataset dict from source episodes and a mapping of
    episode_id → generated_text.

    Episodes without a generated text are SKIPPED (logged as warnings).
    """
    # Load original vocab to include in output (maintains format compatibility)
    with gzip.open(vocab_source_path, "rt") as f:
        source_data = json.load(f)
    instruction_vocab = source_data.get("instruction_vocab", {})

    assembled_episodes = []
    skipped = []

    for ep in source_episodes:
        eid = ep["episode_id"]
        text = generated_texts.get(eid)
        if text is None or not text.strip():
            skipped.append(eid)
            continue
        assembled_episodes.append(assemble_episode(ep, text, tokenizer))

    if skipped:
        print(f"WARNING: Skipped {len(skipped)} episodes with no generated text: {skipped[:10]}...")

    return {
        "episodes": assembled_episodes,
        "instruction_vocab": instruction_vocab,
        "_generation_meta": {
            "source": "s2_pipeline_new/gate4_instructions",
            "n_episodes": len(assembled_episodes),
            "n_skipped": len(skipped),
        },
    }


def save_dataset(dataset: Dict, output_path: Path) -> None:
    """Save assembled dataset as .json.gz."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)
    size_kb = output_path.stat().st_size / 1024
    print(f"Saved {len(dataset['episodes'])} episodes to {output_path} ({size_kb:.1f} KB)")


def load_generated_texts(texts_json: Path) -> Dict[int, str]:
    """Load generated texts from a {episode_id: instruction_text} JSON file."""
    with open(texts_json) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


if __name__ == "__main__":
    import argparse

    GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
    OUTPUT_PATH = Path("/home/kemal/VLNav/s2_pipeline_new/outputs/datasets/val_unseen_generated.json.gz")

    parser = argparse.ArgumentParser(description="Assemble generated VLN dataset")
    parser.add_argument("--texts-json", type=Path, required=False,
                        help="JSON file mapping episode_id → generated instruction_text")
    parser.add_argument("--gt-path", type=Path, default=GT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    # Load source episodes
    with gzip.open(args.gt_path, "rt") as f:
        data = json.load(f)
    source_episodes = data["episodes"]
    print(f"Source: {len(source_episodes)} episodes from {args.gt_path}")

    # Load tokenizer
    tok = VLNTokenizer(str(args.gt_path))
    print(f"Tokenizer: vocab size = {tok.num_vocab}")

    if args.texts_json:
        generated_texts = load_generated_texts(args.texts_json)
        print(f"Loaded {len(generated_texts)} generated texts from {args.texts_json}")
        dataset = assemble_dataset(source_episodes, generated_texts, tok, str(args.gt_path))
        save_dataset(dataset, args.output)
    else:
        # Demo: show what 3 assembled episodes would look like
        print("\n=== Demo: assemble first 3 episodes with placeholder text ===")
        demo_texts = {
            ep["episode_id"]: f"[PLACEHOLDER — run Gate 4 first] Path: {len(ep['reference_path'])} waypoints."
            for ep in source_episodes[:3]
        }
        dataset = assemble_dataset(source_episodes[:3], demo_texts, tok, str(args.gt_path))
        print(json.dumps(dataset["episodes"][0], indent=2)[:800])
        print("\nRun with --texts-json <path> after Gate 4 generates instructions.")
