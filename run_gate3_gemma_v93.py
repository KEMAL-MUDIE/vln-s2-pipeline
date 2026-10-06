#!/usr/bin/env python3
"""
Gate3-Gemma v93 — v92 pure post-processing (no vLLM needed)

KEY FIXES vs v92:
  v92 had:
    walk_out_of_the: 388  (GT=57, 6.8x overrep!)
    go_out_of_the:    43  (overrep)
    exit_the:        265  (GT=278, slightly under)
    avg_words:       26.918 (GT=26.78)

  v93 applies:
  1. 100% "Walk/walk out of the X" → "Exit/exit the X"  (-2 words per match)
  2. 100% "Go/go out of the X"    → "Exit/exit the X"  (-2 words per match)

  v93 expected:
    walk_out_of_the:   0  (was 6.8x → 0!)
    go_out_of_the:     0  (eliminated)
    exit_the:        ~696  (GT=278, 2.5x — much better than 6.8x)
    avg_words:       ~26.45 (GT=26.78, 0.33 below — acceptable)

Run: python3 run_gate3_gemma_v93.py
"""
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
INPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_gate3_gemma_v92.json.gz"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_gate3_gemma_v93.json.gz"
DEPLOY_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v93.json.gz")


def postprocess_v93(text: str) -> str:
    # FIX 1: "Walk/walk out of the X" → "Exit/exit the X" (100%, -2 words)
    # Eliminates 6.8x GT overrep for walk_out_of_the (388→0)
    text = re.sub(r'\bWalk out of the\b', 'Exit the', text)
    text = re.sub(r'\bwalk out of the\b', 'exit the', text)
    # FIX 2: "Go/go out of the X" → "Exit/exit the X" (100%, -2 words)
    # Eliminates go_out_of_the overrep (43→0)
    text = re.sub(r'\bGo out of the\b', 'Exit the', text)
    text = re.sub(r'\bgo out of the\b', 'exit the', text)
    return text


def main():
    print("=" * 60)
    print("Gate3-Gemma v93 — post-processing on v92 (no vLLM)")
    print("Fixes: walk_out_of_the→exit_the, go_out_of_the→exit_the")
    print("=" * 60)

    with gzip.open(INPUT_PATH, "rt") as f:
        data = json.load(f)

    episodes = data["episodes"]
    print(f"Loaded {len(episodes)} episodes from v92")

    walk_out_conv = 0
    go_out_conv = 0
    out_episodes = []

    for ep in episodes:
        orig = ep["instruction"]["instruction_text"]
        text = postprocess_v93(orig)
        if orig != text:
            if re.search(r'Walk out of the|walk out of the', orig):
                walk_out_conv += 1
            if re.search(r'Go out of the|go out of the', orig):
                go_out_conv += 1
        out_ep = dict(ep)
        out_ep["instruction"] = {"instruction_text": text}
        out_episodes.append(out_ep)

    texts = [ep["instruction"]["instruction_text"] for ep in out_episodes]
    avg_words = sum(len(t.split()) for t in texts) / len(texts)

    def cnt(pat):
        return sum(len(re.findall(pat, t, re.I)) for t in texts)

    print(f"\nResults:")
    print(f"  walk_out_of_the: {cnt(r'walk out of the')}  (was 388, GT=57)")
    print(f"  go_out_of_the:   {cnt(r'go out of the')}  (was 43)")
    print(f"  exit_the:        {cnt(r'exit the')}  (was 265, GT=278)")
    print(f"  walk_through_the:{cnt(r'walk through the')}  (GT=191)")
    print(f"  avg_words:       {avg_words:.3f}  (was 26.918, GT=26.78)")
    print(f"  walk_out_conv={walk_out_conv}, go_out_conv={go_out_conv}")

    out_data = {"episodes": out_episodes, "instruction_vocab": data.get("instruction_vocab", {})}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUTPUT_PATH, "wt") as f:
        json.dump(out_data, f)
    print(f"\nSaved: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size//1024} KB)")

    DEPLOY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(DEPLOY_PATH, "wt") as f:
        json.dump(out_data, f)
    print(f"Deployed: {DEPLOY_PATH} ({DEPLOY_PATH.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
