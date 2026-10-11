#!/usr/bin/env python3
"""
Deploy our Gemma reproduce of Kemal's top5 (run_annotator_configured.py output) as InternNav labels.

  in : CACHE_ROOT/top5_reproduce/{train,seen,unseen}_v{N}.json.gz   (runner output; instruction_text
       there is source_dataset text = GT/ChronoNav, generated text only in _generated_instruction)
  out: HABITAT_BASE/{split}/{split}_v{N}_reproduce.json.gz   instruction_text <- _generated_instruction.text
       for all 3 splits, normalized + val_unseen_patched vocab, tokens [] (same as convert_top5_generated.py)

  --fix: run_annotator_configured.py --fix output (key frames + hfov 79 + black-frame check) ({short}_v{N}_fix.json.gz)
         -> {split}_v{N}_reproduce_fix.json.gz

Usage: python3 deploy_top5_reproduce.py [--fix] [v264 v272 ...]
"""
import argparse
import gzip
import json

from convert_top5_generated import SPLITS, VERSIONS, normalize
from local_paths import CACHE_ROOT, HABITAT_BASE, VOCAB_SOURCE

SRC = CACHE_ROOT / "top5_reproduce"
GT_EPISODE_KEYS = ("episode_id", "trajectory_id", "scene_id", "start_position", "start_rotation",
                   "goals", "reference_path", "info", "instruction")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("versions", nargs="*", default=VERSIONS)
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()
    sfx = "_fix" if a.fix else ""
    vocab = json.load(gzip.open(VOCAB_SOURCE))["instruction_vocab"]
    for v in a.versions:
        for short, split in SPLITS.items():
            src = SRC / f"{short}_{v}{sfx}.json.gz"
            dest = HABITAT_BASE / split / f"{split}_{v}_reproduce{sfx}.json.gz"
            if dest.exists():  # never overwrite anything in the label tree
                print(f"  skip (exists) {dest}")
                continue
            if not src.exists():
                print(f"  missing {src}")
                continue
            d = json.load(gzip.open(src))
            for e in d["episodes"]:
                e["instruction"]["instruction_text"] = normalize(e["_generated_instruction"]["text"])
                e["instruction"]["instruction_tokens"] = []
            # habitat VLNEpisode(**episode) rejects any non-GT key (_generated_instruction etc.)
            d["episodes"] = [{k: e[k] for k in GT_EPISODE_KEYS if k in e} for e in d["episodes"]]
            d["instruction_vocab"] = vocab
            with gzip.open(dest, "wt") as f:
                json.dump(d, f)
            print(f"  {dest}  {len(d['episodes'])} eps")


if __name__ == "__main__":
    main()
