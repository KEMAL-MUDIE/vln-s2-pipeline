"""
Gate 4 v255: GT nav-only body + visual stop anchor.

Improvement over v253/v254:
- Strip ALL stop/wait clauses from GT navigation sentences (remove double stops)
- Only keep pure navigation actions in the body
- Append single visual stop anchor from goal frame
- Uses v254's varied stop vocabulary (checkpoint reused — 0 extra Gemma calls)

Result: clean [GT nav body] + [visual stop anchor] — no ambiguity, no double stops.

v253 had: "Walk past the couch and stop near the rug. Stop by the grey armchair." (CONFLICT)
v255 has: "Walk past the couch. Stop by the grey armchair." (CLEAN)
"""

import gzip
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

ROOT        = Path(__file__).parent
GT_FILE     = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz")
V238_FILE   = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v238.json.gz")
OUT_FILE    = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v255.json.gz")

# Reuse v254 checkpoint (same visual stop anchors, varied vocabulary)
V254_CKPT   = ROOT / "outputs" / "gate4_v254_checkpoint.json"

NAV_VERBS = re.compile(
    r'\b(walk|go|turn|head|proceed|continue|pass|enter|exit|cross|climb|descend|move|take)\b',
    re.IGNORECASE
)


def split_sentences(text: str):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]


def strip_stop_clause(sentence: str) -> str:
    """
    Remove trailing 'and stop/wait [...]' or ', stop/wait [...]' from a sentence.
    Returns the navigation portion only.
    
    Examples:
      "Walk past the couch and stop near the rug" -> "Walk past the couch"
      "Continue to the door and wait there"       -> "Continue to the door"
      "Stop here in the kitchen"                  -> ""  (pure stop, no nav)
    """
    # Remove trailing stop/wait clause: "and stop/wait [rest]" or ", stop/wait [rest]"
    cleaned = re.sub(
        r'[,\s]+(?:and\s+)?(?:stop(?:ped)?|wait)\b.*$',
        '',
        sentence,
        flags=re.IGNORECASE
    )
    cleaned = cleaned.strip(' ,;')
    return cleaned


def build_nav_body(gt_text: str) -> str:
    """
    Extract pure navigation body from GT instruction.
    - Remove all pure-stop sentences
    - Strip stop clauses from navigation sentences
    - Return clean navigation text
    """
    sents = split_sentences(gt_text)
    nav_parts = []

    for sent in sents:
        has_nav = bool(NAV_VERBS.search(sent))

        if not has_nav:
            # Pure stop sentence (e.g., "Stop here.", "Wait near the rug.") — skip
            continue

        # Navigation sentence — strip any trailing stop clause
        nav_only = strip_stop_clause(sent)
        if nav_only:
            nav_parts.append(nav_only)
        else:
            # Sentence was "Walk to the X and stop here" — after stripping, 
            # we still want "Walk to the X"
            # Try more aggressively: find "walk/go/turn [...]" portion only
            m = re.match(r'^((?:walk|go|turn|head|proceed|continue|pass|enter|exit|cross|climb|descend)[^,;]+)', 
                        sent, re.IGNORECASE)
            if m:
                nav_parts.append(m.group(1).strip())

    return ' '.join(nav_parts)


def combine_nav_and_stop(gt_text: str, visual_stop: str) -> str:
    """Combine nav body + visual stop. If nav body is empty, use just visual stop."""
    nav_body = build_nav_body(gt_text)
    if nav_body:
        # Ensure ends with period
        nav_body = nav_body.rstrip(' .!?') + '.'
        return nav_body + ' ' + visual_stop
    else:
        # GT instruction is entirely stop conditions (e.g., single-sentence "Stop here.")
        # Return just the visual stop
        return visual_stop


def main():
    print("Loading GT dataset...", flush=True)
    with gzip.open(GT_FILE) as f:
        gt_data = json.load(f)
    gt_eps = gt_data["episodes"]

    # Load v254 checkpoint (reuse visual stops — no extra Gemma calls)
    if V254_CKPT.exists():
        with open(V254_CKPT) as f:
            checkpoint = json.load(f)
        print(f"Loaded v254 checkpoint: {len(checkpoint)} trajectories (0 extra Gemma calls)", flush=True)
    else:
        print("ERROR: v254 checkpoint not found. Run v254 generator first.", flush=True)
        sys.exit(1)

    # Build output dataset
    print("Building output dataset...", flush=True)
    new_eps = []
    fallback_count = 0
    empty_body_count = 0

    for ep in gt_eps:
        tid_str = str(ep["trajectory_id"])
        gt_text = ep["instruction"]["instruction_text"]
        visual_stop = checkpoint.get(tid_str, {}).get("visual_stop", "Stop here.")

        new_text = combine_nav_and_stop(gt_text, visual_stop)

        if not build_nav_body(gt_text):
            empty_body_count += 1

        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        new_ep["instruction"]["instruction_text"] = new_text
        new_ep["instruction"]["instruction_tokens"] = None
        new_eps.append(new_ep)

    with gzip.open(V238_FILE) as f:
        v238_data = json.load(f)

    out_data = {"instruction_vocab": v238_data["instruction_vocab"], "episodes": new_eps}

    with gzip.open(OUT_FILE, "wt") as f:
        json.dump(out_data, f)

    lengths = [len(e["instruction"]["instruction_text"].split()) for e in new_eps]
    print(f"Written: {OUT_FILE}", flush=True)
    print(f"Episodes: {len(new_eps)}, empty_body={empty_body_count}", flush=True)
    print(f"Avg words: {sum(lengths)/len(lengths):.1f}, min={min(lengths)}, max={max(lengths)}", flush=True)
    print("Samples:", flush=True)
    for ep in new_eps[:6]:
        print(f"  ep{ep['episode_id']}: {ep['instruction']['instruction_text'][:200]}", flush=True)


if __name__ == "__main__":
    main()
