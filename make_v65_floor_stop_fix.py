#!/usr/bin/env python3
"""
v65: Apply floor-stop fix to v64 dataset (no LLM re-generation needed).

PROBLEM (v64): 131/1484 stop-episodes (8.8%) have floor-based stops:
  "Stop in front of the white hallway floor" — not navigable!
  "Wait near the light-colored floor" — useless landmark

ROOT CAUSE: VLM Phase 1 goal descriptions sometimes see only floor/tile/carpet
  when the stop location is a featureless area. The stop phrase generation
  then picks this as the goal landmark.

FIX (v65): Detect and replace floor/tile/carpet stop phrases with simpler alternatives:
  - If hallway context: "Stop in the hallway"
  - If room context: "Stop in the kitchen" / "Stop in the bedroom" / etc.
  - Otherwise: "Stop" (minimal, always safe)

IMPORTANT: This fix is applied as post-processing to the v64 assembled dataset.
  No Phase 2 LLM calls needed. Reuses v64 token assignments.
  Only the instruction_text changes (instruction_tokens are also updated).
"""
import gzip
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
V64_PATH = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v64.json.gz"
V65_PATH = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v65.json.gz"

# Add tokenizer to path
sys.path.insert(0, str(ROOT))
from gate5_tokenizer.tokenizer import VLNTokenizer

# Floor/surface words that make bad stop landmarks
_FLOOR_WORDS = re.compile(
    r'\b(floor(?:ing)?|tile(?:s)?|carpet(?:ing)?|linoleum|mat|hardwood\s+floor|'
    r'tiled\s+floor|marble\s+floor|wood\s+floor|stone\s+floor|'
    r'polished\s+floor|light[\-\s]colored\s+floor|white\s+floor|'
    r'grey\s+floor|gray\s+floor)\b',
    re.IGNORECASE
)

# Pattern: Stop/Wait [anywhere] [floor word] [anything]
_FLOOR_STOP_RE = re.compile(
    r'\b(Stop|Wait)\b(?:.{0,80}?)' + _FLOOR_WORDS.pattern + r'.*$',
    re.IGNORECASE | re.DOTALL
)

# Also fix: "Stop near the rug" / "Stop at the mat" (floor coverings as stop target)
_RUG_STOP_RE = re.compile(
    r'\b(Stop|Wait)\s+(?:near|at|by|in\s+front\s+of|beside|next\s+to)\s+the\s+'
    r'(?:\w+\s+)*(?:rug|mat|carpet|runner|doormat)\b.*$',
    re.IGNORECASE | re.DOTALL
)


def fix_floor_stop(txt: str) -> tuple:
    """Fix floor-based stop phrase. Returns (fixed_txt, was_fixed)."""
    for pattern in (_FLOOR_STOP_RE, _RUG_STOP_RE):
        m = pattern.search(txt)
        if not m:
            continue

        action = m.group(1)  # "Stop" or "Wait"
        prefix = txt[:m.start()].rstrip()

        # Determine better replacement based on room context in full instruction
        full_lower = txt.lower()
        if re.search(r'\bhallway\b', full_lower):
            replacement = f"{action} in the hallway."
        elif re.search(r'\bkitchen\b', full_lower):
            replacement = f"{action} in the kitchen."
        elif re.search(r'\bbedroom\b', full_lower):
            replacement = f"{action} in the bedroom."
        elif re.search(r'\bliving room\b', full_lower):
            replacement = f"{action} in the living room."
        elif re.search(r'\bdining room\b', full_lower):
            replacement = f"{action} in the dining room."
        elif re.search(r'\bbathroom\b', full_lower):
            replacement = f"{action} in the bathroom."
        elif re.search(r'\boffice\b', full_lower):
            replacement = f"{action} in the office."
        elif re.search(r'\bstaircase|stairs|stairway\b', full_lower):
            replacement = f"{action} at the stairs."
        elif re.search(r'\bdoorway|doorframe\b', full_lower):
            replacement = f"{action} in the doorway."
        else:
            replacement = f"{action}."

        return prefix + " " + replacement, True

    return txt, False


def compute_stats(texts):
    import statistics
    n = len(texts)
    words = [len(t.split()) for t in texts]
    stop_n = sum(1 for t in texts if re.search(r'\bstop\b', t, re.I))
    wait_n = sum(1 for t in texts if re.search(r'\bwait\b', t, re.I))
    anchor_n = sum(1 for t in texts if re.search(r'turn\s+(left|right)\s+(at|past|through|into|around)\s+the\s+\w', t, re.I))
    hallway_n = sum(1 for t in texts if re.search(r'\bhallway\b', t, re.I))
    thru_n = sum(1 for t in texts if re.search(r'\bthrough the\b', t, re.I))
    wp_n = sum(1 for t in texts if re.search(r'\bwalk past the\b', t, re.I))
    floor_stop_n = sum(1 for t in texts if _FLOOR_STOP_RE.search(t) or _RUG_STOP_RE.search(t))

    print(f"  avg_words:     {sum(words)/n:.1f}  (GT=26.8)")
    print(f"  stop%:         {100*stop_n/n:.1f}%  (GT=50.8%)")
    print(f"  wait%:         {100*wait_n/n:.1f}%  (GT=30.5%)")
    print(f"  turn-anchor%:  {100*anchor_n/n:.1f}%  (GT=16.5%)")
    print(f"  hallway%:      {100*hallway_n/n:.1f}%  (GT=20.4%)")
    print(f"  through the%:  {100*thru_n/n:.1f}%  (GT=27.2%)")
    print(f"  walk past the%: {100*wp_n/n:.1f}%  (GT=8.7%)")
    print(f"  floor-stop%:   {100*floor_stop_n/n:.1f}%  (target: 0%)")
    # GT-match
    anchor_err = abs(anchor_n/n - 0.165)
    stop_err = abs(stop_n/n - 0.508)
    wait_err = abs(wait_n/n - 0.305)
    gt_match = 1.0 - (anchor_err * 2 + stop_err + wait_err)
    print(f"  GT-match:      {gt_match:.3f}  (target ≥ 0.90)")


def main():
    print(f"Loading v64 dataset...")
    with gzip.open(V64_PATH) as f:
        data = json.load(f)

    # VLNTokenizer loads from a .json.gz path — use the GT dataset path
    from gate5_tokenizer.tokenizer import VOCAB_SOURCE
    tokenizer = VLNTokenizer(VOCAB_SOURCE)
    episodes = data["episodes"]
    n_fixed = 0
    n_total_stop = 0

    texts_before = [ep["instruction"]["instruction_text"] for ep in episodes]

    for ep in episodes:
        txt = ep["instruction"]["instruction_text"]
        if re.search(r'\b(stop|wait)\b', txt, re.I):
            n_total_stop += 1

        new_txt, was_fixed = fix_floor_stop(txt)
        if was_fixed:
            n_fixed += 1
            ep["instruction"]["instruction_text"] = new_txt
            ep["instruction"]["instruction_tokens"] = tokenizer.encode(new_txt)

    texts_after = [ep["instruction"]["instruction_text"] for ep in episodes]

    print(f"\n=== v65 Floor-Stop Fix ===")
    print(f"  Episodes processed: {len(episodes)}")
    print(f"  Episodes with stop/wait: {n_total_stop}")
    print(f"  Floor-stop episodes fixed: {n_fixed} ({100*n_fixed/max(n_total_stop,1):.1f}% of stops)")
    print(f"\nBEFORE (v64):")
    compute_stats(texts_before)
    print(f"\nAFTER (v65):")
    compute_stats(texts_after)

    with gzip.open(V65_PATH, "wt") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"\nSaved: {V65_PATH}")


if __name__ == "__main__":
    main()
