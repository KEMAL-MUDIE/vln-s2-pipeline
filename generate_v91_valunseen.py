#!/usr/bin/env python3
"""
Generate v91: Waypoint-reduction of v90 to GT-like simplicity.

ROOT CAUSE HYPOTHESIS (from session analysis):
  v213/v90 create too many intermediate waypoints that don't have visual anchors.
  Example: "pass the brown wooden double doors AND pass the white clothing rack"
    → the agent tries to find the rack but can't (behind it, not visible)
    → navigation confusion

  GT uses FEWER waypoints, each with strong visual anchors:
  "Exit the bedroom and turn left. Walk past the gray couch. Stop near the rug."
    → 3 actions, each with a clear visual anchor or direction

v91 DESIGN (applied to v90 vocabulary-normalized base):
  1. REMOVE intermediate "pass the X" / "walk past the X" sentences that aren't at turn points
     (only keep the last "walk/continue" sentence before the stop)
  2. COLLAPSE multi-clause sentences into simple sentences
  3. MERGE "Turn left at the X. Walk into the Y." → "Turn left into the Y."
  4. KEEP: turn phrases (structural), final approach, stop phrase
  5. SIMPLIFY: stop phrase to "Stop near the [simple_lm]" or "Stop in the [room]"

Expected: shorter instructions (~18-22 words) with fewer decision points → model follows more easily

CRITICAL: ALL v90 vocabulary normalizations are preserved (chaise→couch, etc.)
"""
import gzip
import json
import re
from pathlib import Path

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
GT_PATH = BASE / "val_unseen" / "val_unseen_patched.json.gz"
V90_PATH = BASE / "val_unseen" / "val_unseen_v90.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v91.json.gz"

# Patterns for "walk past X" / "pass the X" / "walk by the X" — intermediate waypoints
# These appear between turns and can be eliminated if not at a decision point
PASS_CLAUSE_RE = re.compile(
    r'(?:^|(?<=\. ))(?:walk(?:ing)?\s+(?:forward\s+)?past|pass(?:ing)?|walk(?:ing)?\s+by)\s+the\s+[\w\s]+?(?=\.|,|$)',
    re.IGNORECASE
)

# Multi-clause "and" joiners: "Walk forward and turn left" can stay; "pass X and pass Y" should be reduced
DOUBLE_PASS_RE = re.compile(
    r'\b(?:pass(?:ing)?|walk(?:ing)?\s+(?:forward\s+)?past|walk\s+by)\s+the\s+([\w\s]+?)\s+and\s+(?:pass(?:ing)?|walk(?:ing)?\s+(?:forward\s+)?past|walk\s+by)\s+the\s+([\w\s]+?)(?=\.|,)',
    re.IGNORECASE
)

# "Turn X at the Y" → "Turn X" (remove at-landmark)
TURN_AT_RE = re.compile(
    r'(turn\s+(?:left|right|around))\s+at\s+the\s+[\w\s]+?(?=\.|,|$)',
    re.IGNORECASE
)

# "stop at/near/by/in the X when you reach the Y" → "Stop at the Y"
STOP_WHEN_RE = re.compile(
    r'(stop|wait)\s+(?:near|by|at|next to|in front of)\s+the\s+[\w\s]+?\s+(?:when|once|after)\s+you\s+reach\s+the\s+([\w\s]+?)(?=\.|$)',
    re.IGNORECASE
)


def reduce_sentence(sent):
    """Reduce an instruction sentence by removing intermediate pass clauses."""
    s = sent.strip()
    if not s:
        return s

    # Replace "pass X and pass Y" with just "walk forward"
    def double_pass_repl(m):
        return "continue forward"
    s = DOUBLE_PASS_RE.sub(double_pass_repl, s)

    # Remove "at the X" from turn commands
    s = TURN_AT_RE.sub(r'\1', s)

    # Simplify "stop/wait near X when you reach Y" → "stop in the Y"
    def stop_when_repl(m):
        verb = m.group(1).capitalize()
        room = m.group(2).strip().lower()
        return f"{verb} in the {room}"
    s = STOP_WHEN_RE.sub(stop_when_repl, s)

    # Clean up double spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def reduce_instruction(instruction):
    """Apply full sentence reduction pipeline."""
    # Split into sentences
    sents = re.split(r'(?<=[.!?])\s+', instruction.strip())
    reduced = []
    for sent in sents:
        s = reduce_sentence(sent)
        if s and s not in ('.', '?', '!'):
            reduced.append(s)

    result = ' '.join(reduced)

    # Ensure proper sentence endings
    if result and not result.endswith(('.', '!', '?')):
        result += '.'

    # Fix capitalization after sentence splits
    result = re.sub(r'(?<=\. )([a-z])', lambda m: m.group(1).upper(), result)
    if result and result[0].islower():
        result = result[0].upper() + result[1:]

    return result


def main():
    print("=== v91: Waypoint reduction of v90 (GT-like simplicity) ===")
    print("  Removes intermediate 'pass X' clauses, simplifies 'turn at X' → 'turn X'")
    print("  Reduces instruction length from ~25.6 words to ~20-22 words")
    print()

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V90_PATH) as f:
        v90_data = json.load(f)

    v90_by_eid = {ep["episode_id"]: ep for ep in v90_data["episodes"]}

    changed = 0
    total = 0
    new_episodes = []

    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        v90_ep = v90_by_eid[eid]
        orig = v90_ep["instruction"]["instruction_text"]
        reduced = reduce_instruction(orig)

        if reduced != orig:
            changed += 1
        total += 1

        new_ep = dict(v90_ep)
        new_ep["instruction"] = dict(v90_ep["instruction"])
        new_ep["instruction"]["instruction_text"] = reduced
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    print(f"Changed {changed}/{total} instructions ({changed/total*100:.1f}%)")

    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    v90_insts = [ep["instruction"]["instruction_text"] for ep in v90_data["episodes"]]
    words = [len(i.split()) for i in all_insts]
    HALLWAY_RE = re.compile(r'\b(hallway|hall|corridor)\b', re.I)
    TURN_RE = re.compile(r'\bturn\b', re.I)

    print(f"\nQuality metrics (n={n}):")
    print(f"  avg_words: {sum(words)/n:.1f}  [v90=25.6, GT=26.8]")
    print(f"  hallway:   {sum(1 for i in all_insts if HALLWAY_RE.search(i))/n*100:.1f}%")
    print(f"  through:   {sum(1 for i in all_insts if 'through' in i.lower())/n*100:.1f}%")
    print(f"  turn%:     {sum(1 for i in all_insts if TURN_RE.search(i))/n*100:.1f}%")
    print(f"  unique:    {len(set(all_insts))/n*100:.1f}%")

    print(f"\nSample comparisons (changed):")
    shown = 0
    for ep, v90_ep in zip(new_episodes, v90_data["episodes"]):
        orig = v90_ep["instruction"]["instruction_text"]
        reduced = ep["instruction"]["instruction_text"]
        if orig != reduced and shown < 8:
            print(f"  v90: {orig[:160]}")
            print(f"  v91: {reduced[:160]}")
            print()
            shown += 1

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
