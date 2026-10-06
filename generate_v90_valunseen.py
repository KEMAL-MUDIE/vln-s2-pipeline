#!/usr/bin/env python3
"""
Generate v90: Vocabulary normalization of v213 to match GT training distribution.

KEY INSIGHT: GT achieves 63.57% using these vocab patterns:
  - Room: bedroom, kitchen, hallway, living room, bathroom (simple room names)
  - Furniture stops: couch, sofa, chair, table, bed, rug, mat (generic names)
  - Landmarks: door, doorway, stairs, railing, window (architectural)
  - Actions: walk, go, exit, enter, pass, turn (simple verbs)

v213 achieves 36.87% using:
  - Furniture: chaise lounge, armchair, dining table, coffee table (over-specific)
  - Objects: wooden double doors, glass partition, floor mosaic (VLM-specific)
  - Actions: walk forward past, continue through, stop near (slightly off)

v90 DESIGN:
  Load v213 → apply vocabulary normalization:
    FURNITURE: chaise lounge→couch, armchair→chair, dining table→table, coffee table→table,
               side table→table, end table→table, kitchen table→table,
               lounge chair→chair, accent chair→chair, ottoman→footstool,
               massage table→table, billiard table→pool table,
               sofa→couch (normalize to single term)
    DOORS: double doors→doorway, sliding door→door, glass door→door,
           french doors→doorway, screen door→door, door frame→doorway,
           wooden door→door, metal door→door
    MISC: television→TV, shelving unit→shelves, clothing rack→closet area,
          floor mosaic→floor, tile floor→floor, carpet→rug (simpler),
          wall art→art, potted plant→plant, light fixture→light
    VERBS: walk forward past→walk past, go forward past→walk past,
           stop near the→stop by the (minor normalization)

Expected: SR improvement 2-8pp beyond v213 (from 36.87% toward 40-45%)
          by reducing vocabulary mismatch between v213 and GT training data
"""
import gzip
import json
import re
from pathlib import Path

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
GT_PATH = BASE / "val_unseen" / "val_unseen_patched.json.gz"
V213_PATH = BASE / "val_unseen" / "val_unseen_auto_v213.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v90.json.gz"


# (pattern, replacement) — applied in order, case-insensitive
NORMALIZATIONS = [
    # Furniture: specific → generic
    (r'\bchaise lounge\b', 'couch'),
    (r'\bchaisse lounge\b', 'couch'),
    (r'\barmchair\b', 'chair'),
    (r'\baccent chair\b', 'chair'),
    (r'\blounge chair\b', 'chair'),
    (r'\boccasional chair\b', 'chair'),
    (r'\bdining table\b', 'table'),
    (r'\bcoffee table\b', 'table'),
    (r'\bside table\b', 'table'),
    (r'\bend table\b', 'table'),
    (r'\bkitchen table\b', 'table'),
    (r'\bkitchen island\b', 'kitchen counter'),
    (r'\bmassage table\b', 'table'),
    (r'\bpool table\b', 'table'),
    (r'\bbilliard table\b', 'table'),
    (r'\bnight(stand|table)\b', 'nightstand'),
    (r'\bdresser\b', 'dresser'),
    (r'\bsofa\b', 'couch'),
    (r'\bsectional\b', 'couch'),
    (r'\blove seat\b', 'couch'),
    (r'\bloveseat\b', 'couch'),
    (r'\bfuton\b', 'couch'),
    (r'\bdesk chair\b', 'chair'),
    (r'\bbar stool\b', 'stool'),
    (r'\bbench\b', 'bench'),
    # Doors: specific → generic
    (r'\bdouble (?:wooden |glass |metal )?doors?\b', 'doorway'),
    (r'\b(?:wooden |glass |metal )?double doors?\b', 'doorway'),
    (r'\bsliding glass door\b', 'door'),
    (r'\bsliding door\b', 'door'),
    (r'\bfrench doors?\b', 'doorway'),
    (r'\bscreen door\b', 'door'),
    (r'\bglass door\b', 'door'),
    (r'\bwooden door\b', 'door'),
    (r'\bmetal door\b', 'door'),
    (r'\bdoor frame\b', 'doorway'),
    (r'\bdoor jamb\b', 'doorway'),
    (r'\barched doorway\b', 'doorway'),
    (r'\barch(ed)? entry\b', 'entry'),
    (r'\barch(ed)? entrance\b', 'entrance'),
    # Objects: over-specific → generic
    (r'\btelevision set\b', 'TV'),
    (r'\btelevision\b', 'TV'),
    (r'\bTV set\b', 'TV'),
    (r'\bshelving unit\b', 'shelves'),
    (r'\bbookshelf\b', 'bookcase'),
    (r'\bclothing rack\b', 'rack'),
    (r'\bfloor mosaic\b', 'floor'),
    (r'\btile floor\b', 'floor'),
    (r'\btiled floor\b', 'floor'),
    (r'\bhardwood floor\b', 'floor'),
    (r'\bpotted plant\b', 'plant'),
    (r'\bwall art\b', 'artwork'),
    (r'\blight fixture\b', 'light'),
    (r'\blamp\b', 'lamp'),
    (r'\bchandelier\b', 'light fixture'),
    (r'\bpendant light\b', 'light'),
    (r'\bgranite counter\b', 'counter'),
    (r'\bmarble counter\b', 'counter'),
    (r'\bkitchen counter\b', 'counter'),
    # Action verbs: normalize to GT patterns
    (r'\bwalk forward past\b', 'walk past'),
    (r'\bgo forward past\b', 'walk past'),
    (r'\bcontinue forward past\b', 'walk past'),
    (r'\bwalk past\b', 'walk past'),  # normalize case
    # Stop phrase: normalize to most common GT forms
    (r'\bwait next to\b', 'wait near'),
    (r'\bstop next to\b', 'stop by'),
    (r'\bstop in front of the\b', 'stop at the'),
    # Hallway: normalize walk_through_hallway to walk_down_hallway (GT pattern)
    (r'\bwalk through the hallway\b', 'walk down the hallway'),
    (r'\bwalk through the corridor\b', 'walk down the hallway'),
    (r'\bgo through the hallway\b', 'go down the hallway'),
    (r'\bcontinue through the hallway\b', 'continue down the hallway'),
    # Room transition clean up
    (r'\bwhen you reach\b', 'once you reach'),
    (r'\bgo out of the\b', 'exit the'),
]

# Precompile patterns
COMPILED = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in NORMALIZATIONS]


def normalize(instruction):
    s = instruction
    for pat, repl in COMPILED:
        s = pat.sub(repl, s)
    # Clean up extra spaces
    s = re.sub(r'  +', ' ', s).strip()
    # Fix sentence capitalization (ensure first letter after '. ' is uppercase)
    s = re.sub(r'(?<=\. )([a-z])', lambda m: m.group(1).upper(), s)
    # Capitalize first character of the whole instruction
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s


def main():
    print("=== v90: Vocabulary normalization of v213 to GT training distribution ===")
    print("  Simplifies: chaise lounge→couch, armchair→chair, dining table→table, etc.")
    print("  Normalizes: door phrases, stop verbs, hallway through→down")
    print()

    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V213_PATH) as f:
        v213_data = json.load(f)

    v213_by_eid = {ep["episode_id"]: ep for ep in v213_data["episodes"]}

    changed = 0
    total = 0
    new_episodes = []

    for ep in gt_data["episodes"]:
        eid = ep["episode_id"]
        v213_ep = v213_by_eid[eid]
        orig = v213_ep["instruction"]["instruction_text"]
        norm = normalize(orig)

        if norm != orig:
            changed += 1
        total += 1

        new_ep = dict(v213_ep)
        new_ep["instruction"] = dict(v213_ep["instruction"])
        new_ep["instruction"]["instruction_text"] = norm
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

    print(f"Changed {changed}/{total} instructions ({changed/total*100:.1f}%)")

    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    v213_insts = [ep["instruction"]["instruction_text"] for ep in v213_data["episodes"]]
    words = [len(i.split()) for i in all_insts]
    HALLWAY_RE = re.compile(r'\b(hallway|hall|corridor)\b', re.I)
    TURN_RE = re.compile(r'\bturn\b', re.I)

    print(f"\nQuality metrics (n={n}):")
    print(f"  avg_words: {sum(words)/n:.1f}  [v213=26.8, GT=26.8]")
    print(f"  hallway:   {sum(1 for i in all_insts if HALLWAY_RE.search(i))/n*100:.1f}%")
    print(f"  through:   {sum(1 for i in all_insts if 'through' in i.lower())/n*100:.1f}%")
    print(f"  turn%:     {sum(1 for i in all_insts if TURN_RE.search(i))/n*100:.1f}%")
    print(f"  unique:    {len(set(all_insts))/n*100:.1f}%")
    print(f"  chaise_lounge: {sum(1 for i in all_insts if 'chaise' in i.lower())}")
    print(f"  couch:      {sum(1 for i in all_insts if 'couch' in i.lower())}")
    print(f"  armchair:  {sum(1 for i in all_insts if 'armchair' in i.lower())}")
    print(f"  chair:     {sum(1 for i in all_insts if ' chair' in i.lower())}")
    print(f"  doorway:   {sum(1 for i in all_insts if 'doorway' in i.lower())}")
    print(f"  door\":     {sum(1 for i in all_insts if ' door' in i.lower())}")

    print(f"\nSample comparisons (changed):")
    shown = 0
    for ep, v213_ep in zip(new_episodes, v213_data["episodes"]):
        orig = v213_ep["instruction"]["instruction_text"]
        norm = ep["instruction"]["instruction_text"]
        if orig != norm and shown < 8:
            print(f"  v213: {orig[:150]}")
            print(f"  v90:  {norm[:150]}")
            print()
            shown += 1

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
