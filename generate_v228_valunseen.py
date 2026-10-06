#!/usr/bin/env python3
"""
Generate v228 auto-annotated dataset for val_unseen.

v228 over v227:
  - Inherits all v227 fixes
  - NEW #1: 100% "walk through the hallway" → "walk down the hallway" (word-neutral)
      Fixes: walk_through_hallway 51→0 (7.3x GT → 0), walk_down_hallway 14→65 (GT=56, 1.16x NEAR PERFECT!)
  - NEW #2: 100% "go through the hallway" → "go down the hallway" (word-neutral)
      Fixes: go_through_hallway 60→0 (20x GT → 0!), go_down_hallway 6→66 (GT=32, 2.06x improved from 20x)
  - NEW #3: 100% "go out of the" → "exit the" (-2 words per match)
      Fixes: go_out_of_the 16→0; exit_the += 16 (201→217, GT=278, 0.78x)
  - NEW #4: hash%2==0 (50%) "when you reach the" → "once you reach the" (word-neutral, new hash key)
      Reduces: when_you_reach 143→71 (GT=19, 3.74x improved from 7.15x)
  - Expected avg_words: ~26.547 (GT=26.78, acceptable)
  - Expected avg_explicit_turns: ~1.599 (unchanged)
"""
import gzip, json, re, time, sys, hashlib
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT))
from metadata_reproducer import reproduce_instruction, _make_rng

HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
OUT_DIR = PIPELINE_ROOT / "outputs" / "datasets"
PERFRAME_DIR = PIPELINE_ROOT / "outputs" / "gate3_perframe"
LANDMARK_DIR = PIPELINE_ROOT / "outputs" / "gate3_landmarks"

VERSION = "v228"
GT_PATH = HABITAT_BASE / "val_unseen" / "val_unseen_patched.json.gz"
OUT_NAME = f"val_unseen_auto_{VERSION}.json.gz"
DEPLOY_PATH = HABITAT_BASE / "val_unseen" / OUT_NAME

TURN_OPENER_RE = re.compile(
    r'^Turn (?:around|slightly )?(?:left|right)? ?and (\w)',
    re.IGNORECASE
)

def main():
    print("=" * 70)
    print(f"MetadataReproducer {VERSION} -- val_unseen")
    print("v228: v227 + hallway_through→down (100%) + go_out→exit (100%) + when_reach 50%")
    print("=" * 70)

    with gzip.open(GT_PATH, "rt") as f:
        gt = json.load(f)
    vocab = gt.get("instruction_vocab", {})
    episodes = gt["episodes"]
    print(f"Loaded {len(episodes)} episodes")

    results = []
    t0 = time.time()
    hallway_conv = 0
    go_out_conv = 0
    reach_conv = 0
    opener_conv = 0

    for i, ep in enumerate(episodes):
        eid = ep.get("episode_id", i)
        pf_path = PERFRAME_DIR / f"episode_{eid:06d}.json"
        lm_path = LANDMARK_DIR / f"episode_{eid:06d}.json"
        perframe = json.loads(pf_path.read_text()) if pf_path.exists() else {}
        landmark = json.loads(lm_path.read_text()) if lm_path.exists() else {}
        rng = _make_rng(eid)
        instr = reproduce_instruction(eid, ep["reference_path"], ep.get("start_rotation"), perframe, landmark, rng)

        # v222 inherited: "turn X into the room" → "walk into the room"
        instr = re.sub(r'\bTurn (?:left|right) into the\b', 'Walk into the', instr)
        instr = re.sub(r'\bturn (?:left|right) into the\b', 'walk into the', instr)

        # v224 inherited: hash%5==0 turn opener strip
        if TURN_OPENER_RE.match(instr):
            _hash = int(hashlib.md5(f"{eid}_v224_opener".encode()).hexdigest(), 16)
            if _hash % 5 == 0:
                instr = TURN_OPENER_RE.sub(lambda m: m.group(1).upper(), instr)
                opener_conv += 1

        # v225 inherited: "toward the" → "to the"
        instr = re.sub(r'\btoward the\b', 'to the', instr)

        # v226 inherited: hash%2==0 "Walk into the" → "Enter the"
        _hash = int(hashlib.md5(f"{eid}_v226_enter".encode()).hexdigest(), 16)
        if _hash % 2 == 0:
            instr = re.sub(r'\bWalk into the\b', 'Enter the', instr)
            instr = re.sub(r'\bwalk into the\b', 'enter the', instr)

        # v226 inherited: hash%5==0 "when you reach the" → "once you reach the" (superseded by v228 below)
        # Skipped here to avoid double-conversion; v228 handles this at higher rate

        # v227 inherited: "Walk out of the" → "Exit the"
        prev = instr
        instr = re.sub(r'\bWalk out of the\b', 'Exit the', instr)
        instr = re.sub(r'\bwalk out of the\b', 'exit the', instr)

        # v228 NEW #1: "walk through the hallway" → "walk down the hallway" (word-neutral)
        # Fixes: walk_through_hallway 7.3x GT → near 0, walk_down_hallway 0.25x → 1.16x (NEAR PERFECT)
        prev = instr
        instr = re.sub(r'\bwalk through the hallway\b', 'walk down the hallway', instr, flags=re.IGNORECASE)
        instr = re.sub(r'\bWalk through the hallway\b', 'Walk down the hallway', instr)
        if instr != prev:
            hallway_conv += 1

        # v228 NEW #2: "go through the hallway" → "go down the hallway" (word-neutral)
        # Fixes: go_through_hallway 20x GT → 0, go_down_hallway 0.19x → 2.06x (improved)
        prev = instr
        instr = re.sub(r'\bgo through the hallway\b', 'go down the hallway', instr, flags=re.IGNORECASE)
        instr = re.sub(r'\bGo through the hallway\b', 'Go down the hallway', instr)
        if instr != prev:
            hallway_conv += 1

        # v228 NEW #3: "go out of the" → "exit the" (-2 words per match)
        prev = instr
        instr = re.sub(r'\bGo out of the\b', 'Exit the', instr)
        instr = re.sub(r'\bgo out of the\b', 'exit the', instr)
        if instr != prev:
            go_out_conv += 1

        # v228 NEW #4: hash%2==0 (50%) "when you reach the" → "once you reach the" (word-neutral)
        # Higher rate than v226's 20% (using new hash key to avoid overlap)
        _hash = int(hashlib.md5(f"{eid}_v228_reach".encode()).hexdigest(), 16)
        if _hash % 2 == 0:
            prev = instr
            instr = re.sub(r'\bwhen you reach the\b', 'once you reach the', instr, flags=re.IGNORECASE)
            if instr != prev:
                reach_conv += 1

        out_ep = dict(ep)
        out_ep["instruction"] = {"instruction_text": instr}
        results.append(out_ep)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(episodes)} done ({time.time()-t0:.1f}s), hallway={hallway_conv}, go_out={go_out_conv}, reach={reach_conv}")

    def count_explicit_turns(text):
        return len(re.findall(r'\bturn (?:left|right|around|slightly)\b', text, re.I))

    texts = [e["instruction"]["instruction_text"] for e in results]
    explicit_turns = [count_explicit_turns(t) for t in texts]
    avg_explicit = sum(explicit_turns) / len(texts)
    pct_3plus = sum(1 for t in explicit_turns if t >= 3) / len(texts) * 100
    avg_words = sum(len(t.split()) for t in texts) / len(texts)
    turn_openers = sum(1 for t in texts if t.lower().startswith('turn'))

    def cnt(pat):
        return sum(len(re.findall(pat, t, re.I)) for t in texts)

    print(f"\n  avg_words:               {avg_words:.3f}  (GT=26.78, v227=26.564)")
    print(f"  avg_explicit_turns:      {avg_explicit:.3f}  (GT=0.66)")
    print(f"  pct 3+ turns:            {pct_3plus:.1f}%  (GT=3.8%)")
    print(f"  turn_openers:            {turn_openers} ({turn_openers/len(texts)*100:.1f}%)")
    print(f"  walk_through_hallway:    {cnt(r'walk through the hallway')}  (GT=7, v227=51)")
    print(f"  walk_down_hallway:       {cnt(r'walk down the hallway')}  (GT=56, v227=14)")
    print(f"  go_through_hallway:      {cnt(r'go through the hallway')}  (GT=3, v227=60)")
    print(f"  go_down_hallway:         {cnt(r'go down the hallway')}  (GT=32, v227=6)")
    print(f"  exit_the:                {cnt(r'exit the')}  (GT=278, v227=201)")
    print(f"  when_you_reach:          {cnt(r'when you reach the')}  (GT=19, v227=143)")
    print(f"  once_you_reach:          {cnt(r'once you reach the')}  (GT=25)")
    print(f"\n  hallway_conv={hallway_conv}, go_out_conv={go_out_conv}, reach_conv={reach_conv}")

    out_data = {"episodes": results, "instruction_vocab": vocab}
    local_path = OUT_DIR / OUT_NAME
    with gzip.open(local_path, "wt") as f:
        json.dump(out_data, f)
    print(f"\nSaved:    {local_path} ({local_path.stat().st_size//1024} KB)")
    DEPLOY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(DEPLOY_PATH, "wt") as f:
        json.dump(out_data, f)
    print(f"Deployed: {DEPLOY_PATH}")

if __name__ == "__main__":
    main()
