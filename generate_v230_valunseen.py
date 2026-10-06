#!/usr/bin/env python3
"""
Generate v230 auto-annotated dataset for val_unseen.

v230 over v229:
  - Inherits all v229 fixes
  - NEW: 100% "walk straight past" → "walk past" (-1 word per match)
      CRITICAL: walk_straight_past 143→0 (9.5x GT → 0!), walk_past 656→799 (GT=216, 3.7x)
      Rationale: 9.5x overrep for "walk straight past" is worse than resulting 3.7x for "walk past".
      "walk past" at 3.7x aligns better with GT's dominant form for traversal-past-landmark.
  - Word impact: "walk straight past" → "walk past" costs -1 word × 143 = -0.078 avg_words
  - Expected avg_words: ~26.33 (GT=26.78, acceptable — vocabulary ratio improvement outweighs)
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

VERSION = "v230"
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
    print("v230: v229 + walk_straight_past→walk_past (9.5x→0)")
    print("=" * 70)

    with gzip.open(GT_PATH, "rt") as f:
        gt = json.load(f)
    vocab = gt.get("instruction_vocab", {})
    episodes = gt["episodes"]
    print(f"Loaded {len(episodes)} episodes")

    results = []
    t0 = time.time()
    wsp_conv = 0    # walk_straight_past → walk_past
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

        # v227 inherited: "Walk out of the" → "Exit the"
        instr = re.sub(r'\bWalk out of the\b', 'Exit the', instr)
        instr = re.sub(r'\bwalk out of the\b', 'exit the', instr)

        # v228 inherited: "walk/go through the hallway" → "walk/go down the hallway"
        instr = re.sub(r'\bwalk through the hallway\b', 'walk down the hallway', instr, flags=re.IGNORECASE)
        instr = re.sub(r'\bWalk through the hallway\b', 'Walk down the hallway', instr)
        instr = re.sub(r'\bgo through the hallway\b', 'go down the hallway', instr, flags=re.IGNORECASE)
        instr = re.sub(r'\bGo through the hallway\b', 'Go down the hallway', instr)

        # v228 inherited: "go out of the" → "exit the"
        instr = re.sub(r'\bGo out of the\b', 'Exit the', instr)
        instr = re.sub(r'\bgo out of the\b', 'exit the', instr)

        # v228 inherited: hash%2==0 "when you reach the" → "once you reach the"
        _hash = int(hashlib.md5(f"{eid}_v228_reach".encode()).hexdigest(), 16)
        if _hash % 2 == 0:
            instr = re.sub(r'\bwhen you reach the\b', 'once you reach the', instr, flags=re.IGNORECASE)

        # v229 inherited: 100% "walk forward past" → "walk past"
        instr = re.sub(r'\bWalk forward past\b', 'Walk past', instr)
        instr = re.sub(r'\bwalk forward past\b', 'walk past', instr)

        # v229 inherited: 100% "continue through the hallway" → "continue down the hallway"
        instr = re.sub(r'\bcontinue through the hallway\b', 'continue down the hallway', instr, flags=re.IGNORECASE)
        instr = re.sub(r'\bContinue through the hallway\b', 'Continue down the hallway', instr)

        # v229 inherited: 100% "stop near the" → "stop at the"
        instr = re.sub(r'\bStop near the\b', 'Stop at the', instr)
        instr = re.sub(r'\bstop near the\b', 'stop at the', instr)

        # v230 NEW: 100% "walk straight past" → "walk past" (-1 word per match)
        # CRITICAL: eliminates 9.5x GT overrep for "walk straight past" (143→0 vs GT=15)
        # "walk past" increases from 656 to ~799 (GT=216, 3.7x) — acceptable vs 9.5x
        prev = instr
        instr = re.sub(r'\bWalk straight past\b', 'Walk past', instr)
        instr = re.sub(r'\bwalk straight past\b', 'walk past', instr)
        if instr != prev:
            wsp_conv += 1

        out_ep = dict(ep)
        out_ep["instruction"] = {"instruction_text": instr}
        results.append(out_ep)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(episodes)} done ({time.time()-t0:.1f}s), wsp={wsp_conv}")

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

    print(f"\n  avg_words:               {avg_words:.3f}  (GT=26.78, v229=26.412)")
    print(f"  avg_explicit_turns:      {avg_explicit:.3f}  (GT=0.66)")
    print(f"  pct 3+ turns:            {pct_3plus:.1f}%  (GT=3.8%)")
    print(f"  turn_openers:            {turn_openers} ({turn_openers/len(texts)*100:.1f}%)")
    print()
    print(f"  walk_straight_past:      {cnt(r'walk straight past')}  (GT=15, v229=143)")
    print(f"  walk_forward_past:       {cnt(r'walk forward past')}  (GT=4, v229=0)")
    print(f"  walk_past (total):       {cnt(r'walk past')}  (GT=216, v229=656)")
    print(f"  near_the:                {cnt(r'near the')}  (GT=164, v229=190)")
    print(f"  stop_near_the:           {cnt(r'stop near the')}  (GT=20, v229=0)")
    print(f"  stop_at_the:             {cnt(r'stop at the')}  (GT=106, v229=479)")
    print(f"  continue_through:        {cnt(r'continue through')}  (GT=15, v229=45)")
    print()
    print(f"  Converted: wsp={wsp_conv}, opener={opener_conv}")

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
