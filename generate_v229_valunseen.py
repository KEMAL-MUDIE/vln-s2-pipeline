#!/usr/bin/env python3
"""
Generate v229 auto-annotated dataset for val_unseen.

v229 over v228:
  - Inherits all v228 fixes
  - NEW #1: 100% "walk forward past" → "walk past" (-1 word per match)
      CRITICAL: walk_forward_past 247→0 (61x GT → 0!), walk_past 409→656 (GT=216, 3.04x)
      The 61x overrep is far worse than the resulting 3.04x overrep — clear win.
      All 247 "walk forward past" in v228 occur as "walk forward past the X" (past is a preposition).
  - NEW #2: 100% "continue through the hallway" → "continue down the hallway" (word-neutral)
      Extends v228's walk/go-through-hallway fix to continue-through-hallway.
      Reduces: continue_through 57→45 (GT=15, 3.80x→3.0x).
  - NEW #3: 100% "stop near the" → "stop at the" (word-neutral)
      CRITICAL: stop_near_the 267→0 (13.35x GT → 0!), stop_at_the 212→479 (GT=106, 4.52x).
      Same logic: 13.35x is far worse than resulting 4.52x.
  - Word impact: "walk forward past" → "walk past" costs -1 word × 247 = -0.134 avg_words
  - Expected avg_words: ~26.41 (GT=26.78, slightly more below than v228=26.546 — acceptable)
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

VERSION = "v229"
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
    print("v229: v228 + walk_forward_past→walk_past + continue_through_hallway→continue_down + stop_near→stop_at")
    print("=" * 70)

    with gzip.open(GT_PATH, "rt") as f:
        gt = json.load(f)
    vocab = gt.get("instruction_vocab", {})
    episodes = gt["episodes"]
    print(f"Loaded {len(episodes)} episodes")

    results = []
    t0 = time.time()
    wfp_conv = 0      # walk_forward_past → walk_past
    cth_conv = 0      # continue_through_hallway → continue_down_hallway
    snear_conv = 0    # stop_near_the → stop_at_the
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

        # v228 inherited: "walk through the hallway" → "walk down the hallway"
        instr = re.sub(r'\bwalk through the hallway\b', 'walk down the hallway', instr, flags=re.IGNORECASE)
        instr = re.sub(r'\bWalk through the hallway\b', 'Walk down the hallway', instr)

        # v228 inherited: "go through the hallway" → "go down the hallway"
        instr = re.sub(r'\bgo through the hallway\b', 'go down the hallway', instr, flags=re.IGNORECASE)
        instr = re.sub(r'\bGo through the hallway\b', 'Go down the hallway', instr)

        # v228 inherited: "go out of the" → "exit the"
        instr = re.sub(r'\bGo out of the\b', 'Exit the', instr)
        instr = re.sub(r'\bgo out of the\b', 'exit the', instr)

        # v228 inherited: hash%2==0 "when you reach the" → "once you reach the"
        _hash = int(hashlib.md5(f"{eid}_v228_reach".encode()).hexdigest(), 16)
        if _hash % 2 == 0:
            instr = re.sub(r'\bwhen you reach the\b', 'once you reach the', instr, flags=re.IGNORECASE)

        # v229 NEW #1: 100% "walk forward past" → "walk past" (-1 word per match)
        # CRITICAL: eliminates 61x GT overrep for "walk forward past" (247→0 vs GT=4)
        # "walk past" increases from 409 to 656 (GT=216, 3.04x) — acceptable vs 61x
        prev = instr
        instr = re.sub(r'\bWalk forward past\b', 'Walk past', instr)
        instr = re.sub(r'\bwalk forward past\b', 'walk past', instr)
        if instr != prev:
            wfp_conv += 1

        # v229 NEW #2: 100% "continue through the hallway" → "continue down the hallway" (word-neutral)
        # Extends v228's walk/go-through-hallway fix to continue variant (12 instances)
        prev = instr
        instr = re.sub(r'\bcontinue through the hallway\b', 'continue down the hallway', instr, flags=re.IGNORECASE)
        instr = re.sub(r'\bContinue through the hallway\b', 'Continue down the hallway', instr)
        if instr != prev:
            cth_conv += 1

        # v229 NEW #3: 100% "stop near the" → "stop at the" (word-neutral)
        # CRITICAL: eliminates 13.35x GT overrep for "stop near the" (267→0 vs GT=20)
        # "stop at the" increases from 212 to 479 (GT=106, 4.52x) — less severe than 13.35x
        prev = instr
        instr = re.sub(r'\bStop near the\b', 'Stop at the', instr)
        instr = re.sub(r'\bstop near the\b', 'stop at the', instr)
        if instr != prev:
            snear_conv += 1

        out_ep = dict(ep)
        out_ep["instruction"] = {"instruction_text": instr}
        results.append(out_ep)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(episodes)} done ({time.time()-t0:.1f}s), wfp={wfp_conv}, cth={cth_conv}, snear={snear_conv}")

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

    print(f"\n  avg_words:               {avg_words:.3f}  (GT=26.78, v228=26.546)")
    print(f"  avg_explicit_turns:      {avg_explicit:.3f}  (GT=0.66)")
    print(f"  pct 3+ turns:            {pct_3plus:.1f}%  (GT=3.8%)")
    print(f"  turn_openers:            {turn_openers} ({turn_openers/len(texts)*100:.1f}%)")
    print()
    print(f"  walk_forward_past:       {cnt(r'walk forward past')}  (GT=4, v228=247)")
    print(f"  walk_past (total):       {cnt(r'walk past')}  (GT=216, v228=409)")
    print(f"  walk_straight_past:      {cnt(r'walk straight past')}  (GT=15, v228=143)")
    print(f"  continue_through:        {cnt(r'continue through')}  (GT=15, v228=57)")
    print(f"  continue_down:           {cnt(r'continue down')}  (GT=30, v228=~12)")
    print(f"  continue_through_hall:   {cnt(r'continue through the hallway')}  (v228=12)")
    print(f"  stop_near_the:           {cnt(r'stop near the')}  (GT=20, v228=267)")
    print(f"  stop_at_the:             {cnt(r'stop at the')}  (GT=106, v228=212)")
    print(f"  wait_near_the:           {cnt(r'wait near the')}  (GT=94, v228=190)")
    print(f"  near_the:                {cnt(r'near the')}  (GT=164, v228=457)")
    print()
    print(f"  Converted: wfp={wfp_conv}, cth={cth_conv}, snear={snear_conv}, opener={opener_conv}")

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
