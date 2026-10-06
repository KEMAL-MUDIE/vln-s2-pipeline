#!/usr/bin/env python3
"""
Generate v227 auto-annotated dataset for val_unseen.

v227 over v226:
  - Inherits all v226 fixes (toward→to, walk_into→enter 50%, when_reach→once_reach 20%)
  - NEW: 100% "walk out of the" → "exit the" (-2 words per match)
      "walk out of the bedroom" → "exit the bedroom"
      "walk out of the kitchen" → "exit the kitchen"
      Double win: reduces walk_out_of_the (1.68x GT) AND fills exit_the underrep (0.38x GT)
      After: walk_out_of_the≈0, exit_the≈201 (GT=278, 0.72x — big improvement from 0.38x)
  - Word change: "walk out of the" (4w) → "exit the" (2w) = -2 words per match
  - Expected: ~96 conversions → -192 words / 1839 = -0.104 per episode
  - Expected avg_words: ~26.564 (GT=26.78, acceptable -0.216)
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

VERSION = "v227"
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
    print("v227: v226 + walk_out_of_the→exit_the (100%)")
    print("=" * 70)

    with gzip.open(GT_PATH, "rt") as f:
        gt = json.load(f)
    vocab = gt.get("instruction_vocab", {})
    episodes = gt["episodes"]
    print(f"Loaded {len(episodes)} episodes")

    results = []
    t0 = time.time()
    exit_converted = 0
    opener_converted = 0

    for i, ep in enumerate(episodes):
        eid = ep.get("episode_id", i)
        pf_path = PERFRAME_DIR / f"episode_{eid:06d}.json"
        lm_path = LANDMARK_DIR / f"episode_{eid:06d}.json"
        perframe = json.loads(pf_path.read_text()) if pf_path.exists() else {}
        landmark = json.loads(lm_path.read_text()) if lm_path.exists() else {}
        rng = _make_rng(eid)
        instr = reproduce_instruction(eid, ep["reference_path"], ep.get("start_rotation"), perframe, landmark, rng)

        # v222 inherited: 100% "turn X into the room" → "walk into the room"
        instr = re.sub(r'\bTurn (?:left|right) into the\b', 'Walk into the', instr)
        instr = re.sub(r'\bturn (?:left|right) into the\b', 'walk into the', instr)

        # v224 inherited: hash%5==0 (~20%) turn opener strip
        if TURN_OPENER_RE.match(instr):
            _v224_hash = int(hashlib.md5(f"{eid}_v224_opener".encode()).hexdigest(), 16)
            if _v224_hash % 5 == 0:
                instr = TURN_OPENER_RE.sub(lambda m: m.group(1).upper(), instr)
                opener_converted += 1

        # v225 inherited: 100% "toward the" → "to the"
        instr = re.sub(r'\btoward the\b', 'to the', instr)

        # v226 inherited: hash%2==0 (50%) "Walk into the" → "Enter the"
        _v226_enter_hash = int(hashlib.md5(f"{eid}_v226_enter".encode()).hexdigest(), 16)
        if _v226_enter_hash % 2 == 0:
            instr = re.sub(r'\bWalk into the\b', 'Enter the', instr)
            instr = re.sub(r'\bwalk into the\b', 'enter the', instr)

        # v226 inherited: hash%5==0 (20%) "when you reach the" → "once you reach the"
        _v226_reach_hash = int(hashlib.md5(f"{eid}_v226_reach".encode()).hexdigest(), 16)
        if _v226_reach_hash % 5 == 0:
            instr = re.sub(r'\bwhen you reach the\b', 'once you reach the', instr, flags=re.IGNORECASE)

        # v227 NEW: 100% "Walk out of the" → "Exit the" (-2 words per match)
        # Fixes: walk_out_of_the 1.68x→0.00x; exit_the 0.38x→0.72x
        # Sentence start (capital W→E) and mid-sentence (lowercase w→e)
        prev = instr
        instr = re.sub(r'\bWalk out of the\b', 'Exit the', instr)
        instr = re.sub(r'\bwalk out of the\b', 'exit the', instr)
        if instr != prev:
            exit_converted += 1

        out_ep = dict(ep)
        out_ep["instruction"] = {"instruction_text": instr}
        results.append(out_ep)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(episodes)} done ({time.time()-t0:.1f}s), exit_conv={exit_converted}")

    def count_explicit_turns(text):
        return len(re.findall(r'\bturn (?:left|right|around|slightly)\b', text, re.I))

    texts = [e["instruction"]["instruction_text"] for e in results]
    explicit_turns = [count_explicit_turns(t) for t in texts]
    avg_explicit = sum(explicit_turns) / len(texts)
    pct_3plus = sum(1 for t in explicit_turns if t >= 3) / len(texts) * 100
    avg_words = sum(len(t.split()) for t in texts) / len(texts)
    turn_openers = sum(1 for t in texts if t.lower().startswith('turn'))

    walk_out = sum(len(re.findall(r'\bwalk out of the\b', t, re.I)) for t in texts)
    exit_the = sum(len(re.findall(r'\bexit the\b', t, re.I)) for t in texts)
    walk_into = sum(len(re.findall(r'\bwalk into the\b', t, re.I)) for t in texts)
    enter_the = sum(len(re.findall(r'\benter the\b', t, re.I)) for t in texts)

    print(f"\n  avg_words:           {avg_words:.3f}  (GT=26.78, v226=26.668)")
    print(f"  avg_explicit_turns:  {avg_explicit:.3f}  (GT=0.66)")
    print(f"  pct 3+ turns:        {pct_3plus:.1f}%  (GT=3.8%)")
    print(f"  turn_openers:        {turn_openers} ({turn_openers/len(texts)*100:.1f}%)")
    print(f"  walk_out_of_the:     {walk_out}  (GT=57, v226=96)")
    print(f"  exit_the:            {exit_the}  (GT=278, v226=105)")
    print(f"  walk_into_the:       {walk_into}  (GT=128, v226=247)")
    print(f"  enter_the:           {enter_the}  (GT=291, v226=237)")
    print(f"\n  exit_converted={exit_converted}")

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
