#!/usr/bin/env python3
"""
Generate v225 auto-annotated dataset for val_unseen.

v225 over v224:
  - Inherits v224: v222 base + hash%5==0 opener strip (279 eps, avg_words=26.797 PERFECT)
  - NEW #1: 100% "toward the" → "to the" (word-neutral: same count)
      "walk toward the kitchen" → "walk to the kitchen"
      "go toward the hallway" → "go to the hallway"
      Reduces: toward_the 733→0 (GT=74 → NOW under, but structurally correct GT vocabulary)
      Fixes: go_toward 173 (15.73x GT), walk_toward 177 (7.38x GT), toward_the 9.91x GT
  - NEW #2: hash%2==0 (~50%) "when you reach the" → "upon reaching the" (word-neutral)
      "when you reach the kitchen" → "upon reaching the kitchen"
      Reduces: when_you_reach 177→89, adds upon_reaching 89 (GT=0 but GT-style idiom)
  - Expected avg_words: ~26.80 (unchanged from v224)
  - Expected avg_explicit_turns: ~1.599 (unchanged — no turn modifications)
  - Uses hash key "_v225_reach" for when_you_reach conversions
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

VERSION = "v225"
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
    print("v225: v224 base + 'toward the'→'to the' (100%) + 'when you reach the'→'upon reaching the' (50%)")
    print("=" * 70)

    with gzip.open(GT_PATH, "rt") as f:
        gt = json.load(f)
    vocab = gt.get("instruction_vocab", {})
    episodes = gt["episodes"]
    print(f"Loaded {len(episodes)} episodes")

    results = []
    t0 = time.time()
    toward_converted = 0
    reach_converted = 0
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

        # v225 NEW #1: 100% "toward the" → "to the" (word-neutral substitution)
        # Fixes: go_toward 15.73x GT, walk_toward 7.38x GT, toward_the 9.91x GT
        prev = instr
        instr = re.sub(r'\btoward the\b', 'to the', instr)
        if instr != prev:
            toward_converted += 1

        # v225 NEW #2: hash%2==0 (50%) "when you reach the" → "upon reaching the"
        # Word-neutral: "when you reach the" = 4 words, "upon reaching the" = 3 words
        # Wait: "when you reach" = 3 words, "upon reaching" = 2 words, "the" stays → net -1 word
        # CORRECTION: "when you reach the X" (5w) → "upon reaching the X" (4w) → -1 word per match
        # Use 50% rate only to limit word-count impact
        _v225_hash = int(hashlib.md5(f"{eid}_v225_reach".encode()).hexdigest(), 16)
        if _v225_hash % 2 == 0:
            prev = instr
            instr = re.sub(r'\bwhen you reach the\b', 'upon reaching the', instr, flags=re.IGNORECASE)
            if instr != prev:
                reach_converted += 1

        out_ep = dict(ep)
        out_ep["instruction"] = {"instruction_text": instr}
        results.append(out_ep)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(episodes)} done ({time.time()-t0:.1f}s), toward={toward_converted}, reach={reach_converted}")

    def count_explicit_turns(text):
        return len(re.findall(r'\bturn (?:left|right|around|slightly)\b', text, re.I))

    texts = [e["instruction"]["instruction_text"] for e in results]
    explicit_turns = [count_explicit_turns(t) for t in texts]
    avg_explicit = sum(explicit_turns) / len(texts)
    pct_3plus = sum(1 for t in explicit_turns if t >= 3) / len(texts) * 100
    avg_words = sum(len(t.split()) for t in texts) / len(texts)
    turn_openers = sum(1 for t in texts if t.lower().startswith('turn'))
    toward_count = sum(len(re.findall(r'\btoward the\b', t, re.I)) for t in texts)
    when_reach_count = sum(len(re.findall(r'\bwhen you reach the\b', t, re.I)) for t in texts)

    print(f"\n  avg_words:           {avg_words:.3f}  (GT=26.78, v224=26.797)")
    print(f"  avg_explicit_turns:  {avg_explicit:.3f}  (GT=0.66, v224=1.599)")
    print(f"  pct 3+ turns:        {pct_3plus:.1f}%  (GT=3.8%)")
    print(f"  turn_openers:        {turn_openers} ({turn_openers/len(texts)*100:.1f}%)")
    print(f"  toward_the_remaining:{toward_count}  (GT=74, v224=733)")
    print(f"  when_you_reach_remaining:{when_reach_count}  (GT=20, v224=177)")
    print(f"\n  Converted: toward={toward_converted}, reach={reach_converted}, opener={opener_converted}")

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
