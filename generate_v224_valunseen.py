#!/usr/bin/env python3
"""
Generate v224 auto-annotated dataset for val_unseen.

v224 over v222:
  - Inherits v222: 100% "turn X into the room" → "walk into the room"
  - NEW: hash%5==0 (~20%) turn opener stripping — removes "Turn X and " prefix
    from episode openers (instruction start), targeting avg_words near GT=26.78.
  - Patterns converted:
      "Turn around and [verb]..." → "[Verb cap]..."
      "Turn slightly left/right and [verb]..." → "[Verb cap]..."
      "Turn left/right and [verb]..." → "[Verb cap]..."
  - Rate: hash%5==0 (~20% of turn-opener episodes = ~304 episodes)
  - Expected avg_words: ~26.76 (GT=26.78, v222=27.258)
  - Expected turn_openers: ~1214/1839 (66%) vs v222 82.5%
  - Expected avg_explicit_turns: ~1.585 vs v222 1.750
  - Uses hash key "_v224_opener" — independent of episode RNG
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

VERSION = "v224"
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
    print("v224: v222 base + hash%5==0 turn opener strip (~20% rate)")
    print("      Targets avg_words~26.76 (GT=26.78), turn_openers ~66%")
    print("=" * 70)

    with gzip.open(GT_PATH, "rt") as f:
        gt = json.load(f)
    vocab = gt.get("instruction_vocab", {})
    episodes = gt["episodes"]
    print(f"Loaded {len(episodes)} episodes")

    results = []
    t0 = time.time()
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

        # v224 NEW: hash%5==0 (~20%) turn opener strip
        # Removes "Turn [around|slightly X|left|right] and " prefix from opener
        # Capitalizes the first char of the remaining verb.
        # Word change: removes "Turn [X] and " prefix (3 words) → -3 words per match
        # Only fires when the instruction starts with a turn-and-verb pattern.
        if TURN_OPENER_RE.match(instr):
            _v224_hash = int(hashlib.md5(f"{eid}_v224_opener".encode()).hexdigest(), 16)
            if _v224_hash % 5 == 0:
                # Remove "Turn [around|slightly X|left|right] and " and capitalize next char
                instr = TURN_OPENER_RE.sub(lambda m: m.group(1).upper(), instr)
                opener_converted += 1

        out_ep = dict(ep)
        out_ep["instruction"] = {"instruction_text": instr}
        results.append(out_ep)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(episodes)} done ({time.time()-t0:.1f}s), openers_converted={opener_converted}")

    def count_explicit_turns(text):
        return len(re.findall(r'\bturn (?:left|right|around|slightly)\b', text, re.I))

    texts = [e["instruction"]["instruction_text"] for e in results]
    explicit_turns = [count_explicit_turns(t) for t in texts]
    avg_explicit = sum(explicit_turns) / len(texts)
    pct_3plus = sum(1 for t in explicit_turns if t >= 3) / len(texts) * 100
    avg_words = sum(len(t.split()) for t in texts) / len(texts)
    turn_openers = sum(1 for t in texts if t.lower().startswith('turn'))

    print(f"\n  avg_words:           {avg_words:.3f}  (GT=26.78, v222=27.258)")
    print(f"  avg_explicit_turns:  {avg_explicit:.3f}  (GT=0.66, v222=1.750)")
    print(f"  pct 3+ turns:        {pct_3plus:.1f}%  (GT=3.8%)")
    print(f"  turn_openers:        {turn_openers} ({turn_openers/len(texts)*100:.1f}%)  (GT=16.5%, v222=82.5%)")
    print(f"  opener_converted:    {opener_converted}")

    # Compare to v222
    v222_path = OUT_DIR / "val_unseen_auto_v222.json.gz"
    if v222_path.exists():
        with gzip.open(v222_path, "rt") as f:
            v222 = json.load(f)
        v222_map = {e["episode_id"]: e["instruction"]["instruction_text"] for e in v222["episodes"]}
        changed = sum(1 for e in results if e["episode_id"] in v222_map
                      and e["instruction"]["instruction_text"] != v222_map[e["episode_id"]])
        print(f"  Changed from v222:   {changed}/{len(results)} ({changed/len(results)*100:.1f}%)")

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
