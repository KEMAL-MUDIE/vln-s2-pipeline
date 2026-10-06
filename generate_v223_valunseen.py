#!/usr/bin/env python3
"""
Generate v221 auto-annotated dataset for val_unseen.

v222 over v221:
  - Increases implicit conversion rate: 100% (all occurrences converted)
  - Same pattern: "turn left/right into the [room]" → "Walk into the [room]"
  - Expected avg_explicit_turns: ~1.73 (GT=0.66, v220=1.84)
  - Uses new hash key _v221_implicit to avoid overlap with v220
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

VERSION = "v223"
GT_PATH = HABITAT_BASE / "val_unseen" / "val_unseen_patched.json.gz"
OUT_NAME = f"val_unseen_auto_{VERSION}.json.gz"
DEPLOY_PATH = HABITAT_BASE / "val_unseen" / OUT_NAME

def main():
    print("=" * 70)
    print(f"MetadataReproducer {VERSION} -- val_unseen")
    print("v223: convert turn-X-and-walk-into-room to implicit walk-into-room")
    print("=" * 70)

    with gzip.open(GT_PATH, "rt") as f:
        gt = json.load(f)
    vocab = gt.get("instruction_vocab", {})
    episodes = gt["episodes"]
    print(f"Loaded {len(episodes)} episodes")

    results = []
    t0 = time.time()
    for i, ep in enumerate(episodes):
        eid = ep.get("episode_id", i)
        pf_path = PERFRAME_DIR / f"episode_{eid:06d}.json"
        lm_path = LANDMARK_DIR / f"episode_{eid:06d}.json"
        perframe = json.loads(pf_path.read_text()) if pf_path.exists() else {}
        landmark = json.loads(lm_path.read_text()) if lm_path.exists() else {}
        rng = _make_rng(eid)
        instr = reproduce_instruction(eid, ep["reference_path"], ep.get("start_rotation"), perframe, landmark, rng)

        # v222 → v223 inherited: "turn X into the room" → "Walk into the room" (100%)
        instr = re.sub(r'\bTurn (?:left|right) into the\b', 'Walk into the', instr)
        instr = re.sub(r'\bturn (?:left|right) into the\b', 'walk into the', instr)

        # v223 NEW: "Turn X and walk/go/continue/head [prep] the [room]" → "[Verb] [prep] the [room]"
        # Removes explicit turn direction when continuation provides clear room-based context.
        # Examples: "Turn right and walk through the hallway" → "Walk through the hallway"
        #           "Turn left and continue into the living room" → "Continue into the living room"
        # Word change: removes "Turn X and " prefix (3 words) → -3 words per match
        # 100% conversion — no hash (all occurrences)
        instr = re.sub(
            r'\bTurn (?:left|right) and (walk|go|continue|head) (into|through|down|along|toward) the\b',
            lambda m: m.group(1).capitalize() + ' ' + m.group(2) + ' the',
            instr
        )
        instr = re.sub(
            r'\bturn (?:left|right) and (walk|go|continue|head) (into|through|down|along|toward) the\b',
            lambda m: m.group(1) + ' ' + m.group(2) + ' the',
            instr
        )

        out_ep = dict(ep)
        out_ep["instruction"] = {"instruction_text": instr}
        results.append(out_ep)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(episodes)} done ({time.time()-t0:.1f}s)")

    def count_explicit_turns(text):
        return len(re.findall(r'\bturn (?:left|right|around|slightly)\b', text, re.I))

    texts = [e["instruction"]["instruction_text"] for e in results]
    explicit_turns = [count_explicit_turns(t) for t in texts]
    avg_explicit = sum(explicit_turns) / len(texts)
    pct_3plus = sum(1 for t in explicit_turns if t >= 3) / len(texts) * 100
    avg_words = sum(len(t.split()) for t in texts) / len(texts)

    print(f"\n  avg_words:           {avg_words:.1f}  (GT=26.8)")
    print(f"  avg_explicit_turns:  {avg_explicit:.2f}  (GT=0.66, v218=1.97) <- KEY")
    print(f"  pct 3+ turns:        {pct_3plus:.1f}%  (GT=3.8%)")

    # Compare to v218
    v218_path = OUT_DIR / "val_unseen_auto_v218.json.gz"
    if v218_path.exists():
        with gzip.open(v218_path, "rt") as f:
            v218 = json.load(f)
        v218_map = {e["episode_id"]: e["instruction"]["instruction_text"] for e in v218["episodes"]}
        changed = sum(1 for e in results if e["episode_id"] in v218_map
                      and e["instruction"]["instruction_text"] != v218_map[e["episode_id"]])
        print(f"\n  Changed from v218: {changed}/{len(results)} ({changed/len(results)*100:.1f}%)")

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
