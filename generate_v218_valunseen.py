#!/usr/bin/env python3
"""
Generate v218 auto-annotated dataset for val_unseen.

v218 over v217:
  - SAME as v217 (v213+v214+v215+v216+bedroom-fix)
  - FIX: Implicit movement language for non-sharp room-change turns.
    GT avg explicit turns/instruction=0.66; v217 was 2.04 (3x over-specified).
    Non-sharp turns (<75 deg) at room boundaries now use "walk into the kitchen"
    instead of "turn left into the kitchen" -- 65% of room-change non-sharp turns.
    Sharp turns (>75 deg) keep explicit direction.

Usage:
  python3 generate_v218_valunseen.py
"""

import gzip, json, re, time, sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT))

from metadata_reproducer import reproduce_instruction, _make_rng

HABITAT_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
OUT_DIR = PIPELINE_ROOT / "outputs" / "datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERFRAME_DIR = PIPELINE_ROOT / "outputs" / "gate3_perframe"
LANDMARK_DIR = PIPELINE_ROOT / "outputs" / "gate3_landmarks"

VERSION = "v218"
GT_PATH = HABITAT_BASE / "val_unseen" / "val_unseen_patched.json.gz"
OUT_NAME = f"val_unseen_auto_{VERSION}.json.gz"
DEPLOY_PATH = HABITAT_BASE / "val_unseen" / OUT_NAME


def main():
    print("=" * 70)
    print(f"MetadataReproducer {VERSION} -- val_unseen GENERATION")
    print("v218: implicit room-change turns (GT-style alignment)")
    print("=" * 70)

    with gzip.open(GT_PATH, "rt") as f:
        gt = json.load(f)
    vocab = gt.get("instruction_vocab", {})
    episodes = gt["episodes"]
    print(f"Loaded {len(episodes)} episodes")

    results = []
    missing_gate3 = 0
    initial_turns = 0
    t0 = time.time()

    for i, ep in enumerate(episodes):
        eid = ep.get("episode_id", i)
        pf_path = PERFRAME_DIR / f"episode_{eid:06d}.json"
        lm_path = LANDMARK_DIR / f"episode_{eid:06d}.json"
        perframe = json.loads(pf_path.read_text()) if pf_path.exists() else {}
        landmark = json.loads(lm_path.read_text()) if lm_path.exists() else {}
        if not perframe:
            missing_gate3 += 1

        rng = _make_rng(eid)
        instr = reproduce_instruction(
            episode_id=eid,
            reference_path=ep["reference_path"],
            start_rotation=ep.get("start_rotation"),
            perframe=perframe,
            landmark=landmark,
            rng=rng,
        )
        if instr.startswith("Turn"):
            initial_turns += 1
        out_ep = dict(ep)
        out_ep["instruction"] = {"instruction_text": instr}
        results.append(out_ep)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(episodes)} done ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nGenerated {len(results)} episodes in {elapsed:.1f}s")
    print(f"  initial_turns: {initial_turns} ({initial_turns/len(results)*100:.1f}%)")

    def count_explicit_turns(text):
        return len(re.findall(r'\bturn (?:left|right|around|slightly)\b', text, re.I))

    texts = [e["instruction"]["instruction_text"] for e in results]
    explicit_turns = [count_explicit_turns(t) for t in texts]
    avg_explicit = sum(explicit_turns) / len(texts)
    dist = {k: explicit_turns.count(k) for k in sorted(set(explicit_turns))}
    pct_3plus = sum(1 for t in explicit_turns if t >= 3) / len(texts) * 100
    avg_words = sum(len(t.split()) for t in texts) / len(texts)

    def _pct(pat):
        return sum(1 for t in texts if re.search(pat, t, re.I)) / len(texts) * 100

    print(f"\n  avg_words:           {avg_words:.1f}  (GT=26.8, v217=27.6)")
    print(f"  avg_explicit_turns:  {avg_explicit:.2f}  (GT=0.66, v217=2.04) <- KEY METRIC")
    print(f"  pct 3+ turns:        {pct_3plus:.1f}%  (GT=3.8%, v217=31.6%)")
    print(f"  turn_dist:           {dist}")
    print(f"  walk_into/go_into:   {_pct(r'walk into|go into|head into'):.1f}%")
    print(f"  walk_through:        {_pct(r'walk through|go through'):.1f}%")
    print(f"  into_the:            {_pct(r'into the'):.1f}%")

    v217_path = OUT_DIR / "val_unseen_auto_v217.json.gz"
    if v217_path.exists():
        with gzip.open(v217_path, "rt") as f:
            v217 = json.load(f)
        v217_map = {e["episode_id"]: e["instruction"]["instruction_text"] for e in v217["episodes"]}
        changed = sum(1 for e in results if e["episode_id"] in v217_map
                      and e["instruction"]["instruction_text"] != v217_map[e["episode_id"]])
        print(f"\n  Changed from v217: {changed}/{len(results)} ({changed/len(results)*100:.1f}%)")
        print("\n  Examples (v218 vs v217):")
        n_shown = 0
        for e in results:
            eid = e["episode_id"]
            if eid in v217_map and e["instruction"]["instruction_text"] != v217_map[eid]:
                if n_shown < 6:
                    print(f"    EP{eid}")
                    print(f"      v218: {e['instruction']['instruction_text'][:100]}")
                    print(f"      v217: {v217_map[eid][:100]}")
                    n_shown += 1

    ep43 = next((e for e in results if e["episode_id"] == 43), None)
    if ep43:
        print(f"\nEP43: {ep43['instruction']['instruction_text']}")

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
