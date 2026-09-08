#!/usr/bin/env python3
"""
Generate v217 auto-annotated dataset for val_unseen.

v217 over v216:
  - Same as v216 (v213 + v214 stop-quality + v215 door-through + v216 room-at-turn)
  - FIX: bedroom removed from Rule 3 SPECIFIC_ROOMS.
    Bedroom→bedroom same-room paths keep their "bed" stop landmark (useful visual anchor).
    Bathroom/kitchen/office same-room Rule 3 override kept (unambiguously wrong there).

Usage:
  python3 generate_v217_valunseen.py
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

VERSION = "v217"

GT_PATH = HABITAT_BASE / "val_unseen" / "val_unseen_patched.json.gz"
OUT_NAME = f"val_unseen_auto_{VERSION}.json.gz"
DEPLOY_PATH = HABITAT_BASE / "val_unseen" / OUT_NAME


def main():
    print("=" * 70)
    print(f"MetadataReproducer {VERSION} — val_unseen GENERATION")
    print("v217: bedroom excluded from Rule 3 (keeps 'bed' anchor for bedroom paths)")
    print("=" * 70)

    # Load vocab from GT
    with gzip.open(GT_PATH, "rt") as f:
        gt = json.load(f)
    vocab = gt.get("instruction_vocab", {})
    episodes = gt["episodes"]
    print(f"Loaded {len(episodes)} episodes, vocab={len(vocab.get('word_list', []))} words")

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
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(episodes)} done ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nGenerated {len(results)} episodes in {elapsed:.1f}s")
    print(f"  initial_turns: {initial_turns} ({initial_turns/len(results)*100:.1f}%)")
    if missing_gate3:
        print(f"  WARNING: {missing_gate3} episodes missing gate3 data → path-only")

    # Vocab stats
    texts = [e["instruction"]["instruction_text"] for e in results]
    avg_words = sum(len(t.split()) for t in texts) / len(texts)
    def _pct(pat):
        return sum(1 for t in texts if re.search(pat, t, re.I)) / len(texts) * 100

    print(f"\n  avg_words:    {avg_words:.1f}")
    print(f"  walk_through: {_pct(r'walk through'):.1f}%")
    print(f"  walk_past:    {_pct(r'walk past'):.1f}%")
    print(f"  stop:         {_pct(r'\bstop\b'):.1f}%")
    print(f"  into the:     {_pct(r'into the'):.1f}%")

    # Count changes vs v213
    v213_path = OUT_DIR / "val_unseen_auto_v213.json.gz"
    if v213_path.exists():
        with gzip.open(v213_path, "rt") as f:
            v213 = json.load(f)
        v213_map = {e["episode_id"]: e["instruction"]["instruction_text"] for e in v213["episodes"]}
        changed = sum(1 for e in results if e["episode_id"] in v213_map
                      and e["instruction"]["instruction_text"] != v213_map[e["episode_id"]])
        print(f"\n  Changed from v213: {changed}/{len(results)} ({changed/len(results)*100:.1f}%)")

    # Count changes vs v216
    v216_path = OUT_DIR / "val_unseen_generated_meta_v216.json.gz"
    if v216_path.exists():
        with gzip.open(v216_path, "rt") as f:
            v216 = json.load(f)
        v216_map = {e["episode_id"]: e["instruction"]["instruction_text"] for e in v216["episodes"]}
        changed16 = sum(1 for e in results if e["episode_id"] in v216_map
                        and e["instruction"]["instruction_text"] != v216_map[e["episode_id"]])
        print(f"  Changed from v216: {changed16}/{len(results)} ({changed16/len(results)*100:.1f}%)")
        # Show some fixed episodes (bedroom-bedroom cases)
        print("\n  Fixed bedroom regression examples (v216→v217):")
        n_shown = 0
        for e in results:
            eid = e["episode_id"]
            if eid in v216_map and e["instruction"]["instruction_text"] != v216_map[eid]:
                v216_txt = v216_map[eid]
                v217_txt = e["instruction"]["instruction_text"]
                if n_shown < 5:
                    print(f"    EP{eid}: {v217_txt[:80]}")
                    print(f"      was: {v216_txt[:80]}")
                    n_shown += 1

    # Save local
    out_data = {"episodes": results, "instruction_vocab": vocab}
    local_path = OUT_DIR / OUT_NAME
    with gzip.open(local_path, "wt") as f:
        json.dump(out_data, f)
    size_kb = local_path.stat().st_size // 1024
    print(f"\nSaved:    {local_path} ({size_kb} KB)")

    # Deploy to NVMe
    DEPLOY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(DEPLOY_PATH, "wt") as f:
        json.dump(out_data, f)
    print(f"Deployed: {DEPLOY_PATH}")

    # Check EP115 specifically
    ep115 = next((e for e in results if e["episode_id"] == 115), None)
    if ep115:
        print(f"\nEP115 check (bedroom regression): {ep115['instruction']['instruction_text']}")

    print(f"\n{VERSION} ready! Launch eval with:")
    print(f"  nohup bash scripts/run_eval_valunseen_auto_v217.sh &")


if __name__ == "__main__":
    main()
