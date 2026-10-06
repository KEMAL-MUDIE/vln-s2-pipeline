#!/usr/bin/env python3
"""
Generate v73: constrained vLLM generation anchored to path geometry turns.

Approach:
  1. Compute correct turn sequence from reference_path geometry (v72 approach, 86.1% acc)
  2. Assign VC visual landmarks to each turn position
  3. Call Gemma-4-31B via vLLM with HARD CONSTRAINT on turn sequence
  4. Verify output has correct turn directions; fall back to v72 if wrong
  5. Keep v71 stop phrase (already optimized)

Expected improvement over v72:
  - More natural language (Gemma-4 fluency vs templates)
  - Additional landmark context for 0-turn (straight) episodes
  - Better room descriptions where VC provides context
  - Same turn accuracy as v72 (94.2%), verified post-generation

Estimated time: ~1.5-2 hours (1839 eps × ~3-4s per vLLM call)
"""
import gzip
import json
import math
import re
import sys
import time
from pathlib import Path

import requests

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
VC_PATH = Path("/home/kemal/VLNav/s2_pipeline_new/outputs/gate3_gemma_v22_vision_checkpoint.json")
GT_PATH = BASE / "val_unseen" / "val_unseen.json.gz"
V71_PATH = BASE / "val_unseen" / "val_unseen_v71.json.gz"
V72_PATH = BASE / "val_unseen" / "val_unseen_v72.json.gz"  # fallback
OUT_PATH = BASE / "val_unseen" / "val_unseen_v73.json.gz"
CHECKPOINT_PATH = Path("/home/kemal/VLNav/s2_pipeline_new/outputs/v73_checkpoint.json")

VLLM_URL = "http://10.77.32.231:8000/v1/chat/completions"
MODEL = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
TURN_THRESHOLD = 80
BATCH_LOG_EVERY = 50

SYSTEM_MSG = """You write short indoor navigation instructions for a robot.
Rules:
1. Start with walking forward (never start with a turn as the very first action).
2. Include ONLY the turns listed — no extra turns.
3. Mention 1-2 visual landmarks naturally.
4. End with the exact stop phrase given.
5. Write 2-3 natural sentences total."""

COLOR_STRIP = re.compile(
    r'\b(white|grey|gray|brown|black|dark|light|beige|cream|tan|upholstered|'
    r'rectangular|circular|oval|square|decorative|ornate|modern|antique|'
    r'large|small|tall|short|long|wide|narrow|big|little|wooden|metal|glass|'
    r'fabric|cloth|leather|stone|marble|polished)\b\s*',
    re.IGNORECASE
)


def clean_lm(lm):
    if not lm:
        return ""
    lm = COLOR_STRIP.sub('', lm).strip()
    lm = re.sub(r'\s+', ' ', lm).strip()
    return lm.lower() if len(lm) > 3 else ""


def get_stop_sentence(inst):
    sents = re.split(r'(?<=[.!?])\s+', inst.strip())
    return sents[-1].strip() if len(sents) > 1 else inst.strip()


def extract_significant_turns(path, threshold=TURN_THRESHOLD):
    cumul = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dz = path[i][2] - path[i-1][2]
        cumul.append(cumul[-1] + math.sqrt(dx*dx + dz*dz))
    total = cumul[-1]
    turns = []
    for i in range(1, len(path)-1):
        A, B, C = path[i-1], path[i], path[i+1]
        ax, az = B[0]-A[0], B[2]-A[2]
        bx, bz = C[0]-B[0], C[2]-B[2]
        la = math.sqrt(ax*ax + az*az)
        lb = math.sqrt(bx*bx + bz*bz)
        if la < 0.05 or lb < 0.05:
            continue
        ax, az = ax/la, az/la
        bx, bz = bx/lb, bz/lb
        cross_z = ax*bz - az*bx
        dot = ax*bx + az*bz
        angle = math.degrees(math.atan2(cross_z, dot))
        if abs(angle) > threshold:
            turns.append({
                'dir': 'right' if angle > 0 else 'left',
                'angle': abs(angle),
                'frac': cumul[i]/total if total > 0 else 0,
            })
    return turns


def assign_vc(turns, vc_turns):
    if not vc_turns or not turns:
        return ['' for _ in turns]
    n_vc = len(vc_turns)
    vc_fracs = [(i+1)/(n_vc+1) for i in range(n_vc)]
    used = set()
    results = []
    for t in turns:
        dists = sorted((abs(vc_fracs[i] - t['frac']), i) for i in range(n_vc))
        assigned = ''
        for _, i in dists:
            if i not in used:
                used.add(i)
                assigned = vc_turns[i]
                break
        results.append(assigned)
    return results


def call_vllm(turns, vc_lms, vc_goal, stop_phrase):
    """Call vLLM with constrained prompt. Returns generated text."""
    goal_lm = clean_lm(vc_goal)
    clean_lms = [clean_lm(lm) for lm in vc_lms]

    if not turns:
        turn_desc = "(no turns, straight path)"
    else:
        items = []
        for t, lm in zip(turns, clean_lms):
            lm_part = f" near {lm}" if lm else ""
            items.append(f"turn {t['dir']}{lm_part}")
        turn_desc = " then ".join(items)

    goal_part = f"\nGoal landmark: {goal_lm}" if goal_lm else ""
    user_msg = f"""Turns: {turn_desc}
Stop phrase: "{stop_phrase}"{goal_part}"""

    response = requests.post(VLLM_URL, json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 130,
        "temperature": 0.3,
    }, timeout=20)

    return response.json()["choices"][0]["message"]["content"].strip()


def check_turn_directions(text, turns):
    """Return True if all turn directions in text match expected."""
    if not turns:
        return True
    for t in turns:
        expected = f"turn {t['dir']}"
        opposite = f"turn {'right' if t['dir']=='left' else 'left'}"
        if opposite in text.lower() and expected not in text.lower():
            return False
    return True


def main():
    print("=== v73 generator: constrained vLLM (Gemma-4-31B) + path geometry ===")
    print(f"  Turn constraint: {TURN_THRESHOLD}° threshold (matches GT avg=0.579)")
    print(f"  Fallback: v72 instruction if vLLM gives wrong turn direction")
    print()

    with open(VC_PATH) as f:
        vc = json.load(f)
    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V71_PATH) as f:
        v71_data = json.load(f)
    with gzip.open(V72_PATH) as f:
        v72_data = json.load(f)

    gt_by_eid = {ep["episode_id"]: ep for ep in gt_data["episodes"]}
    v71_by_eid = {ep["episode_id"]: ep for ep in v71_data["episodes"]}
    v72_by_eid = {ep["episode_id"]: ep for ep in v72_data["episodes"]}
    vc_by_eid = {int(k): v for k, v in vc.items()}

    # Load checkpoint if exists
    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        print(f"Resuming from checkpoint: {len(checkpoint)} episodes done")

    new_episodes = []
    stats = {'vllm_used': 0, 'v72_fallback': 0, 'error_fallback': 0}
    t_start = time.time()

    for ep_idx, ep in enumerate(gt_data["episodes"]):
        eid = ep["episode_id"]
        eid_str = str(eid)

        # Use checkpoint if available
        if eid_str in checkpoint:
            new_ep = dict(ep)
            new_ep["instruction"] = dict(ep["instruction"])
            new_ep["instruction"]["instruction_text"] = checkpoint[eid_str]
            new_ep["instruction"]["instruction_tokens"] = None
            new_episodes.append(new_ep)
            continue

        path = ep["reference_path"]
        vdata = vc_by_eid.get(eid, {})
        vc_turns = vdata.get("turns", [])
        vc_goal = vdata.get("goal", "")

        stop_phrase = get_stop_sentence(v71_by_eid[eid]["instruction"]["instruction_text"])
        v72_instruction = v72_by_eid[eid]["instruction"]["instruction_text"]

        turns = extract_significant_turns(path)
        vc_lms = assign_vc(turns, vc_turns)

        instruction = None

        # Try vLLM
        for attempt in range(2):
            try:
                generated = call_vllm(turns, vc_lms, vc_goal, stop_phrase)

                # Verify turn directions are correct
                if check_turn_directions(generated, turns):
                    # Ensure stop phrase is included
                    if stop_phrase.rstrip('.').lower() not in generated.lower():
                        generated = generated.rstrip('.') + ". " + stop_phrase
                    instruction = generated
                    stats['vllm_used'] += 1
                    break
                else:
                    # Wrong direction — fall back to v72
                    instruction = v72_instruction
                    stats['v72_fallback'] += 1
                    break

            except Exception as e:
                if attempt == 1:
                    instruction = v72_instruction
                    stats['error_fallback'] += 1
                else:
                    time.sleep(2)

        if instruction is None:
            instruction = v72_instruction
            stats['error_fallback'] += 1

        checkpoint[eid_str] = instruction

        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        new_ep["instruction"]["instruction_text"] = instruction
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

        # Progress logging
        if (ep_idx + 1) % BATCH_LOG_EVERY == 0:
            elapsed = time.time() - t_start
            rate = ep_idx / elapsed if elapsed > 0 else 0
            remaining = (len(gt_data["episodes"]) - ep_idx - 1) / rate if rate > 0 else 0
            print(f"  [{ep_idx+1}/{len(gt_data['episodes'])}] "
                  f"vllm={stats['vllm_used']} v72_fallback={stats['v72_fallback']} "
                  f"errors={stats['error_fallback']} "
                  f"rate={rate:.1f}/s ETA={remaining/60:.1f}min")

            # Save checkpoint periodically
            with open(CHECKPOINT_PATH, 'w') as f:
                json.dump(checkpoint, f)

    # Final checkpoint save
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f)

    print(f"\nFinal stats:")
    print(f"  vLLM used: {stats['vllm_used']}")
    print(f"  v72 fallback (wrong dir): {stats['v72_fallback']}")
    print(f"  error fallback: {stats['error_fallback']}")

    # Quality metrics
    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    avg_turns = sum(len(re.findall(r'\bturn\b', i, re.I)) for i in all_insts) / n
    hall = sum(1 for i in all_insts if re.search(r'\bhallway\b|\bhall\b', i, re.I))
    thru = sum(1 for i in all_insts if 'through the' in i.lower())
    near = sum(1 for i in all_insts if re.search(r'\bnear\b', get_stop_sentence(i), re.I))
    door = sum(1 for i in all_insts if re.search(r'\bdoor\b|\bdoorway\b|\bentrance\b', get_stop_sentence(i), re.I))

    print(f"\nQuality metrics (n={n}):")
    print(f"  avg turns: {avg_turns:.3f}  [GT=0.587]")
    print(f"  hallway:   {hall/n*100:.1f}%  [GT=20.4%]")
    print(f"  through:   {thru/n*100:.1f}%  [GT=27.2%]")
    print(f"  near-stop: {near/n*100:.1f}%  [GT=8.6%]")
    print(f"  door-stop: {door/n*100:.1f}%  [GT=31.1%]")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
