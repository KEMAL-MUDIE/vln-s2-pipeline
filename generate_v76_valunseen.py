#!/usr/bin/env python3
"""
Generate v76: length-calibrated instructions targeting GT avg=26.8 words.

Key finding: v75 avg_words=13.5 (HALF of GT=26.8, v61hpp=26.1).
Model trained on ~27 word instructions; shorter inputs likely cause navigation failures.

v76 approach:
  0-turn episodes (1047, 56.9%): multi-sentence template based on path length
    - Short (<6m):    1 sentence  ~12 words
    - Medium (6-10m): 2 sentences ~18 words
    - Long (>10m):    3 sentences ~24 words
    All use: "Walk forward [context] toward the [vc_goal]. [optional bridge]. [v71_stop]"
  Turn episodes (792, 43.1%): Gemma with length guidance (target 25-30 words, 3-4 sentences)
    System: "3-4 natural sentences, about 25-30 words total"
    This pushes Gemma to write longer, richer instructions

Expected stats:
  avg_words: ~22-26 (vs v75=13.5, GT=26.8)
  hallway: ~5-8% (Gemma may naturally add hallway in longer text)
  through: ~3-8% (Gemma longer text more likely to add "through")
  avg_turns: 0.579 (same geometry)
  near: 8.2% (same v71 stops)
"""
import gzip
import json
import math
import re
import time
from pathlib import Path

import requests

BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")
VC_PATH = Path("/home/kemal/VLNav/s2_pipeline_new/outputs/gate3_gemma_v22_vision_checkpoint.json")
GT_PATH = BASE / "val_unseen" / "val_unseen.json.gz"
V71_PATH = BASE / "val_unseen" / "val_unseen_v71.json.gz"
V74_PATH = BASE / "val_unseen" / "val_unseen_v74.json.gz"
OUT_PATH = BASE / "val_unseen" / "val_unseen_v76.json.gz"
CHECKPOINT_PATH = Path("/home/kemal/VLNav/s2_pipeline_new/outputs/v76_checkpoint.json")

VLLM_URL = "http://10.77.32.231:8000/v1/chat/completions"
MODEL = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
TURN_THRESHOLD = 80
BATCH_LOG_EVERY = 50

# Key change: target 3-4 sentences, ~25-30 words to match GT distribution
SYSTEM_MSG = """You write indoor navigation instructions for a robot.
Rules:
1. Start with walking forward. Never start with a turn.
2. Include ONLY the listed turns, in order.
3. End with the EXACT stop phrase given.
4. Write 3-4 natural sentences totaling about 25-30 words. Be specific but concise."""

COLOR_STRIP = re.compile(
    r'\b(white|grey|gray|brown|black|dark|light|beige|cream|tan|upholstered|'
    r'rectangular|circular|oval|square|decorative|ornate|modern|antique|'
    r'large|small|tall|short|long|wide|narrow|big|little|wooden|metal|glass|'
    r'fabric|cloth|leather|stone|marble|polished)\b\s*',
    re.IGNORECASE
)

STOP_VERBS = re.compile(r'\b(stop|wait|halt|pause|stand)\b', re.IGNORECASE)
TRAILING_JUNK = re.compile(r'[\s,;.]*(and then|then|and)\s*$', re.IGNORECASE)


def clean_lm(lm):
    if not lm:
        return ""
    lm = COLOR_STRIP.sub('', lm).strip()
    lm = re.sub(r'\s+', ' ', lm).strip()
    return lm.lower() if len(lm) > 3 else ""


def path_length(path):
    total = 0.0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dz = path[i][2] - path[i-1][2]
        total += math.sqrt(dx*dx + dz*dz)
    return total


def get_stop_sentence(inst):
    sents = re.split(r'(?<=[.!?])\s+', inst.strip())
    return sents[-1].strip() if len(sents) > 1 else inst.strip()


def replace_stop_phrase(v_inst, v71_stop):
    matches = list(STOP_VERBS.finditer(v_inst))
    if not matches:
        return v_inst.rstrip('.') + ". " + v71_stop
    last_match = matches[-1]
    nav_body = v_inst[:last_match.start()].rstrip()
    nav_body = TRAILING_JUNK.sub('', nav_body).rstrip('.,;:')
    if nav_body:
        return nav_body + ". " + v71_stop
    else:
        return v71_stop


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


def build_0turn_instruction(vc_goal, v71_stop, plen):
    """Length-calibrated template for 0-turn episodes."""
    goal_lm = clean_lm(vc_goal)
    if not goal_lm:
        goal_lm = "the destination"

    if plen < 6.0:
        # Short path: 1 sentence + stop (~12w)
        return f"Walk forward toward the {goal_lm}. {v71_stop}"
    elif plen < 10.0:
        # Medium path: 2 sentences + stop (~18w)
        return f"Walk straight ahead toward the {goal_lm}. Continue forward until you reach it. {v71_stop}"
    elif plen < 15.0:
        # Long path: 2 sentences + stop (~22w)
        return f"Walk straight ahead and continue forward past the area, making your way toward the {goal_lm}. Keep going straight until you get there. {v71_stop}"
    else:
        # Very long path: 3 sentences + stop (~26w)
        return f"Walk straight ahead and continue forward through the space. Keep going past the open area, heading toward the {goal_lm} at the far end. {v71_stop}"


def call_vllm_turn(turns, vc_lms, vc_goal, stop_phrase):
    goal_lm = clean_lm(vc_goal)
    clean_lms = [clean_lm(lm) for lm in vc_lms]

    items = []
    for t, lm in zip(turns, clean_lms):
        lm_part = f" near {lm}" if lm else ""
        items.append(f"turn {t['dir']}{lm_part}")
    turn_desc = " then ".join(items)

    goal_part = f"\nGoal: {goal_lm}" if goal_lm else ""
    user_msg = f"""Turns: {turn_desc}
Stop phrase: "{stop_phrase}"{goal_part}"""

    response = requests.post(VLLM_URL, json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 150,
        "temperature": 0.3,
    }, timeout=20)

    return response.json()["choices"][0]["message"]["content"].strip()


def check_turn_directions(text, turns):
    if not turns:
        return True
    for t in turns:
        expected = f"turn {t['dir']}"
        opposite = f"turn {'right' if t['dir']=='left' else 'left'}"
        if opposite in text.lower() and expected not in text.lower():
            return False
    return True


def main():
    print("=== v76 generator: length-calibrated (target ~25 words to match GT) ===")
    print(f"  0-turn: multi-sentence template based on path length (short/med/long)")
    print(f"  Turn: Gemma with '3-4 sentences, ~25-30 words' guidance")
    print(f"  GT avg=26.8w, v61hpp=26.1w, v75=13.5w (too short!)")
    print()

    with open(VC_PATH) as f:
        vc = json.load(f)
    with gzip.open(GT_PATH) as f:
        gt_data = json.load(f)
    with gzip.open(V71_PATH) as f:
        v71_data = json.load(f)
    with gzip.open(V74_PATH) as f:
        v74_data = json.load(f)

    v71_by_eid = {ep["episode_id"]: ep for ep in v71_data["episodes"]}
    v74_by_eid = {ep["episode_id"]: ep for ep in v74_data["episodes"]}
    vc_by_eid = {int(k): v for k, v in vc.items()}

    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        print(f"Resuming from checkpoint: {len(checkpoint)} episodes done")

    new_episodes = []
    stats = {'template': 0, 'vllm_used': 0, 'v74_fallback': 0, 'error_fallback': 0}
    t_start = time.time()

    for ep_idx, ep in enumerate(gt_data["episodes"]):
        eid = ep["episode_id"]
        eid_str = str(eid)

        if eid_str in checkpoint:
            new_ep = dict(ep)
            new_ep["instruction"] = dict(ep["instruction"])
            new_ep["instruction"]["instruction_text"] = checkpoint[eid_str]
            new_ep["instruction"]["instruction_tokens"] = None
            new_episodes.append(new_ep)
            continue

        path = ep["reference_path"]
        plen = path_length(path)
        vdata = vc_by_eid.get(eid, {})
        vc_turns_raw = vdata.get("turns", [])
        vc_goal = vdata.get("goal", "")

        v71_stop = get_stop_sentence(v71_by_eid[eid]["instruction"]["instruction_text"])
        v74_instruction = v74_by_eid[eid]["instruction"]["instruction_text"]

        turns = extract_significant_turns(path)

        if not turns:
            instruction = build_0turn_instruction(vc_goal, v71_stop, plen)
            stats['template'] += 1
        else:
            vc_lms = assign_vc(turns, vc_turns_raw)
            instruction = None

            for attempt in range(2):
                try:
                    generated = call_vllm_turn(turns, vc_lms, vc_goal, v71_stop)
                    if check_turn_directions(generated, turns):
                        instruction = replace_stop_phrase(generated, v71_stop)
                        stats['vllm_used'] += 1
                        break
                    else:
                        instruction = v74_instruction
                        stats['v74_fallback'] += 1
                        break
                except Exception:
                    if attempt == 1:
                        instruction = v74_instruction
                        stats['error_fallback'] += 1
                    else:
                        time.sleep(2)

            if instruction is None:
                instruction = v74_instruction
                stats['error_fallback'] += 1

        checkpoint[eid_str] = instruction

        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        new_ep["instruction"]["instruction_text"] = instruction
        new_ep["instruction"]["instruction_tokens"] = None
        new_episodes.append(new_ep)

        if (ep_idx + 1) % BATCH_LOG_EVERY == 0:
            elapsed = time.time() - t_start
            rate = ep_idx / elapsed if elapsed > 0 else 0
            remaining = (len(gt_data["episodes"]) - ep_idx - 1) / rate if rate > 0 else 0
            print(f"  [{ep_idx+1}/{len(gt_data['episodes'])}] "
                  f"template={stats['template']} vllm={stats['vllm_used']} "
                  f"v74fb={stats['v74_fallback']} errors={stats['error_fallback']} "
                  f"rate={rate:.1f}/s ETA={remaining/60:.1f}min")
            with open(CHECKPOINT_PATH, 'w') as f:
                json.dump(checkpoint, f)

    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f)

    print(f"\nFinal stats:")
    print(f"  template (0-turn): {stats['template']}")
    print(f"  vLLM (turn eps): {stats['vllm_used']}")
    print(f"  v74 fallback: {stats['v74_fallback']}")
    print(f"  error fallback: {stats['error_fallback']}")

    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]

    def get_stop_sent(inst):
        sents = re.split(r'(?<=[.!?])\s+', inst.strip())
        return sents[-1].strip() if len(sents) > 1 else inst.strip()

    avg_turns = sum(len(re.findall(r'\bturn\b', i, re.I)) for i in all_insts) / n
    hall = sum(1 for i in all_insts if re.search(r'\bhallway\b|\bhall\b', i, re.I))
    thru = sum(1 for i in all_insts if 'through the' in i.lower())
    near = sum(1 for i in all_insts if re.search(r'\bnear\b', get_stop_sent(i), re.I))
    door = sum(1 for i in all_insts if re.search(r'\bdoor\b|\bdoorway\b|\bentrance\b', get_stop_sent(i), re.I))

    print(f"\nQuality metrics (n={n}):")
    print(f"  avg turns: {avg_turns:.3f}  [GT=0.587]")
    print(f"  avg words: {sum(words)/n:.1f}  [GT=26.8, v75=13.5]")
    print(f"  hallway:   {hall/n*100:.1f}%  [GT=20.4%]")
    print(f"  through:   {thru/n*100:.1f}%  [GT=27.2%]")
    print(f"  near-stop: {near/n*100:.1f}%  [GT=8.6%]")
    print(f"  door-stop: {door/n*100:.1f}%  [GT=31.1%]")
    print(f"  unique:    {len(set(all_insts))/n*100:.1f}%")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
