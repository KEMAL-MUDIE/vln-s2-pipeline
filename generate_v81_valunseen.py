#!/usr/bin/env python3
"""
Generate v81: vc_goal anchor + neutral 'through the area' spatial recovery.

v80 gap:    through=3.5% (GT=27.2%) — Gemma compact body doesn't generate spatial traversal language
v78 error:  through=35.9% (GT=27.2%) — "through the kitchen" examples triggered room name fabrication

v81 fix: allow NEUTRAL spatial terms only ("through the area", "past the area") — no room names.
  Turn Gemma system: "You may say 'through the area' or 'past the area'. Never say 'hallway'."
  This recovers spatial traversal pattern WITHOUT fabricating kitchen/bedroom/living room names.

Expected:
  through: ~15-25% (Gemma uses neutral terms selectively, not always)
  hallway: <5% (ban maintained)
  avg_words: ~25-28 (2-sentence body + vc_goal + stop)
  toward: ~99% (vc_goal anchored for all turn eps)

GT:  through=27.2%, hallway=20.4%, avg_words=26.8
v80: through=3.5%, hallway=4.9%, avg_words=24.6, toward=99.5%
v78: through=35.9%, hallway=4.9%, avg_words=21.8 (room fabrication)
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
OUT_PATH = BASE / "val_unseen" / "val_unseen_v81.json.gz"
CHECKPOINT_PATH = Path("/home/kemal/VLNav/s2_pipeline_new/outputs/v81_checkpoint.json")

VLLM_URL = "http://10.77.32.231:8000/v1/chat/completions"
MODEL = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
TURN_THRESHOLD = 80
BATCH_LOG_EVERY = 50

# Key: allow neutral spatial terms but NOT room names or "hallway"
SYSTEM_MSG = """You write navigation body instructions for a robot.
Rules:
1. Start by walking forward. Never start with a turn.
2. Include ONLY the listed turns, in order.
3. Do NOT write any stop phrase — it is added later.
4. Write exactly 2 sentences totaling 15-20 words.
5. You may say "through the area" or "past the area" for traversal. Never say "hallway"."""

COLOR_STRIP = re.compile(
    r'\b(white|grey|gray|brown|black|dark|light|beige|cream|tan|upholstered|'
    r'rectangular|circular|oval|square|decorative|ornate|modern|antique|'
    r'large|small|tall|short|long|wide|narrow|big|little|wooden|metal|glass|'
    r'fabric|cloth|leather|stone|marble|polished)\b\s*',
    re.IGNORECASE
)

STOP_VERBS = re.compile(r'\b(stop|wait|halt|pause|stand)\b', re.IGNORECASE)
TRAILING_JUNK = re.compile(r'[\s,;.]*(and then|then|and)\s*$', re.IGNORECASE)
HALLWAY_RE = re.compile(r'\b(hallway|hall)\b', re.IGNORECASE)
ROOM_NAMES = re.compile(
    r'\b(kitchen|bedroom|bathroom|living room|dining room|office|study|library|'
    r'garage|basement|attic|foyer|entryway|lobby|corridor|passage|vestibule)\b',
    re.IGNORECASE
)


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


def strip_stop_from_body(body):
    matches = list(STOP_VERBS.finditer(body))
    if not matches:
        return body.rstrip('.,;: ')
    last_match = matches[-1]
    nav = body[:last_match.start()].rstrip()
    nav = TRAILING_JUNK.sub('', nav).rstrip('.,;:')
    return nav if nav else body.rstrip('.,;: ')


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
    goal_lm = clean_lm(vc_goal)
    if not goal_lm:
        goal_lm = "the destination"
    if plen < 6.0:
        return f"Walk forward toward the {goal_lm}. {v71_stop}"
    elif plen < 10.0:
        return f"Walk straight ahead toward the {goal_lm}. Continue forward until you reach it. {v71_stop}"
    elif plen < 15.0:
        return f"Walk straight ahead and continue forward past the area, making your way toward the {goal_lm}. Keep going straight until you get there. {v71_stop}"
    else:
        return f"Walk straight ahead and continue forward through the area. Keep going, heading toward the {goal_lm} at the far end. {v71_stop}"


def call_vllm_turn_body(turns, vc_lms):
    clean_lms = [clean_lm(lm) for lm in vc_lms]
    items = []
    for t, lm in zip(turns, clean_lms):
        lm_part = f" near {lm}" if lm else ""
        items.append(f"turn {t['dir']}{lm_part}")
    turn_desc = " then ".join(items)
    user_msg = f"Turns: {turn_desc}"

    response = requests.post(VLLM_URL, json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 110,
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


def has_bad_content(text):
    return bool(HALLWAY_RE.search(text)) or bool(ROOM_NAMES.search(text))


def build_turn_instruction(turns, vc_lms, vc_goal, v71_stop, v74_fallback):
    goal_lm = clean_lm(vc_goal)

    for attempt in range(3):
        try:
            body = call_vllm_turn_body(turns, vc_lms)
            if not check_turn_directions(body, turns):
                return v74_fallback, 'dir_fallback'
            if has_bad_content(body) and attempt < 2:
                continue  # retry on hallway or room names
            nav_body = strip_stop_from_body(body)
            if not nav_body:
                nav_body = "Walk forward and then " + " and ".join(
                    f"turn {t['dir']}" for t in turns)
            if goal_lm:
                instruction = f"{nav_body}. Continue toward the {goal_lm}. {v71_stop}"
            else:
                instruction = f"{nav_body}. {v71_stop}"
            return instruction, 'vllm'
        except Exception:
            if attempt == 2:
                return v74_fallback, 'error'
            time.sleep(2)

    return v74_fallback, 'error'


def main():
    print("=== v81 generator: vc_goal anchor + neutral 'through the area' recovery ===")
    print(f"  v80: through=3.5% (GT=27.2%). v81 allows 'through the area'/'past the area'.")
    print(f"  Bans hallway AND specific room names (kitchen/bedroom/etc.) — neutral only.")
    print(f"  Expected through~15-25%, hallway<5%, avg_words~25-28, toward~99%")
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
    stats = {'template': 0, 'vllm_used': 0, 'goal_anchored': 0,
             'bad_retry': 0, 'dir_fallback': 0, 'error_fallback': 0}
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
            instruction, mode = build_turn_instruction(
                turns, vc_lms, vc_goal, v71_stop, v74_instruction)
            if mode == 'vllm':
                stats['vllm_used'] += 1
                if clean_lm(vc_goal):
                    stats['goal_anchored'] += 1
            elif mode == 'dir_fallback':
                stats['dir_fallback'] += 1
            else:
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
                  f"goal_anch={stats['goal_anchored']} "
                  f"dirfb={stats['dir_fallback']} errs={stats['error_fallback']} "
                  f"rate={rate:.1f}/s ETA={remaining/60:.1f}min")
            with open(CHECKPOINT_PATH, 'w') as f:
                json.dump(checkpoint, f)

    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f)

    print(f"\nFinal stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    n = len(new_episodes)
    all_insts = [ep["instruction"]["instruction_text"] for ep in new_episodes]
    words = [len(i.split()) for i in all_insts]

    def get_stop_sent(inst):
        sents = re.split(r'(?<=[.!?])\s+', inst.strip())
        return sents[-1].strip() if len(sents) > 1 else inst.strip()

    avg_turns_kw = sum(len(re.findall(r'\bturn\b', i, re.I)) for i in all_insts) / n
    hall = sum(1 for i in all_insts if re.search(r'\bhallway\b|\bhall\b', i, re.I))
    thru = sum(1 for i in all_insts if 'through the' in i.lower())
    toward = sum(1 for i in all_insts if 'toward' in i.lower())
    near = sum(1 for i in all_insts if re.search(r'\bnear\b', get_stop_sent(i), re.I))
    rooms = sum(1 for i in all_insts if ROOM_NAMES.search(i))

    print(f"\nQuality metrics (n={n}):")
    print(f"  avg turns: {avg_turns_kw:.3f}  [GT=0.587]")
    print(f"  avg words: {sum(words)/n:.1f}  [GT=26.8, v80=24.6, v81 target=25-28]")
    print(f"  hallway:   {hall/n*100:.1f}%  [GT=20.4%, v80=4.9%]")
    print(f"  room names:{rooms/n*100:.1f}%  [should be ~0%]")
    print(f"  through:   {thru/n*100:.1f}%  [GT=27.2%, v80=3.5%, v78=35.9%]")
    print(f"  toward:    {toward/n*100:.1f}%  [v80=99.5%]")
    print(f"  near-stop: {near/n*100:.1f}%  [GT=8.6%]")
    print(f"  unique:    {len(set(all_insts))/n*100:.1f}%")

    out_data = dict(gt_data)
    out_data["episodes"] = new_episodes
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(out_data, f)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
