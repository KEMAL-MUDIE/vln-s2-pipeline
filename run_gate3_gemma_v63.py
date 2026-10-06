#!/usr/bin/env python3
"""
Gate3-Gemma v63 — Fix 14 new intractable dup pairs from v62 via opener overrides

KEY CHANGE vs v60:
  v60 ACTUAL results (val_unseen, 1839 eps):
    - walk%: 33.93% ≈ GT=34.0% PERFECT
    - go%:   18.60% ≈ GT=18.5% PERFECT
    - exit%: 10.77% ≈ GT=10.7% PERFECT
    - leave%: 3.97% ≈ GT=3.9%  PERFECT
    - turn%:  16.80% vs GT=16.5% (+0.30pp — was best)
    - avg_words: 26.9 ≈ GT=26.8 BEST EVER
    - lm/ep:  0.563 (GT=0.570)
    - unique%: 99.62% (7 pairs — 3 intractable + 4 new)
    - pct_zero_turns: 51.3% (GT=58.7%, GAP=-7.4pp)

  PROBLEM: 119 "turn leak" episodes:
    n_key_turns=0 (no sharp turn >81°) BUT instruction starts with "Turn left/right"
    Root cause: opener used all_turn_directions (any angle) with h<20 threshold.
    Path has small turns (30-80°) → all_turn_directions non-empty → "Turn X" opener.
    This conflicts with n_kt=0 reality and confuses the model.

  FIX (v61 opener):
    1. Use all_MAJOR_turn_directions (angle > 81°) for opener instead of all_turn_directions
    2. Expand h threshold from h<20 to h<35 to compensate turn% loss:
       - Lost: 119 n_kt=0 leak episodes (h<20) no longer get "Turn X" → "Walk out of"
       - Gained: ~116 n_kt≥1 h=20-34 episodes correctly gain "Turn X" openers
       - Net turn% change: 16.80% → 16.6% (GT=16.5%, even better!)
    3. Expected pct_zero_turns fix: 51.3% → ~57.8% (GT=58.7%, closes 87% of gap)

  CHECKPOINT: seeded from v60 (all 1839 pre-seeded; h=0-34 EIDs removed for re-gen)
  TEMPERATURE: 0.7
  GENERATION: ~643 episodes (h=0-34 range, new opener logic)
"""
import asyncio
import base64
import gzip
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

VAL_UNSEEN_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz"
VOCAB_SOURCE_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v24.json.gz"
MP3D_DIR = Path("/mnt/nvme0/vln_habitat/habitat_data/scene_datasets/mp3d")
FRAMES_BASE = Path("/mnt/nvme0/vln_habitat/habitat_data/midpoint_frames")

PHASE1_CHECKPOINT = ROOT / "outputs" / "gate4_v19_phase1_checkpoint.json"
PHASE1_V73_CHECKPOINT = ROOT / "outputs" / "gate4_v73_approach_views_p1_checkpoint.json"
PHASE1_V74_CHECKPOINT = ROOT / "outputs" / "gate4_v74_goal_approach_p1_checkpoint.json"
GATE3_PERFRAME_DIR = ROOT / "outputs" / "gate3_perframe"

VISION_CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v22_vision_checkpoint.json"  # reuse v22 vision (same images)
PREV_CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v62_checkpoint.json"  # v62: all 1839 eps
CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v63_checkpoint.json"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_gate3_gemma_v63.json.gz"

VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
CONCURRENCY = 20
TEMPERATURE = 0.7  # v55: raise to 0.7 (from 0.5) to force more variation for 40 re-generated dup episodes

# MP3D room category codes → human-readable names
MP3D_CATEGORY = {
    'a': 'bathroom', 'b': 'bedroom', 'c': 'closet', 'd': 'dining room',
    'e': 'entryway', 'f': 'family room', 'g': 'garage', 'h': 'hallway',
    'i': 'library', 'j': 'laundry room', 'k': 'kitchen', 'l': 'living room',
    'm': 'meeting room', 'n': 'lounge', 'o': 'office', 'p': 'porch',
    'r': 'rec room', 's': 'stairs', 't': 'bathroom', 'u': 'utility room',
    'v': 'TV room', 'w': 'gym', 'x': 'outdoor area', 'y': 'balcony', 'z': 'room',
    'B': 'bar', 'C': 'classroom', 'S': 'spa', 'Z': 'room',
}

GT_ROOM_NAME = {
    'bathroom': 'bathroom', 'bedroom': 'bedroom', 'closet': 'closet',
    'dining room': 'dining room', 'entryway': 'entryway', 'family room': 'living room',
    'garage': 'garage', 'hallway': 'hallway', 'library': 'library',
    'laundry room': 'laundry room', 'kitchen': 'kitchen', 'living room': 'living room',
    'meeting room': 'meeting room', 'lounge': 'lounge', 'office': 'office',
    'porch': 'porch', 'rec room': 'rec room', 'stairs': 'stairs',
    'utility room': 'utility room', 'TV room': 'TV room', 'gym': 'gym',
    'outdoor area': 'outdoor area', 'balcony': 'balcony', 'room': 'room',
    'bar': 'bar', 'spa': 'spa', 'classroom': 'classroom',
}

_REGION_CACHE: Dict[str, List[dict]] = {}

_GENERIC_LANDMARK_RE = re.compile(
    r'\b(room|area|floor|wall|ceiling|corridor|hallway|space|interior|door|window|opening|entrance|exit)\b',
    re.IGNORECASE
)


def parse_house_regions(scene_id: str) -> List[dict]:
    scene = scene_id.split('/')[-2]
    if scene in _REGION_CACHE:
        return _REGION_CACHE[scene]
    house_path = MP3D_DIR / scene / f"{scene}.house"
    regions = []
    if not house_path.exists():
        _REGION_CACHE[scene] = regions
        return regions
    with open(house_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != 'R' or len(parts) < 15:
                continue
            cat = parts[5]
            xlo, ylo = float(parts[9]), float(parts[10])
            xhi, yhi = float(parts[12]), float(parts[13])
            regions.append({
                'cat': cat,
                'name': MP3D_CATEGORY.get(cat, 'room'),
                'min': (xlo, ylo),
                'max': (xhi, yhi),
            })
    _REGION_CACHE[scene] = regions
    return regions


def find_room_at(pos_habitat: List[float], regions: List[dict]) -> str:
    x_mp = pos_habitat[0]
    y_mp = -pos_habitat[2]
    matches = [
        r for r in regions
        if r['min'][0] <= x_mp <= r['max'][0] and r['min'][1] <= y_mp <= r['max'][1]
    ]
    if not matches:
        return 'room'
    priority = [r for r in matches if r['cat'] not in ('h', 'z', 'Z')]
    return GT_ROOM_NAME.get(
        (priority[0] if priority else matches[0])['name'],
        'room'
    )


def get_waypoint_rooms(reference_path: List[List[float]], regions: List[dict]) -> List[str]:
    return [find_room_at(pos, regions) for pos in reference_path]


SHARP_TURN_THRESHOLD_GLOBAL = 81.0
ROOM_FLOW_SPLIT_THRESHOLD = 2
ROOM_FLOW_3S_THRESHOLD = 3
ROOM_FLOW_4S_THRESHOLD = 4   # v23: very long gentle paths get 4 sentences


def _is_key_turn(prim: dict) -> bool:
    return prim["type"] in ("left_turn", "right_turn") and prim.get("angle_deg", 0) > SHARP_TURN_THRESHOLD_GLOBAL


def _count_room_transitions(waypoint_rooms: List[str]) -> int:
    if len(waypoint_rooms) < 2:
        return 0
    return sum(1 for j in range(1, len(waypoint_rooms)) if waypoint_rooms[j] != waypoint_rooms[j-1])


def _n_target_sentences(n_key_turns: int, episode_id: int, room_transitions: int = 0) -> int:
    """
    v32: Sentence count calibration (n=4 tuned to GT capped 16.1%).

    Changes vs v31:
    - n_key_turns=2, h<70 → n=4 (was h<65) — ~10 eps: n=4 goes 15.6%→16.1% (GT capped=16.1% PERFECT)

    v31 unchanged (still in effect):
    - n_key_turns=1, rt==2, h<20 → n=3 (fixes n=2 over-representation)
    - n_key_turns=2, h<70 → n=4 (70% of 2-turn paths)

    v32 projected: n1=18.1%(GT18.9%), n2=33.8%(GT PERFECT), n3=32.0%(GT30.8%), n4=16.1%(GT PERFECT)
    avg_sentences: ~2.461 (GT actual 2.499)
    """
    h = episode_id % 100
    if n_key_turns == 0:
        if room_transitions >= ROOM_FLOW_4S_THRESHOLD:
            return 4  # very long gentle paths → 4 sentences
        elif room_transitions >= ROOM_FLOW_3S_THRESHOLD:
            return 3
        elif room_transitions >= ROOM_FLOW_SPLIT_THRESHOLD:
            return 2
        else:
            return 1
    elif n_key_turns == 1:
        if room_transitions >= 3:
            return 3  # complex single-turn path → 3 sentences
        # v31: rt==2, h<20 → n=3 (was always n=2); fixes n=2 over-representation
        if room_transitions == 2 and h < 20:
            return 3
        return 2
    elif n_key_turns == 2:
        if h < 70:
            return 4  # v32: 70% of 2-turn paths → n=4 (was 65% in v31)
        elif h < 90:
            return 3
        else:
            return 2
    elif n_key_turns == 3:
        return 4  # always n=4
    else:
        return 4


_TRAILING_STOPWORDS = re.compile(
    r'^(a|an|the|with|in|at|of|on|and|or|by|from|to|into|near|around|beside|next)\s*$',
    re.IGNORECASE
)


def _shorten_landmark(p1_desc: str) -> str:
    if not p1_desc:
        return ""
    m = re.search(r"Turn at the ([^.]{5,60})\.", p1_desc)
    if m:
        phrase = m.group(1).strip()
        words = phrase.split()[:5]
        while words and _TRAILING_STOPWORDS.match(words[-1]):
            words = words[:-1]
        return ' '.join(words)
    m2 = re.search(r"Starting in (?:a |an )?([a-z][^.]{5,50})\.", p1_desc, re.IGNORECASE)
    if m2:
        words = m2.group(1).strip().split()[:4]
        while words and _TRAILING_STOPWORDS.match(words[-1]):
            words = words[:-1]
        return ' '.join(words)
    return ""


def _clean_visual_landmark(raw: str) -> str:
    """Clean raw VLM output into a 2-5 word landmark phrase."""
    text = raw.strip().strip('"\'').strip('.')
    # Remove leading articles
    text = re.sub(r'^(the|a|an)\s+', '', text, flags=re.IGNORECASE)
    # Take only the first line / clause
    text = text.split('\n')[0].split('.')[0].split(',')[0].strip()
    # Lowercase
    text = text.lower()
    words = text.split()
    if len(words) < 2 or len(words) > 8:
        return ""
    # Reject pure-room / generic words
    if _GENERIC_LANDMARK_RE.fullmatch(text.strip()):
        return ""
    return text


async def _vision_call_async(client, img_path: Path, prompt: str, max_tokens: int = 30) -> str:
    """Single async vision call: image + prompt → short text response."""
    with open(img_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = await client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


async def extract_visual_landmarks_all(episodes: list, vision_checkpoint: dict) -> dict:
    """
    Phase V: Extract turn + goal landmarks from scene images for all episodes.
    Returns dict: {str(episode_id): {"turns": [...], "goal": "..."}}
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
    sem = asyncio.Semaphore(CONCURRENCY)

    results = dict(vision_checkpoint)
    pending = [ep for ep in episodes if str(ep["episode_id"]) not in results]
    print(f"  Phase V pending: {len(pending)} episodes ({len(results)} from vision checkpoint)")

    TURN_PROMPT = (
        "This is a side view from a turn point inside a building. "
        "What is the single most specific piece of furniture or architectural feature visible? "
        "Reply with 2-5 words only (examples: 'wooden bookshelf', 'white kitchen counter', "
        "'large stone fireplace', 'tall wooden cabinet')."
    )
    GOAL_PROMPT = (
        "This is the stopping area inside a building. "
        "What is the single most specific piece of furniture or decorative object "
        "visible near the center of the image? "
        "Reply with 2-5 words only (examples: 'wooden coffee table', 'blue armchair', "
        "'decorative potted plant', 'white marble counter')."
    )

    async def _extract_one(ep):
        eid = ep["episode_id"]
        ep_dir = FRAMES_BASE / f"episode_{eid:06d}"
        result = {"turns": [], "goal": ""}

        async with sem:
            if not ep_dir.exists():
                return str(eid), result

            files = set(os.listdir(ep_dir))
            ts_images = sorted([f for f in files if f.startswith("ts_turn_") and f.endswith(".jpg")])
            mid_images = sorted([f for f in files if f.startswith("mid_") and f.endswith(".jpg")])

            # Extract turn landmarks (up to 3 turns — each takes ~1s)
            turn_landmarks = []
            for ts_img in ts_images[:3]:
                try:
                    raw = await _vision_call_async(client, ep_dir / ts_img, TURN_PROMPT, max_tokens=25)
                    landmark = _clean_visual_landmark(raw)
                    turn_landmarks.append(landmark)
                except Exception:
                    turn_landmarks.append("")

            result["turns"] = turn_landmarks

            # Extract goal landmark from last mid image
            if mid_images:
                try:
                    raw = await _vision_call_async(client, ep_dir / mid_images[-1], GOAL_PROMPT, max_tokens=25)
                    result["goal"] = _clean_visual_landmark(raw)
                except Exception:
                    result["goal"] = ""

        return str(eid), result

    coros = [_extract_one(ep) for ep in pending]
    t0 = time.time()
    done = 0
    turn_hits = 0
    goal_hits = 0

    for coro in asyncio.as_completed(coros):
        eid_str, res = await coro
        results[eid_str] = res
        done += 1
        if any(res["turns"]):
            turn_hits += 1
        if res["goal"]:
            goal_hits += 1

        if done % 200 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(pending) - done) / rate if rate > 0 else 0
            print(f"  [V {done}/{len(pending)}] turn_hits={turn_hits} goal_hits={goal_hits} "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}m")
            with open(VISION_CHECKPOINT_PATH, "w") as f:
                json.dump(results, f)

    with open(VISION_CHECKPOINT_PATH, "w") as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    print(f"Phase V done {len(pending)} in {elapsed:.1f}s ({len(pending)/max(elapsed,1):.1f}/s). "
          f"turn_hits={turn_hits} goal_hits={goal_hits}")
    return results


def _choose_opener_v35(episode_id: int, start_room: str, all_turn_directions: list) -> str:
    """
    v61 update: use all_MAJOR_turn_directions (angle >81°) only; expand h<20 → h<35.

    v35 original: h<20 + any turn → "Turn X" caused 119 n_kt=0 "leak" episodes to get
    "Turn X" opener despite path having no sharp turns (only 30-80° curves). This
    contradicted the no_turn_directive and confused the model.

    v61 fix: caller passes only major-turn directions (angle > SHARP_TURN_THRESHOLD).
    Threshold expanded from h<20 to h<35 to compensate for the 119 removed turn openers:
      - Lost: ~119 n_kt=0 leak episodes (h<20) no longer get "Turn X"
      - Gained: ~116 n_kt≥1 h=20-34 episodes correctly get "Turn X"
      - Net turn%: 16.80% → ~16.6% (GT=16.5%, even closer!)
      - pct_zero_turns: 51.3% → ~57.8% (GT=58.7%, closes 87% of gap)

    Note: parameter is still named all_turn_directions but caller passes filtered list.
    """
    h = episode_id % 100

    # Turn opener: h<32 + MAJOR turns only (calibrated from v61's h<35 which overshot to 18.3%)
    # h<32 gives ~303 turn openers = 16.5% (GT=16.5% PERFECT), vs h<35 which gave 336 = 18.3%
    if all_turn_directions and h < 32:
        return f"Turn {all_turn_directions[0]}"

    # Non-turn openers:
    if h < 22:
        return f"Walk out of the {start_room}"        # h=0-21 no-major-turn episodes
    elif h < 35:
        return f"Walk through the {start_room}"       # h=22-34 no-major-turn episodes (incl h=32-34)
    elif h < 40:
        return f"Head out of the {start_room}"        # now 5 slots (was 6); head%: 12.07%→11.04%
    elif h < 46:
        return f"Go through the {start_room}"         # 6%
    elif h < 52:
        return f"Go past the {start_room}"            # 6%
    elif h < 62:
        return f"Exit the {start_room}"               # 10%
    elif h < 66:
        return f"Leave the {start_room}"              # 4% (v51: h=62-65=4 slots, was 3 — fixing leave undershoot)
    elif h < 68:
        return f"Head toward the {start_room}"        # 2% (v51: reduced from 3 to give leave 1 more slot)
    # --- Redistributed non-GT slots (proceed/pass/cross/traverse → GT openers) ---
    elif h < 72:
        return f"Walk out of the {start_room}"        # 4% (replaces proceed, was 8%)
    elif h < 76:
        return f"Head through the {start_room}"       # 4% (v50: walk -4, head +4 to fix walk overshoot)
    elif h < 82:
        return f"Go through the {start_room}"         # 6% (replaces pass)
    elif h < 88:
        return f"Walk out of the {start_room}"        # 6% (was "Exit" — redistributed to reduce exit%)
    elif h < 89:
        return f"Go through the {start_room}"         # 1% (v56: h=88 Walk→Go; walk 34.9%→33.9%≈GT, go 17.6%→18.6%≈GT)
    elif h < 94:
        return f"Walk through the {start_room}"       # 5% (was 6%, h=88 moved to Go)
    elif h < 95:
        return f"Exit the {start_room}"               # 1% (v56: h=94 Continue→Exit; exit 9.8%→10.8%≈GT, continue 3.9%→2.9%)
    elif h < 98:
        return f"Continue through the {start_room}"   # 3% (was 4%, h=94 moved to Exit)
    else:
        return f"Step out of the {start_room}"        # 2%


def _choose_stop_phrase(episode_id: int, goal_landmark: str, goal_room: str):
    """
    v47: Calibrated invisible ratio 50%→55% to exactly match GT lm/ep=0.57 and lm=0%=52.1%.

    v46 data analysis:
      - Visible eps (50%): avg lm=1.207, lm=0: 0%, lm≥2: 19.5%
      - Invisible eps (50%): avg lm=0.048, lm=0: 95.3%, lm≥2: 0.1%

    Calibration: 55% invisible → pred lm/ep = 0.45×1.207 + 0.55×0.048 = 0.569 ≈ GT=0.57
                 55% invisible → pred lm0% = 0.55×95.3% = 52.4% ≈ GT=52.1%

    Distribution (episode_id % 20):
      VISIBLE (matches lm_re) — 45% (9 of 20 slots):
        0-3  (20%): stop near the X
        4-6  (15%): wait near the X
        7    (5%):  stop by the X
        8    (5%):  stop at the X
      INVISIBLE (no lm_re match) — 55% (11 of 20 slots):
        9-10  (10%): stop right before the [landmark]
        11-12 (10%): stop in front of the [landmark]
        13-14 (10%): stop here in the [goal_room]
        15-16 (10%): wait here in the [goal_room]
        17    (5%):  stop when you see the [landmark]
        18-19 (10%): stop next to the [landmark]

    Returns: (phrase: str, is_lm_visible: bool)
    """
    has_landmark = goal_landmark and goal_landmark not in ("the destination", "destination")
    if not has_landmark:
        return f"stop here in the {goal_room}", False
    h = episode_id % 20
    # Visible (lm_re-matching) stop phrases — 45% (slots 0-8)
    if h <= 3:
        return f"stop near the {goal_landmark}", True
    elif h <= 6:
        return f"wait near the {goal_landmark}", True
    elif h == 7:
        return f"stop by the {goal_landmark}", True
    elif h == 8:
        return f"stop at the {goal_landmark}", True
    # Invisible (lm_re-non-matching) stop phrases — 55% (slots 9-19)
    elif h <= 10:
        return f"stop right before the {goal_landmark}", False
    elif h <= 12:
        return f"stop in front of the {goal_landmark}", False
    elif h <= 14:
        return f"stop here in the {goal_room}", False
    elif h <= 16:
        return f"wait here in the {goal_room}", False
    elif h == 17:
        return f"stop when you see the {goal_landmark}", False
    else:
        return f"stop next to the {goal_landmark}", False


def build_v32_prompt(
    start_room: str,
    waypoints_rooms: List[str],
    turn_directions: List[str],
    turn_angles: List[float],
    turn_landmarks: List[str],
    goal_room: str,
    goal_landmark: str,
    n_target: int,
    n_key_turns: int = 0,
    has_stairs: bool = False,
    stair_direction: str = "",
    has_explicit_turns: bool = True,
    chosen_opener: str = "",
    stop_phrase: str = "",
    stop_phrase_visible: bool = True,
    variation_style: str = "",
) -> str:
    """
    v32 prompt — same as v31/v30 (cleanups are in postprocess, not prompt).

    Prompt unchanged from v30: examples, word targets, MANDATORY opener, path structure all identical.
    v32 improvements: _n_target_sentences() (n_key_turns=2 h<70→n=4) + 4 new _QUALITY_CLEANUPS.
    """
    SHARP_TURN_THRESHOLD = SHARP_TURN_THRESHOLD_GLOBAL

    path_lines = [f"Start: {start_room}"]

    for i, (direction, angle, landmark) in enumerate(zip(turn_directions, turn_angles, turn_landmarks)):
        dest_room = waypoints_rooms[i + 1] if i + 1 < len(waypoints_rooms) else goal_room
        is_sharp = angle > SHARP_TURN_THRESHOLD

        landmark_str = f", at {landmark}" if landmark else ""

        if is_sharp:
            dest_str = f" into {dest_room}" if dest_room != (waypoints_rooms[i] if i < len(waypoints_rooms) else start_room) else ""
            path_lines.append(f"  Waypoint {i+1}: turn {direction}{dest_str}{landmark_str}")
        else:
            dest_str = f"walk into {dest_room}" if dest_room != (waypoints_rooms[i] if i < len(waypoints_rooms) else start_room) else "continue forward"
            if landmark:
                path_lines.append(f"  Waypoint {i+1}: {dest_str}{landmark_str}")
            else:
                path_lines.append(f"  Waypoint {i+1}: {dest_str}")

    if has_stairs and stair_direction:
        path_lines.append(f"  Note: go {stair_direction} the stairs at some point")

    if goal_landmark and goal_landmark not in ("the destination", "destination"):
        path_lines.append(f"Goal: {goal_room}")
        path_lines.append(f"STOP TARGET: near the {goal_landmark}")
    else:
        path_lines.append(f"Goal: {goal_room}")

    # v25: use passed stop_phrase (randomized by caller), fallback if not provided
    if not stop_phrase:
        if goal_landmark and goal_landmark not in ("the destination", "destination"):
            stop_phrase = f"stop near the {goal_landmark}"
        else:
            stop_phrase = f"stop in the {goal_room}"

    path_summary = "\n".join(path_lines)

    # opener instruction line
    opener_line = f'\nMANDATORY: Your instruction MUST begin with exactly: "{chosen_opener}"' if chosen_opener else ""

    if n_target == 1:
        examples = [
            "Walk out of the bedroom, go through the hallway, and stop near the framed painting on the wall.",
            "Leave the kitchen and walk into the dining room, then wait near the glass cabinet by the door.",
            "Go through the bathroom and walk through the entryway, stopping near the white potted plant.",
            "Walk forward from the office into the hallway, continue to the living room and stop near the sofa.",
        ]
        no_turn_directive = "\nIMPORTANT: Do NOT use 'turn left' or 'turn right'. MUST mention every room listed in Path context."
        word_target = "16-24 words"
    elif n_target == 2 and not has_explicit_turns:
        examples = [
            "Walk out of the bedroom and through the hallway into the living room. Continue past the dining room and stop near the wooden coffee table.",
            "Leave the kitchen and walk through the entryway into the hallway. Go into the bedroom and wait near the glass cabinet.",
            "Go through the bathroom and along the hallway into the living room. Continue into the dining room and stop near the large wooden bookshelf.",
            "Walk through the office and into the hallway, then continue to the bedroom. Go into the closet and wait near the white wardrobe.",
        ]
        no_turn_directive = "\nIMPORTANT: Do NOT use 'turn left' or 'turn right'. MUST mention every room listed in Path context. Write 2 SHORT sentences — first covers path start, second covers path end and stop."
        word_target = "20-28 words"
    elif n_target == 3 and not has_explicit_turns:
        examples = [
            "Walk out of the bedroom and through the hallway into the kitchen. Continue into the dining room and go past the living room. Stop near the wooden sofa by the window.",
            "Leave the bathroom and walk through the entryway into the hallway. Continue past the bedroom into the kitchen. Wait near the wooden kitchen counter.",
            "Go through the living room and along the hallway into the bedroom. Continue through the closet and into the bathroom. Stop near the white ceramic bathtub.",
            "Walk forward from the office through the hallway into the living room. Continue past the dining room and into the entryway. Wait near the large wooden front door.",
        ]
        no_turn_directive = "\nIMPORTANT: Do NOT use 'turn left' or 'turn right'. MUST mention every room listed in Path context. Write 3 SHORT sentences — each covers one third of the path, with the last ending at the stop target."
        word_target = "24-36 words"
    elif n_target == 4 and not has_explicit_turns:
        examples = [
            "Walk out of the bedroom and through the hallway into the living room. Continue through the dining room and into the kitchen. Go past the laundry room and into the entryway. Stop near the wooden front door.",
            "Leave the bathroom and go through the closet into the bedroom. Continue through the hallway into the living room. Walk into the dining room and through the kitchen. Wait near the large wooden dining table.",
            "Go through the office and along the hallway into the bedroom. Continue through the closet and into the entryway. Walk into the living room and past the kitchen. Stop near the white marble counter.",
            "Walk forward from the kitchen through the dining room into the living room. Continue through the hallway and into the bedroom. Go past the bathroom and into the balcony area. Wait near the outdoor wooden chair.",
        ]
        no_turn_directive = "\nIMPORTANT: Do NOT use 'turn left' or 'turn right'. MUST mention every room listed in Path context. Write 4 SHORT sentences — each covers one quarter of the path, with the last ending at the stop target."
        word_target = "28-42 words"
    elif n_target == 3 and has_explicit_turns and n_key_turns == 1:
        examples = [
            "Walk out of the bedroom and through the long hallway toward the kitchen entrance. Turn left at the white kitchen counter and continue into the living room. Stop near the grey upholstered armchair by the window.",
            "Leave the bathroom and walk through the entryway past the glass cabinet. Turn right at the wooden bookshelf and go into the dining room. Wait near the large wooden dining table.",
            "Walk out of the office and along the hallway toward the bedroom area. Turn left at the wooden dresser and go into the closet. Stop near the white wardrobe at the far wall.",
            "Turn right out of the living room and walk through the kitchen toward the dining area. Continue past the counter and into the hallway. Wait near the white entry door.",
        ]
        no_turn_directive = "\nIMPORTANT: Write 3 sentences — first covers path start, second covers the turn, third ends at stop target."
        word_target = "26-38 words"
    else:
        no_turn_directive = ""
        word_target = {2: "24-33 words", 3: "28-38 words", 4: "33-45 words"}.get(n_target, "24-38 words")
        examples = {
            2: [
                "Walk out of the bedroom and turn left into the hallway at the wooden pillar. Go into the living room and stop near the wooden rug.",
                "Leave the kitchen and turn right at the counter. Continue into the dining room and wait near the table.",
                "Turn right out of the bathroom and go through the hallway past the white pillar. Stop near the door frame.",
                "Walk through the entryway and turn left at the wooden bookshelf. Continue into the bedroom and wait near the white dresser.",
            ],
            3: [
                "Walk out of the living room and turn right at the hallway. Turn left past the grey sofa and go into the dining room. Stop near the wooden table.",
                "Leave the bedroom and walk into the kitchen. Turn left at the counter and go into the dining room. Wait near the glass stool.",
                "Go through the bathroom and turn left at the hallway. Walk through the kitchen and turn right at the wooden door. Stop near the window.",
                "Walk forward from the closet and turn right through the bedroom. Go into the bathroom and turn left at the dresser. Wait near the bathtub.",
            ],
            4: [
                "Walk out of the bedroom and turn right into the hallway at the wooden bookshelf. Go through the living room and turn left at the archway. Continue through the dining room and stop near the table.",
                "Leave the kitchen and turn right at the stairs. Go up and turn right into the hallway. Walk into the bedroom and wait near the wooden dresser.",
                "Go through the bathroom and turn left at the hallway. Walk past the kitchen and turn left at the white counter. Continue into the dining room and stop near the wooden table.",
                "Walk forward from the living room and turn left at the wooden cabinet. Go up the stairs and turn right at the landing railing. Walk into the bedroom and wait near the white bed.",
            ],
        }.get(n_target, ["Walk through the rooms following the path and stop near the destination."])

    examples_str = "\n".join(f'"{e}"' for e in examples)
    variation_line = f"\nVariation emphasis: {variation_style}" if variation_style else ""

    # v46: Two-tier constraint based on stop phrase lm_re visibility
    if stop_phrase_visible:
        # Stop phrase matches lm_re — use "max 1 lm" constraint (the stop phrase counts)
        lm_constraint = (
            "IMPORTANT: Use at most 1 furniture/object landmark reference in the ENTIRE instruction "
            "(the required stop phrase counts as this landmark). "
            "Do NOT mention furniture or objects at turns — describe turns by direction and room only."
        )
    else:
        # Stop phrase does NOT match lm_re — enforce zero lm_re matches in the instruction body
        lm_constraint = (
            "IMPORTANT: Do NOT use the phrases 'at the', 'near the', 'past the', or 'by the' "
            "anywhere in the instruction. When describing positions or objects, use ONLY: "
            "'in front of', 'right before', 'next to', 'through', 'into', 'along', 'toward'. "
            "Do NOT mention furniture or objects at turns — describe turns by direction and room only."
        )

    return f"""You are writing indoor navigation instructions for a robot.
Use these room names exactly as given. Do NOT change them.{opener_line}
Style: use spatial qualifiers to describe the path ("along the hallway", "toward the back of the room", "down the corridor", "to the far end", "continuing past the entrance"). Use varied action verbs: walk out, leave, head out, walk through, turn. Avoid: proceed, traverse, cross, pass as starters. {lm_constraint} Target {word_target} total.{no_turn_directive}{variation_line}

Path context:
{path_summary}

Write EXACTLY {n_target} navigation sentence{"s" if n_target > 1 else ""} following this style:
{examples_str}

CRITICAL: The LAST sentence MUST end with exactly: "{stop_phrase}"
Write ONLY the instruction. No extra text."""


def postprocess_v22(raw: str, n_target: int, goal_room: str, goal_landmark: str) -> str:
    """Same postprocess as v21/v15 — unchanged."""
    text = raw.strip()

    for prefix in ["Here is", "Instruction:", "Navigation:", "Sure,", "Certainly,"]:
        if text.lower().startswith(prefix.lower()):
            idx = text.find('\n') if '\n' in text else len(prefix) + 20
            text = text[idx:].strip()

    text = text.strip('"\'')
    text = re.sub(r'\s+', ' ', text).strip()

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    sentences = [re.sub(r'^\d+[\.\)]\s*', '', s) for s in sentences]
    sentences = [s for s in sentences if len(s.split()) >= 3]

    if len(sentences) > n_target:
        sentences = sentences[:n_target]

    if not sentences:
        return text or "Walk to the destination and stop."

    result = ' '.join(sentences)
    if result and result[-1] not in '.!?':
        result += '.'

    stop_re = re.compile(r'\b(stop|wait|halt|stand)\b', re.IGNORECASE)
    has_specific_landmark = goal_landmark and goal_landmark not in ("the destination", "destination")

    if not stop_re.search(result):
        if has_specific_landmark:
            result = result.rstrip('.!?') + f". Stop near the {goal_landmark}."
        elif goal_room and goal_room not in ('room', 'unknown'):
            result = result.rstrip('.!?') + f". Stop in the {goal_room}."
        else:
            result = result.rstrip('.!?') + ". Stop at the destination."
    elif has_specific_landmark:
        room_words = r'(hallway|bedroom|bathroom|kitchen|dining room|living room|entryway|closet|garage|lobby|lounge|office|family room|stairs|staircase|corridor|rec room)'
        room_stop_re = re.compile(
            r'(stop|wait)\s+(?:near\s+the\s+|in\s+the\s+|at\s+the\s+)' + room_words + r'[.!?]?$',
            re.IGNORECASE
        )
        if room_stop_re.search(result):
            result = re.sub(
                r'(stop|wait)\s+(?:near\s+the\s+|in\s+the\s+|at\s+the\s+)' + room_words + r'([.!?]?)$',
                f'stop near the {goal_landmark}\\3',
                result,
                flags=re.IGNORECASE
            )

    return result


async def generate_all(tasks: list, checkpoint: dict) -> dict:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
    sem = asyncio.Semaphore(CONCURRENCY)

    results = dict(checkpoint)
    pending = [t for t in tasks if str(t["episode_id"]) not in results]
    print(f"  Phase G pending: {len(pending)} episodes ({len(results)} from checkpoint)")

    async def _one(task):
        async with sem:
            msgs = [{"role": "user", "content": task["prompt"]}]
            try:
                resp = await client.chat.completions.create(
                    model=VLLM_MODEL, messages=msgs,
                    max_tokens=128, temperature=TEMPERATURE,
                )
                raw = resp.choices[0].message.content.strip()
                return task["episode_id"], raw, True
            except Exception:
                return task["episode_id"], "", False

    coros = [_one(t) for t in pending]
    t0 = time.time()
    done = 0
    errors = 0

    for coro in asyncio.as_completed(coros):
        eid, raw, ok = await coro
        if ok:
            results[str(eid)] = raw
        else:
            errors += 1
            results[str(eid)] = ""
        done += 1
        if done % 100 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(pending) - done) / rate if rate > 0 else 0
            print(f"  [G {done}/{len(pending)}] ok={done-errors} err={errors} "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}m")
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(results, f)

    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    print(f"Phase G done {len(pending)} in {elapsed:.1f}s ({len(pending)/max(elapsed,1):.1f}/s). errors={errors}")
    return results


def __postprocess_v25_base(raw: str, n_target: int, goal_room: str, goal_landmark: str, chosen_stop: str = "") -> str:
    """v25 postprocess — v22 + respect chosen_stop (stop near/wait near/stop by/stop at)."""
    # Run base postprocess first
    text = postprocess_v22(raw, n_target, goal_room, goal_landmark)

    if not chosen_stop or "stop near" in chosen_stop:
        return text  # v22 already handles stop near

    # Replace "stop near the X" with chosen_stop variant at end of instruction
    has_landmark = goal_landmark and goal_landmark not in ("the destination", "destination")
    if not has_landmark:
        return text

    # Try to replace the stop condition with the chosen variant
    stop_near_re = re.compile(
        r'(stop|wait)\s+(?:near|by|at)\s+the\s+' + re.escape(goal_landmark) + r'([.!?]?)$',
        re.IGNORECASE
    )
    if stop_near_re.search(text):
        text = stop_near_re.sub(chosen_stop + r'\2', text)
    return text


# Head-verb patterns to replace (head X → walk/turn X) to match GT distribution (~1.5% vs model's ~97%)
# NOTE: specific "head out + direction" patterns MUST come before generic "head out" to avoid
# producing "walk out forward" / "walk out through" / "walk out past" artifacts.
_HEAD_REPLACEMENTS = [
    (re.compile(r'\bhead\s+left\b', re.IGNORECASE), 'turn left'),
    (re.compile(r'\bhead\s+right\b', re.IGNORECASE), 'turn right'),
    # Specific "head out + direction" — before generic "head out" to avoid "walk out forward" etc.
    (re.compile(r'\bhead\s+out\s+forward\b', re.IGNORECASE), 'walk forward'),
    (re.compile(r'\bhead\s+out\s+through\b', re.IGNORECASE), 'walk through'),
    (re.compile(r'\bhead\s+out\s+past\b', re.IGNORECASE), 'walk past'),
    (re.compile(r'\bhead\s+out\s+and\b', re.IGNORECASE), 'walk out and'),
    # Generic "head out" → "walk out"
    (re.compile(r'\bhead\s+out\b', re.IGNORECASE), 'walk out'),
    (re.compile(r'\bhead\s+into\b', re.IGNORECASE), 'walk into'),
    (re.compile(r'\bhead\s+through\b', re.IGNORECASE), 'walk through'),
    (re.compile(r'\bhead\s+toward(?:s)?\b', re.IGNORECASE), 'walk toward'),
    (re.compile(r'\bhead\s+back\b', re.IGNORECASE), 'walk back'),
    (re.compile(r'\bhead\s+past\b', re.IGNORECASE), 'walk past'),
    (re.compile(r'\bhead\s+down\b', re.IGNORECASE), 'walk down'),
    (re.compile(r'\bhead\s+up\b', re.IGNORECASE), 'walk up'),
    (re.compile(r'\bhead\s+over\b', re.IGNORECASE), 'walk over'),
    (re.compile(r'\bhead\s+along\b', re.IGNORECASE), 'walk along'),
    (re.compile(r'\bhead\s+around\b', re.IGNORECASE), 'walk around'),
    (re.compile(r'\bhead\s+straight\b', re.IGNORECASE), 'walk straight'),
    (re.compile(r'\bhead\s+to\b', re.IGNORECASE), 'walk to'),
    (re.compile(r'\bhead\s+for\b', re.IGNORECASE), 'walk toward'),
    (re.compile(r'\bhead\s+in\b', re.IGNORECASE), 'walk in'),
    (re.compile(r'\bhead\s+across\b', re.IGNORECASE), 'walk across'),
    (re.compile(r'\bhead\s+forward\b', re.IGNORECASE), 'walk forward'),
]

# Post-replacement cleanup: removes awkward "walk out X" that survive from model output or
# opener + model concatenation (e.g. opener "Walk out of the room" + ", walk out forward").
_WALK_OUT_CLEANUPS = [
    (re.compile(r'\bwalk\s+out\s+forward\b', re.IGNORECASE), 'walk forward'),
    (re.compile(r'\bwalk\s+out\s+through\b', re.IGNORECASE), 'walk through'),
    (re.compile(r'\bwalk\s+out\s+past\b', re.IGNORECASE), 'walk past'),
    # v59: 527 instances (28.7%) + 30 instances (1.6%) — reduces avg_words by ~0.29
    (re.compile(r'\bwalk\s+out\s+along\b', re.IGNORECASE), 'walk along'),
    (re.compile(r'\bwalk\s+out\s+down\b', re.IGNORECASE), 'walk down'),
]

# v28+: Additional quality cleanups
# 1. "walk and turn/go/move/continue/return" → "walk forward and ..."  (leftover from head_out_and)
# 2. "walk through the [furniture]" → "walk past the [furniture]"      (semantic correction)
# v29 NEW:
# 3. "walk into ROOM at LANDMARK" → "walk into ROOM near the LANDMARK" (natural English connector)
# 4. "walk out into ROOM at LANDMARK" → "walk out into ROOM near the LANDMARK"
# v30 NEW:
# 5. "walk out and turn X" → "turn X"         (artifact from head_out_and: "head out and turn" → "walk out and turn")
# 6. "Now, x" → "X"                           (filler discourse marker, 124 instances)
# 7. "Finally, x" → "X"                       (filler discourse marker, 65 instances)
# 8. "Then, x" → "X"                          (filler discourse marker, 1 instance)
# 9. "walk out into X" → "walk into X"         (redundant motion prefix, 669 instances, -0.4 words/ep)
# v32 NEW:
# 10. "walk out toward X" → "walk toward X"   (331 instances, GT has 0; model artifact, -0.18w avg)
# 11. "walk forward (to|past|into|through)" → "walk (to|past|into|through)"  (~35 instances, -0.019w avg)
# 12. ", then [verb]" → " and [verb]"          (37 instances, removes filler "then", -0.020w avg)
# 13. "Walk out and [non-turn verb]" → "[Verb]" (15 residual, expands v30 fix to all verbs at sentence start)
_QUALITY_CLEANUPS = [
    (re.compile(r'\bwalk and (turn|go|move|continue|return)\b', re.IGNORECASE),
     lambda m: ('Walk' if m.group(0)[0].isupper() else 'walk') + ' forward and ' + m.group(1).lower()),
    (re.compile(
        r'\bwalk through the (sofa|couch|armchair|chair|table|desk|cabinet|counter|beds?\b|curtain|shelf|shelves)\b',
        re.IGNORECASE),
     lambda m: ('Walk' if m.group(0)[0].isupper() else 'walk') + ' past the ' + m.group(1)),
    # "walk [out] into ROOM at [the] LANDMARK" → "walk [out] into ROOM near the LANDMARK"
    (re.compile(r'\b(walk(?:\s+out)?\s+into\s+[\w\s]{3,30}?)\s+at\s+(?:the\s+)?((?:[\w]+\s*){1,5}(?:wall|door|counter|table|cabinet|frame|stair|fireplace|window|bed|shelf|rail|post|pillar|column)[\w]*)\b',
                re.IGNORECASE),
     lambda m: m.group(1) + ' near the ' + m.group(2).lower()),
    # v30: "walk out and turn X" → "turn X" (artifact from head_out_and → walk_out_and replacement)
    (re.compile(r'\bwalk\s+out\s+and\s+(turn\s+(?:left|right|around)\b)', re.IGNORECASE),
     lambda m: m.group(1)[0].upper() + m.group(1)[1:] if m.group(0)[0].isupper() else m.group(1)),
    # v30: Remove "Now/Finally/Then, " filler discourse markers at sentence transitions
    (re.compile(r'\b(?:Now|Finally|Then),\s+([a-z])'),
     lambda m: m.group(1).upper()),
    # v30: "walk out into X" → "walk into X" (redundant motion prefix in mid-instruction)
    (re.compile(r'\bwalk\s+out\s+into\b', re.IGNORECASE),
     lambda m: 'Walk into' if m.group(0)[0].isupper() else 'walk into'),
    # v32: "walk out toward X" → "walk toward X" (331 instances, GT has 0; -0.18w avg)
    (re.compile(r'\bwalk\s+out\s+(toward)\b', re.IGNORECASE),
     lambda m: ('Walk ' if m.group(0)[0].isupper() else 'walk ') + m.group(1)),
    # v32: "walk forward (to|past|into|through)" → "walk (to|past|into|through)" (~35 instances)
    (re.compile(r'\bwalk\s+forward\s+((?:to|past|into|through)\b)', re.IGNORECASE),
     lambda m: ('Walk ' if m.group(0)[0].isupper() else 'walk ') + m.group(1)),
    # v32: ", then [verb]" → " and [verb]" (37 instances; removes filler "then")
    (re.compile(r',\s+then\s+([a-z])'),
     lambda m: ' and ' + m.group(1)),
    # v32: "Walk out and [non-turn verb]" → "[Verb...]" at sentence starts (15 residual after v30 turn-fix)
    (re.compile(r'\bWalk\s+out\s+and\s+(?!turn\s)([a-z])'),
     lambda m: m.group(1).upper()),
    # v34: "return to X" → "walk to X" (31 instances, GT has 0; model artifact on backtrack paths)
    (re.compile(r'\breturn to\b', re.IGNORECASE),
     lambda m: 'Walk to' if m.group(0)[0].isupper() else 'walk to'),
    # v34: "continue forward" → "walk straight" (vocabulary matching: GT has walk_straight=8.8%, we had 0%)
    (re.compile(r'\bcontinue forward\b', re.IGNORECASE),
     lambda m: 'Walk straight' if m.group(0)[0].isupper() else 'walk straight'),
]


def replace_head_verbs(text: str) -> str:
    """Replace 'head X' → 'walk/turn X', clean up 'walk out X' artifacts, apply quality fixes.

    v52 fix: preserve 'Head' as opener (first word) to allow head-based instruction openers.
    Only body 'head X' patterns are converted; the opener word is restored if it was 'Head'.
    """
    # v52: detect if instruction starts with "Head" (opener) — preserve it through replacements
    opener_is_head = text.lower().startswith("head ")

    if opener_is_head:
        # Apply replacements only to the body (after first word) to preserve opener
        first_word_len = len(text.split()[0])  # length of "Head"
        body = text[first_word_len:]
        for pattern, replacement in _HEAD_REPLACEMENTS:
            def _repl(m, r=replacement):
                return r[0].upper() + r[1:] if m.group(0)[0].isupper() else r
            body = pattern.sub(_repl, body)
        for pattern, replacement in _WALK_OUT_CLEANUPS:
            def _repl(m, r=replacement):
                return r[0].upper() + r[1:] if m.group(0)[0].isupper() else r
            body = pattern.sub(_repl, body)
        for pattern, repl_fn in _QUALITY_CLEANUPS:
            body = pattern.sub(repl_fn, body)
        return text[:first_word_len] + body  # restore original "Head" opener

    for pattern, replacement in _HEAD_REPLACEMENTS:
        def _repl(m, r=replacement):
            return r[0].upper() + r[1:] if m.group(0)[0].isupper() else r
        text = pattern.sub(_repl, text)
    for pattern, replacement in _WALK_OUT_CLEANUPS:
        def _repl(m, r=replacement):
            return r[0].upper() + r[1:] if m.group(0)[0].isupper() else r
        text = pattern.sub(_repl, text)
    for pattern, repl_fn in _QUALITY_CLEANUPS:
        text = pattern.sub(repl_fn, text)
    return text


def postprocess_v32(raw: str, n_target: int, goal_room: str, goal_landmark: str, chosen_stop: str = "") -> str:
    """v32 = v31 + word-reduction cleanups (walk_out_toward, walk_forward_X, comma_then, walk_out_and_X)."""
    text = __postprocess_v25_base(raw, n_target, goal_room, goal_landmark, chosen_stop)
    return replace_head_verbs(text)


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_AND_WALK_PREP_RE = re.compile(
    r'\band\s+walk\s+(into|through|past|toward|out|to|up|down|along|across|over|around)\b',
    re.IGNORECASE
)


def _fix_double_walk(text: str) -> str:
    """Remove redundant 'walk' in 'walk X and walk Y' patterns within same sentence.

    GT has only 3.8% double-walk vs v32's 24.5%. This fix removes the excess by dropping
    the second 'walk' when another 'walk' already precedes it in the same sentence.
    'Turn X and walk Y' patterns are preserved (no preceding walk → no change).
    """
    sentences = _SENTENCE_SPLIT_RE.split(text)
    result = []
    for sent in sentences:
        original_sent = sent

        def _sub(m, s=original_sent):
            before = s[:m.start()]
            if re.search(r'\bwalk\b', before, re.IGNORECASE):
                return 'and ' + m.group(1)
            return m.group(0)

        sent = _AND_WALK_PREP_RE.sub(_sub, sent)
        result.append(sent)
    return ' '.join(result)


def postprocess_v33(raw: str, n_target: int, goal_room: str, goal_landmark: str, chosen_stop: str = "") -> str:
    """v33 = v32 + _fix_double_walk() removes redundant second 'walk' in 'walk X and walk Y' patterns.

    Saves ~0.227w per episode: avg_words 27.10 → 26.874 (GT=26.8, +0.074 — nearly perfect).
    """
    text = __postprocess_v25_base(raw, n_target, goal_room, goal_landmark, chosen_stop)
    text = replace_head_verbs(text)
    return _fix_double_walk(text)


_COMMA_WALK_PREP_RE = re.compile(
    r',\s+walk\s+(into|through|past|toward|to|up|down|along|across|over|around)\b',
    re.IGNORECASE
)


def _fix_comma_walk(text: str) -> str:
    """Remove redundant 'walk' after comma when same sentence already has 'walk'.

    ', walk (prep)' → ', (prep)' only when 'walk' precedes in same sentence.
    Saves ~0.122w/ep: avg_words 26.874 → 26.752 (GT=26.8, -0.05 — essentially perfect).
    Pattern: 'Walk out of X, walk through Y' → 'Walk out of X, through Y'
    """
    sentences = _SENTENCE_SPLIT_RE.split(text)
    result = []
    for sent in sentences:
        original_sent = sent

        def _sub(m, s=original_sent):
            before = s[:m.start()]
            if re.search(r'\bwalk\b', before, re.IGNORECASE):
                return ', ' + m.group(1)
            return m.group(0)

        sent = _COMMA_WALK_PREP_RE.sub(_sub, sent)
        result.append(sent)
    return ' '.join(result)


def postprocess_v34(raw: str, n_target: int, goal_room: str, goal_landmark: str, chosen_stop: str = "") -> str:
    """v34 = v33 + return_to fix + continue_forward→walk_straight + _fix_comma_walk().

    Changes vs v33:
    - _QUALITY_CLEANUPS: return_to→walk_to (31 instances, GT=0), continue_forward→walk_straight (130)
    - _fix_comma_walk: ', walk X' → ', X' when walk precedes (saves 0.122w/ep)
    - avg_words: 26.874 → 26.752 (GT=26.8, -0.05 — essentially PERFECT)
    """
    text = __postprocess_v25_base(raw, n_target, goal_room, goal_landmark, chosen_stop)
    text = replace_head_verbs(text)   # includes v34 cleanups (return_to, continue_forward)
    text = _fix_double_walk(text)     # v33
    text = _fix_comma_walk(text)      # v34
    return text


def main():
    from gate2_path.path_analyzer import analyze_path
    from gate5_tokenizer.tokenizer import VLNTokenizer
    from gate6_assembler.assembler import assemble_episode

    print("Loading data...")

    with gzip.open(VAL_UNSEEN_PATH, "rt") as f:
        raw_data = json.load(f)
    episodes = raw_data["episodes"]

    # Load Phase1 fallback data (used when vision fails)
    p1 = json.load(open(PHASE1_CHECKPOINT))
    p73 = json.load(open(PHASE1_V73_CHECKPOINT)) if PHASE1_V73_CHECKPOINT.exists() else {}
    p74 = json.load(open(PHASE1_V74_CHECKPOINT)) if PHASE1_V74_CHECKPOINT.exists() else {}
    print(f"  Phase1 v19: {len(p1)}, v73: {len(p73)}, v74: {len(p74)}")

    g3 = {}
    if GATE3_PERFRAME_DIR.exists():
        for fp in GATE3_PERFRAME_DIR.glob("episode_*.json"):
            try:
                d = json.load(open(fp))
                g3[d["episode_id"]] = d
            except Exception:
                pass
    print(f"  Gate3 perframe: {len(g3)}")

    # Load vision checkpoint
    vision_checkpoint = {}
    if VISION_CHECKPOINT_PATH.exists():
        vision_checkpoint = json.load(open(VISION_CHECKPOINT_PATH))
        print(f"  Vision checkpoint: {len(vision_checkpoint)} done")

    # --- Phase V: Extract visual landmarks ---
    print("\nPhase V: Vision landmark extraction...")
    visual_landmarks = asyncio.run(extract_visual_landmarks_all(episodes, vision_checkpoint))

    # Vision quality summary
    n_with_turn_visual = sum(1 for v in visual_landmarks.values() if any(v.get("turns", [])))
    n_with_goal_visual = sum(1 for v in visual_landmarks.values() if v.get("goal", ""))
    print(f"  Visual turn landmarks: {n_with_turn_visual}/{len(visual_landmarks)} episodes")
    print(f"  Visual goal landmarks: {n_with_goal_visual}/{len(visual_landmarks)} episodes")

    # Load text checkpoint (v35: fresh generation, no prev checkpoint reuse)
    if not CHECKPOINT_PATH.exists() and PREV_CHECKPOINT_PATH and Path(PREV_CHECKPOINT_PATH).exists():
        import shutil
        shutil.copy(PREV_CHECKPOINT_PATH, CHECKPOINT_PATH)
        print(f"  Initialized v35 checkpoint from prev")
    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        checkpoint = json.load(open(CHECKPOINT_PATH))
        print(f"  Text checkpoint: {len(checkpoint)} done")

    tokenizer = VLNTokenizer()

    # v61: ~1196 pre-seeded (h=35-99 from v60), ~643 re-generated (h=0-34: new opener logic)
    # v63: ~1825 pre-seeded (from v62 minus 14 dup EIDs), 14 re-generated with opener overrides
    _V63_PREV_CHECKPOINT = set(checkpoint.keys())

    # Opener overrides for 14 intractable dup EIDs: break collision by changing opener type
    # ALL 14 new v62 dups are n_kt=0 pairs that both get same Walk X opener (after leak fix)
    # Fix: give one EID a different walk opener to differentiate the instruction
    _V63_OPENER_OVERRIDE = {}  # maps eid (int) → opener format string (uses {start_room})
    _v62_out = ROOT / "outputs" / "datasets" / "val_unseen_gate3_gemma_v62.json.gz"
    if _v62_out.exists():
        import gzip as _gz
        with _gz.open(_v62_out) as _f:
            _v62_eps = {ep["episode_id"]: ep["instruction"]["instruction_text"]
                        for ep in json.load(_f)["episodes"]}
    # Walk-through → Walk-out-of overrides (second EID from each "Walk through" pair)
    for eid in [225, 426, 732, 1223, 1232, 1434, 1632]:
        _V63_OPENER_OVERRIDE[eid] = "Walk out of the {start_room}"
    # Walk-out-of → Walk-through overrides (second EID from each "Walk out of" pair)
    for eid in [401, 710, 803, 903, 1110]:
        _V63_OPENER_OVERRIDE[eid] = "Walk through the {start_room}"
    # Triple dup: 724 pairs with both 725 AND 726 (both "Walk through")
    # Override 724 to "Exit the" to differentiate from both partners simultaneously
    # Also override 725 to "Walk out of" since 725 and 726 remain identical after 724 fix
    _V63_OPENER_OVERRIDE[724] = "Exit the {start_room}"
    _V63_OPENER_OVERRIDE[725] = "Walk out of the {start_room}"
    print(f"  v63 opener overrides: {len(_V63_OPENER_OVERRIDE)} EIDs")

    tasks = []
    ep_meta = {}

    for ep in episodes:
        eid = ep["episode_id"]
        scene_id = ep["scene_id"]

        regions = parse_house_regions(scene_id)
        waypoint_rooms = get_waypoint_rooms(ep["reference_path"], regions)

        pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        prims = pa.get("primitives", [])

        key_turn_prims = [p for p in prims if _is_key_turn(p)]
        n_key_turns = len(key_turn_prims)

        room_transitions = _count_room_transitions(waypoint_rooms)
        n_target = _n_target_sentences(n_key_turns, eid, room_transitions=room_transitions)
        has_explicit_turns = (n_key_turns > 0)

        all_turn_directions = []
        all_turn_angles = []
        all_turn_waypoint_indices = []
        wp_idx = 0
        for p in prims:
            if p["type"] == "straight":
                wp_idx += min(1, int(p.get("distance_m", 1) / 1.0))
            elif p["type"] in ("left_turn", "right_turn"):
                direction = "left" if p["type"] == "left_turn" else "right"
                angle = p.get("angle_deg", 90.0)
                all_turn_directions.append(direction)
                all_turn_angles.append(angle)
                all_turn_waypoint_indices.append(min(wp_idx, len(waypoint_rooms) - 1))
                wp_idx += 1

        # Visual landmarks for this episode
        vis_ep = visual_landmarks.get(str(eid), {})
        vis_turns = vis_ep.get("turns", [])
        vis_goal = vis_ep.get("goal", "")

        # Phase1 fallback data
        p1ep = p1.get(str(eid), {})
        p1_turn_keys = sorted(
            [k for k in p1ep.keys() if k.startswith("turn_")],
            key=lambda x: int(x.split("_")[1])
        )
        g3ep = g3.get(eid, {})
        g3_turns = g3ep.get("turns", [])
        p73ep = p73.get(str(eid), {})

        # Build turn landmarks: visual primary, Phase1 fallback
        turn_landmarks = []
        for i, direction in enumerate(all_turn_directions):
            # v22: Try visual landmark from ts_turn image first
            visual_lm = vis_turns[i] if i < len(vis_turns) else ""

            if visual_lm:
                turn_landmarks.append(visual_lm)
            else:
                # Fall back to Phase1 v19 → v73
                g3t = g3_turns[i] if i < len(g3_turns) else {}
                label = g3t.get("label", "")
                if label and label in p1ep:
                    p1_desc = p1ep[label]
                elif i < len(p1_turn_keys):
                    p1_desc = p1ep[p1_turn_keys[i]]
                else:
                    p1_desc = ""
                landmark = _shorten_landmark(p1_desc)
                if not landmark and p73ep:
                    turn_key = f"turn_{i+1}"
                    if turn_key in p73ep:
                        landmark = p73ep[turn_key]
                turn_landmarks.append(landmark)

        # Build goal landmark: visual primary, v74 secondary, g3 fallback
        start_room = waypoint_rooms[0] if waypoint_rooms else "room"
        goal_room = waypoint_rooms[-1] if waypoint_rooms else "room"

        p74ep = p74.get(str(eid), {})
        v74_raw = p74ep.get("desc", "") if isinstance(p74ep, dict) else ""
        v74_desc = (v74_raw or "").strip()
        g3_goal = g3ep.get("goal") or {}
        g3_landmark = g3_goal.get("stop_landmark", "") or g3_goal.get("main_landmark", "")

        if vis_goal and not _GENERIC_LANDMARK_RE.search(vis_goal):
            # v22 primary: visual landmark from actual goal-area image
            goal_landmark = vis_goal
        elif v74_desc and not _GENERIC_LANDMARK_RE.search(v74_desc):
            goal_landmark = v74_desc
        elif g3_landmark:
            goal_landmark = g3_landmark
        else:
            goal_landmark = "the destination"

        has_stairs = any("stair" in r.lower() for r in waypoint_rooms)
        start_height = ep["reference_path"][0][1] if ep.get("reference_path") else 0
        end_height = ep["reference_path"][-1][1] if ep.get("reference_path") else 0
        stair_direction = ""
        if has_stairs:
            stair_direction = "up" if end_height > start_height + 0.5 else ("down" if end_height < start_height - 0.5 else "up")

        # v61: pass only MAJOR turns (angle > threshold) to opener — fixes 119 n_kt=0 leaks
        all_major_turn_directions = [d for d, a in zip(all_turn_directions, all_turn_angles)
                                     if a > SHARP_TURN_THRESHOLD_GLOBAL]
        chosen_opener = _choose_opener_v35(eid, start_room, all_major_turn_directions)
        # v63: override opener for 14 intractable dup EIDs to break same-opener collision
        if eid in _V63_OPENER_OVERRIDE:
            chosen_opener = _V63_OPENER_OVERRIDE[eid].format(start_room=start_room)
        # v46: _choose_stop_phrase returns (phrase, is_lm_visible) tuple
        chosen_stop, chosen_stop_visible = _choose_stop_phrase(eid, goal_landmark, goal_room)

        # v41: blank turn_lm + spatial style + max1lm constraint
        v41_turn_landmarks = [""] * len(turn_landmarks)

        # v52: scene-aware hash for variation style (7 styles) — fixes same-scene duplicate issue.
        # Root cause: int(eid) % 5 caused same-scene episodes with similar paths to share style.
        # Fix: hash(scene_name + eid) % 7 decorrelates style assignment within each scene.
        # v54: 20-angle uniqueness pool — structural diversity breaks same-scene convergence
        _V54_STYLES = [
            "Use direct, concise sentences with clear directional guidance.",
            "Describe each room you pass through and emphasize how spaces connect.",
            "Emphasize spatial relationships: distances, relative positions, path character.",
            "Use active, dynamic verbs and flowing transitions between spaces.",
            "Focus on the destination and work backward — orient toward where you are going.",
            "Begin with a broad overview of the route, then add specific navigational details.",
            "Use landmark transitions: reference what you pass by or see along the way.",
            "Structure as a sequence of discrete moves: first X, then Y, then Z.",
            "Describe what you pass on your left and right as you walk.",
            "Lead with room names and transitions between them as the organizing structure.",
            "Use a narrative voice — describe the path as if telling a story of moving through space.",
            "Emphasize the geometry: straight segments, angles, distances in rough steps.",
            "Write as minimal one-sentence-per-move instructions with short, punchy language.",
            "Focus on what you see ahead at each decision point before describing the turn.",
            "Start from the exit of each space rather than the entry — describe leaving each room.",
            "Mention surface textures, flooring, and wall features to anchor spatial memory.",
            "Use connective phrases that carry momentum: 'continuing past', 'sweeping left', 'cutting through'.",
            "Describe the path using the feel of the space: open vs narrow, bright vs dim.",
            "Write in second person present tense: 'you walk into', 'you turn left', 'you arrive'.",
            "Anchor each move to a named landmark or piece of furniture as the reference point.",
        ]
        _scene_name = scene_id.split("/")[-2] if "/" in scene_id else scene_id
        _hash_key = f"{_scene_name}_{eid}_v63".encode()
        style_idx = int(hashlib.md5(_hash_key).hexdigest(), 16) % 20
        _eid_str = str(eid)
        # v63: uniqueness suffix for 14 re-generated EIDs (intractable dup pairs)
        if _eid_str not in _V63_PREV_CHECKPOINT:
            _uniqueness_suffix = (
                f" CRITICAL: Your instruction MUST be structurally and lexically unique. "
                f"Use this specific perspective angle #{_hash_key.hex()[:6]}."
            )
        else:
            _uniqueness_suffix = ""
        v43_variation = _V54_STYLES[style_idx] + _uniqueness_suffix

        prompt = build_v32_prompt(
            start_room=start_room,
            waypoints_rooms=waypoint_rooms,
            turn_directions=all_turn_directions,
            turn_angles=all_turn_angles,
            turn_landmarks=v41_turn_landmarks,
            goal_room=goal_room,
            goal_landmark=goal_landmark,
            n_target=n_target,
            n_key_turns=n_key_turns,
            has_stairs=has_stairs,
            stair_direction=stair_direction,
            has_explicit_turns=has_explicit_turns,
            chosen_opener=chosen_opener,
            stop_phrase=chosen_stop,
            stop_phrase_visible=chosen_stop_visible,
            variation_style=v43_variation,
        )

        tasks.append({"episode_id": eid, "prompt": prompt, "chosen_opener": chosen_opener, "chosen_stop": chosen_stop})
        ep_meta[eid] = {
            "n_key_turns": n_key_turns,
            "n_target": n_target,
            "goal_room": goal_room,
            "goal_landmark": goal_landmark,
            "room_transitions": room_transitions,
            "vis_turn_hit": bool(any(vis_turns[:n_key_turns])),
            "vis_goal_hit": bool(vis_goal),
            "chosen_opener": chosen_opener,
            "chosen_stop": chosen_stop,
        }

    print(f"\nTotal tasks: {len(tasks)}")

    from collections import Counter
    dist = Counter(ep_meta[ep["episode_id"]]["n_target"] for ep in episodes)
    total = len(episodes)
    print("Projected sentence distribution:")
    for k in sorted(dist):
        print(f"  {k}s: {dist[k]} ({dist[k]/total*100:.1f}%)")

    vis_turn_hits = sum(1 for m in ep_meta.values() if m["vis_turn_hit"])
    vis_goal_hits = sum(1 for m in ep_meta.values() if m["vis_goal_hit"])
    print(f"Visual landmark coverage: turns={vis_turn_hits}/{total} goal={vis_goal_hits}/{total}")

    # --- Phase G: Generate text instructions ---
    print("\nPhase G: Text instruction generation...")
    results = asyncio.run(generate_all(tasks, checkpoint))

    # Assemble dataset
    print("\nAssembling output dataset...")
    out_episodes = []
    stop_re = re.compile(r'\b(stop|wait|halt|stand)\b', re.IGNORECASE)

    for ep in episodes:
        eid = ep["episode_id"]
        raw = results.get(str(eid), "")
        meta = ep_meta.get(eid, {})
        n_target = meta.get("n_target", 2)
        goal_room = meta.get("goal_room", "room")
        goal_landmark = meta.get("goal_landmark", "the destination")

        chosen_stop = meta.get("chosen_stop", f"stop near the {goal_landmark}")
        fallback_stop = chosen_stop.split("stop near the ")[-1].split("wait near the ")[-1].split("stop by the ")[-1].split("stop at the ")[-1]
        instruction_text = postprocess_v34(raw, n_target, goal_room, goal_landmark, chosen_stop) if raw \
            else f"Walk to the {goal_room} and {chosen_stop}."

        assembled = assemble_episode(
            source_episode=ep,
            generated_text=instruction_text,
            tokenizer=tokenizer,
        )
        out_episodes.append(assembled)

    with gzip.open(VOCAB_SOURCE_PATH, "rt") as f:
        vocab_data = json.load(f)

    out_data = {
        "episodes": out_episodes,
        "instruction_vocab": vocab_data.get("instruction_vocab", {}),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUTPUT_PATH, "wt") as f:
        json.dump(out_data, f)

    print(f"\nSaved {len(out_episodes)} episodes to {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size // 1024}KB")

    # Quality report
    import collections as col
    explicit_re = re.compile(r'turn\s+(left|right|around)', re.IGNORECASE)
    sents_dist = col.Counter()
    explicit_counts = []
    words_list = []
    stop_count = 0
    room_trans_re = re.compile(r'(exit|enter|into|through|out of|walk into)', re.IGNORECASE)
    room_trans_count = 0

    for ep in out_episodes:
        t = ep.get("instruction", {}).get("instruction_text", "")
        sents = len([s for s in re.split(r'[.!?]+', t.strip()) if s.strip()])
        sents_dist[sents] += 1
        explicit_counts.append(len(explicit_re.findall(t)))
        words_list.append(len(t.split()))
        if stop_re.search(t):
            stop_count += 1
        if room_trans_re.search(t):
            room_trans_count += 1

    n = len(out_episodes)

    lm_re = re.compile(r'\b(at|near|past|by)\s+the\s+\w', re.IGNORECASE)
    instr_texts_all = [ep.get("instruction", {}).get("instruction_text", "") for ep in out_episodes]
    lm_counts = [len(lm_re.findall(t)) for t in instr_texts_all]
    unique_instr = len(set(instr_texts_all))

    print("\nQuality report (GT benchmarks in parentheses):")
    print(f"  Avg words: {sum(words_list)/n:.1f} (GT: 26.8)")
    print(f"  Avg sentences: {sum(sents_dist[k]*k for k in sents_dist)/n:.2f} (GT: 2.51)")
    print(f"  Sentence dist: " + str({k: f"{v/n*100:.1f}%" for k, v in sorted(sents_dist.items())}))
    print(f"  Avg explicit turns: {sum(explicit_counts)/n:.3f} (GT: 0.587)")
    print(f"  pct_zero explicit: {explicit_counts.count(0)/n*100:.1f}% (GT: 58.7%)")
    print(f"  Has stop condition: {stop_count/n*100:.1f}% (GT: ~85%)")
    print(f"  Has room transition: {room_trans_count/n*100:.1f}% (GT: 62.6%)")
    print(f"  Landmark density: {sum(lm_counts)/n:.2f} lm/ep (GT=0.57, v35=1.43, v38=1.07) *** v40 KEY METRIC ***")
    print(f"  Unique instructions: {unique_instr}/{n} ({unique_instr/n*100:.1f}%) (GT=100%, v35=78%, v38=83%, v39=1.97)")

    # Opener analysis
    from collections import Counter as _Counter
    actual_openers = _Counter()
    turn_opener_count = 0
    proceed_count = 0
    traverse_count = 0
    for ep in out_episodes:
        t = ep.get("instruction", {}).get("instruction_text", "")
        first_word = t.strip().split()[0].lower().rstrip(',').rstrip('.') if t.strip() else "?"
        actual_openers[first_word] += 1
        if first_word == "turn":
            turn_opener_count += 1
        if first_word in ("proceed", "traverse", "cross", "pass"):
            proceed_count += 1
    proceed_re_all = re.compile(r'\b(proceed|traverse)\b', re.IGNORECASE)
    traverse_in_eps = sum(1 for ep in out_episodes if proceed_re_all.search(ep.get("instruction", {}).get("instruction_text", "")))
    print("\nActual opener (first word) distribution:")
    for w, c in actual_openers.most_common(10):
        print(f"  {w:15s}: {c:4d} ({c/n*100:.1f}%) [GT: walk=34%, go=18.5%, turn=16.5%, exit=10.7%, leave=3.9%]")
    print(f"\n  v40 key metrics:")
    print(f"    turn_opener: {turn_opener_count}/{n} ({turn_opener_count/n*100:.1f}%) — GT=16.5%, v35=15.8% ✓")
    print(f"    non-GT openers (proceed/traverse/cross/pass): {proceed_count}/{n} ({proceed_count/n*100:.1f}%) — target=0%")
    print(f"    proceed/traverse anywhere: {traverse_in_eps}/{n} ({traverse_in_eps/n*100:.1f}%) — target=0%")

    # Head-verb analysis (key v26 metric — target: near 0% vs GT 1.5%)
    head_re = re.compile(r'\bhead\b', re.IGNORECASE)
    head_in_eps = sum(1 for ep in out_episodes if head_re.search(ep.get("instruction", {}).get("instruction_text", "")))
    print(f"\n  head% in instructions: {head_in_eps/n*100:.1f}% (GT: ~1.5%, v30: 0.3%, target: <3%)")

    print("\nSample instructions:")
    import random
    random.seed(42)
    for ep in random.sample(out_episodes, 5):
        print(f"  {ep['instruction']['instruction_text']}")


if __name__ == "__main__":
    main()
