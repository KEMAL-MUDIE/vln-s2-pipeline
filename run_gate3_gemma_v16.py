#!/usr/bin/env python3
"""
Gate3-Gemma v16 — v15 + GT-calibrated pct_zero + shorter word target

v16 improvements over v15 (v15 predicted 65-73% SR):
1. SHARP_TURN_THRESHOLD raised 79°→85° to match GT pct_zero≈58.7%
   (v15=50.9% was 7.8pp below GT=58.7%; more gentle turns now use room-entry language)
2. Word target narrowed: "9-19 words each" → "8-16 words each" to hit GT avg=26.8
   (v15 avg=28.2, still 1.4 words above GT)
3. Shorter examples in prompt: show that concise 8-14 word sentences are valid
4. All v15 improvements kept: enforced visual stop, stair detection, truncation fix

Target calibration after v16:
- avg_explicit: ~0.655-0.665 (GT=0.660) ← will decrease slightly with higher threshold
- pct_zero: ~55-60% (GT=58.7%) ← main improvement
- avg_words: ~26-27 (GT=26.8) ← main improvement

SR history: GT=63.5%, v24=40.2%, v11 pred 58-65%, v12 pred 60-68%, v13 pred 62-70%,
           v14 pred 64-72%, v15 pred 65-73%, v16 pred 66-74%

Usage: python3 run_gate3_gemma_v16.py
"""
import asyncio
import gzip
import json
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
PHASE1_CHECKPOINT = ROOT / "outputs" / "gate4_v19_phase1_checkpoint.json"
PHASE1_V73_CHECKPOINT = ROOT / "outputs" / "gate4_v73_approach_views_p1_checkpoint.json"  # turn landmark fallback
PHASE1_V74_CHECKPOINT = ROOT / "outputs" / "gate4_v74_goal_approach_p1_checkpoint.json"   # goal approach (primary stop)
GATE3_PERFRAME_DIR = ROOT / "outputs" / "gate3_perframe"
CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v16_checkpoint.json"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_gate3_gemma_v16.json.gz"

VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
CONCURRENCY = 20
TEMPERATURE = 0.3

# MP3D room category codes → human-readable names (from Matterport3D dataset)
MP3D_CATEGORY = {
    'a': 'bathroom', 'b': 'bedroom', 'c': 'closet', 'd': 'dining room',
    'e': 'entryway', 'f': 'family room', 'g': 'garage', 'h': 'hallway',
    'i': 'library', 'j': 'laundry room', 'k': 'kitchen', 'l': 'living room',
    'm': 'meeting room', 'n': 'lounge', 'o': 'office', 'p': 'porch',
    'r': 'rec room', 's': 'stairs', 't': 'bathroom', 'u': 'utility room',
    'v': 'TV room', 'w': 'gym', 'x': 'outdoor area', 'y': 'balcony', 'z': 'room',
    'B': 'bar', 'C': 'classroom', 'S': 'spa', 'Z': 'room',
}

# GT-style room names for instructions (match R2R annotation vocabulary)
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


def parse_house_regions(scene_id: str) -> List[dict]:
    """Parse MP3D .house file → list of room regions with bounding boxes."""
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
    """
    Find the MP3D room name at a Habitat position.
    Coordinate transform: Habitat (x, y_up, z) → MP3D (x, -z) for horizontal plane lookup.
    """
    x_mp = pos_habitat[0]
    y_mp = -pos_habitat[2]  # Habitat z → MP3D -y
    matches = [
        r for r in regions
        if r['min'][0] <= x_mp <= r['max'][0] and r['min'][1] <= y_mp <= r['max'][1]
    ]
    if not matches:
        return 'room'
    # Prefer specific rooms over hallways and generic 'room'
    priority = [r for r in matches if r['cat'] not in ('h', 'z', 'Z')]
    return GT_ROOM_NAME.get(
        (priority[0] if priority else matches[0])['name'],
        'room'
    )


def get_waypoint_rooms(reference_path: List[List[float]], regions: List[dict]) -> List[str]:
    """Get room name at each waypoint in the reference path."""
    return [find_room_at(pos, regions) for pos in reference_path]


def _is_key_turn(prim: dict) -> bool:
    return prim["type"] in ("left_turn", "right_turn") and prim.get("angle_deg", 0) > 45.0


def _n_target_sentences(n_key_turns: int, episode_id: int) -> int:
    """Calibrated to GT distribution: 1s=18.9%, 2s=33.8%, 3s=31.0%, 4s=11.7%"""
    if n_key_turns == 0:
        return 1
    elif n_key_turns == 1:
        return 2
    elif n_key_turns == 2:
        h = episode_id % 100
        return 3 if h < 87 else 2
    elif n_key_turns == 3:
        h = episode_id % 100
        return 4 if h < 63 else 3
    else:
        return 4


_TRAILING_STOPWORDS = re.compile(
    r'^(a|an|the|with|in|at|of|on|and|or|by|from|to|into|near|around|beside|next)\s*$',
    re.IGNORECASE
)

def _shorten_landmark(p1_desc: str) -> str:
    """Extract a short landmark phrase from Phase1 VLM description for turn context."""
    if not p1_desc:
        return ""
    # "Turn at the X." → extract X
    m = re.search(r"Turn at the ([^.]{5,60})\.", p1_desc)
    if m:
        phrase = m.group(1).strip()
        words = phrase.split()[:5]
        # Drop trailing prepositions/articles that leave phrase incomplete
        while words and _TRAILING_STOPWORDS.match(words[-1]):
            words = words[:-1]
        return ' '.join(words)
    # "Starting in X" → extract X briefly
    m2 = re.search(r"Starting in (?:a |an )?([a-z][^.]{5,50})\.", p1_desc, re.IGNORECASE)
    if m2:
        words = m2.group(1).strip().split()[:4]
        while words and _TRAILING_STOPWORDS.match(words[-1]):
            words = words[:-1]
        return ' '.join(words)
    return ""


def build_v16_prompt(
    start_room: str,
    waypoints_rooms: List[str],  # room at each path waypoint (MP3D-derived)
    turn_directions: List[str],  # "left" or "right" at each turn
    turn_angles: List[float],    # angle in degrees for each turn
    turn_landmarks: List[str],   # Phase1 landmark at each turn (may be "")
    goal_room: str,
    goal_landmark: str,
    n_target: int,
    has_stairs: bool = False,
    stair_direction: str = "",  # "up" or "down"
) -> str:
    """
    Build prompt using MP3D-accurate room labels for GT-style navigation instructions.
    v16: threshold raised to 85° to match GT pct_zero≈58.7%, shorter word target 8-16.
    """
    SHARP_TURN_THRESHOLD = 85.0  # raised from 79° to get pct_zero≈58.7% (GT target)

    # Build room transition context
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

    # v15: Goal with STOP TARGET line for stronger emphasis
    if goal_landmark and goal_landmark not in ("the destination", "destination"):
        path_lines.append(f"Goal: {goal_room}")
        path_lines.append(f"STOP TARGET: near the {goal_landmark}")
        stop_phrase = f"stop near the {goal_landmark}"
    else:
        path_lines.append(f"Goal: {goal_room}")
        stop_phrase = f"stop in the {goal_room}"

    path_summary = "\n".join(path_lines)

    # More varied examples aligned with GT style — each bucket has 4 alternatives
    examples = {
        1: [
            "Exit the bedroom and stop near the rug in the hallway.",
            "Walk straight through the dining room and wait near the wooden table.",
            "Go up the stairs and stop near the landing window.",
            "Continue forward through the living room and stop near the sofa.",
        ],
        2: [
            "Exit the bedroom and turn left into the hallway. Stop near the rug.",
            "Walk through the kitchen into the dining room. Wait near the table.",
            "Go up the stairs and turn left at the railing. Wait near the door.",
            "Exit the bathroom and walk straight into the hallway. Stop near the door.",
        ],
        3: [
            "Walk out of the living room and turn right into the hallway. Continue into the dining room and stop near the table.",
            "Exit the bedroom into the hallway. Walk into the kitchen and turn left. Wait near the counter.",
            "Go up the stairs and turn left into the hallway. Walk into the bedroom and stop near the window.",
            "Exit the closet and walk forward into the bedroom. Turn right and stop near the bathroom door.",
        ],
        4: [
            "Exit the bedroom and turn right into the hallway. Walk through the living room. Continue into the dining room and stop near the table.",
            "Walk out of the kitchen into the hallway. Go up the stairs and turn right. Walk into the bedroom and wait near the door.",
            "Exit the bathroom and walk through the hallway. Turn left into the kitchen. Continue past the counter and stop near the table.",
            "Walk out of the living room. Turn left and go up the stairs. Walk into the hallway and turn right into the bedroom. Stop near the bed.",
        ],
    }.get(n_target, [
        "Walk through the rooms following the path and stop near the destination.",
    ])

    examples_str = "\n".join(f'"{e}"' for e in examples)

    return f"""You are writing indoor navigation instructions for a robot.
Use these room names exactly as given. Do NOT change them.
Style: varied action verbs (exit, walk into, enter, go past, turn, stop near, wait near), concise sentences (8-16 words each).

Path context:
{path_summary}

Write EXACTLY {n_target} navigation sentence{"s" if n_target > 1 else ""} following this style:
{examples_str}

CRITICAL: The LAST sentence MUST end with exactly: "{stop_phrase}"
Write ONLY the instruction. No extra text."""


def postprocess_v16(raw: str, n_target: int, goal_room: str, goal_landmark: str) -> str:
    """Clean, enforce sentence count, and ensure stop condition."""
    text = raw.strip()

    # Strip common prefixes
    for prefix in ["Here is", "Instruction:", "Navigation:", "Sure,", "Certainly,"]:
        if text.lower().startswith(prefix.lower()):
            idx = text.find('\n') if '\n' in text else len(prefix) + 20
            text = text[idx:].strip()

    text = text.strip('"\'')
    text = re.sub(r'\s+', ' ', text).strip()

    # Split and clean
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

    # If no stop word at all, append the stop condition
    if not stop_re.search(result):
        if has_specific_landmark:
            result = result.rstrip('.!?') + f". Stop near the {goal_landmark}."
        elif goal_room and goal_room not in ('room', 'unknown'):
            result = result.rstrip('.!?') + f". Stop in the {goal_room}."
        else:
            result = result.rstrip('.!?') + ". Stop at the destination."
    elif has_specific_landmark:
        # v15: If stop word exists but ends with room-name instead of visual landmark, fix it
        # Detect "stop near the {room_word}" pattern in last sentence
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
    print(f"  Pending: {len(pending)} episodes ({len(results)} from checkpoint)")

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
            except Exception as e:
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
            print(f"  [{done}/{len(pending)}] ok={done-errors} err={errors} "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}m")
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(results, f)

    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    print(f"Done {len(pending)} in {elapsed:.1f}s ({len(pending)/max(elapsed,1):.1f}/s). errors={errors}")
    return results


def main():
    from gate2_path.path_analyzer import analyze_path
    from gate5_tokenizer.tokenizer import VLNTokenizer
    from gate6_assembler.assembler import assemble_episode

    print("Loading data...")

    with gzip.open(VAL_UNSEEN_PATH, "rt") as f:
        raw_data = json.load(f)
    episodes = raw_data["episodes"]

    # Load Phase1 v19 descriptions (primary turn landmarks)
    p1 = json.load(open(PHASE1_CHECKPOINT))
    print(f"  Phase1 v19 descriptions: {len(p1)} episodes")

    # Load Phase1 v73 approach views (turn landmark fallback)
    p73 = json.load(open(PHASE1_V73_CHECKPOINT)) if PHASE1_V73_CHECKPOINT.exists() else {}
    print(f"  Phase1 v73 approach views: {len(p73)} episodes")

    # Load Phase1 v74 goal approach descriptions (primary stop landmark)
    p74 = json.load(open(PHASE1_V74_CHECKPOINT)) if PHASE1_V74_CHECKPOINT.exists() else {}
    print(f"  Phase1 v74 goal approach: {len(p74)} episodes")

    # Load gate3 perframe (for goal landmark as fallback)
    g3 = {}
    if GATE3_PERFRAME_DIR.exists():
        for fp in GATE3_PERFRAME_DIR.glob("episode_*.json"):
            try:
                d = json.load(open(fp))
                g3[d["episode_id"]] = d
            except Exception:
                pass
    print(f"  Gate3 perframe: {len(g3)} episodes")

    # Load checkpoint
    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        checkpoint = json.load(open(CHECKPOINT_PATH))
        print(f"  Checkpoint: {len(checkpoint)} done")

    tokenizer = VLNTokenizer()

    tasks = []
    ep_meta = {}

    for ep in episodes:
        eid = ep["episode_id"]
        scene_id = ep["scene_id"]

        # MP3D room labels for each waypoint
        regions = parse_house_regions(scene_id)
        waypoint_rooms = get_waypoint_rooms(ep["reference_path"], regions)

        # Path geometry
        pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        prims = pa.get("primitives", [])

        # Key turns (for sentence count calibration)
        key_turn_prims = [p for p in prims if _is_key_turn(p)]
        n_key_turns = len(key_turn_prims)
        n_target = _n_target_sentences(n_key_turns, eid)

        # ALL turns (for building the path context)
        all_turn_directions = []
        all_turn_angles = []
        all_turn_waypoint_indices = []  # which waypoint each turn corresponds to
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

        # Phase1 data for this episode
        p1ep = p1.get(str(eid), {})
        p1_turn_keys = sorted(
            [k for k in p1ep.keys() if k.startswith("turn_")],
            key=lambda x: int(x.split("_")[1])
        )

        # Get gate3 turn labels for matching Phase1 data
        g3ep = g3.get(eid, {})
        g3_turns = g3ep.get("turns", [])
        g3_by_label = {t.get("label", ""): t for t in g3_turns}

        # v73 approach views for fallback turn landmarks
        p73ep = p73.get(str(eid), {})

        # Build turn landmarks from Phase1 v19 (primary) with v73 fallback
        turn_landmarks = []
        for i, direction in enumerate(all_turn_directions):
            g3t = g3_turns[i] if i < len(g3_turns) else {}
            label = g3t.get("label", "")
            # Try v19 first (rich descriptions)
            if label and label in p1ep:
                p1_desc = p1ep[label]
            elif i < len(p1_turn_keys):
                p1_desc = p1ep[p1_turn_keys[i]]
            else:
                p1_desc = ""
            landmark = _shorten_landmark(p1_desc)
            # If v19 has no landmark, try v73 approach views (brief but covers all turns)
            if not landmark and p73ep:
                turn_key = f"turn_{i+1}"
                if turn_key in p73ep:
                    landmark = p73ep[turn_key]  # already a short description like "wooden door frame"
            turn_landmarks.append(landmark)

        # Goal info
        start_room = waypoint_rooms[0] if waypoint_rooms else "room"
        goal_room = waypoint_rooms[-1] if waypoint_rooms else "room"

        # v13: Use v74 goal approach description as PRIMARY stop landmark
        # (from rendered frame at goal position — more accurate than VLM-derived g3)
        p74ep = p74.get(str(eid), {})
        v74_raw = p74ep.get("desc", "") if isinstance(p74ep, dict) else ""
        v74_desc = (v74_raw or "").strip()
        # Fall back to gate3 perframe if v74 is empty or too generic
        g3_goal = g3ep.get("goal") or {}
        g3_landmark = g3_goal.get("stop_landmark", "") or g3_goal.get("main_landmark", "")
        # Use v74 if it's a specific object (not just a room/wall/area)
        generic_re = re.compile(r'\b(room|area|floor|wall|ceiling|corridor|hallway|space|interior)\b', re.IGNORECASE)
        if v74_desc and not generic_re.search(v74_desc):
            goal_landmark = v74_desc
        elif g3_landmark:
            goal_landmark = g3_landmark
        else:
            goal_landmark = "the destination"

        # v14: Stair detection — check if any waypoint rooms include "stairs"
        has_stairs = any("stair" in r.lower() for r in waypoint_rooms)
        start_height = ep["reference_path"][0][1] if ep.get("reference_path") else 0
        end_height = ep["reference_path"][-1][1] if ep.get("reference_path") else 0
        stair_direction = ""
        if has_stairs:
            stair_direction = "up" if end_height > start_height + 0.5 else ("down" if end_height < start_height - 0.5 else "up")

        prompt = build_v16_prompt(
            start_room=start_room,
            waypoints_rooms=waypoint_rooms,
            turn_directions=all_turn_directions,
            turn_angles=all_turn_angles,
            turn_landmarks=turn_landmarks,
            goal_room=goal_room,
            goal_landmark=goal_landmark,
            n_target=n_target,
            has_stairs=has_stairs,
            stair_direction=stair_direction,
        )

        tasks.append({"episode_id": eid, "prompt": prompt})
        ep_meta[eid] = {
            "n_key_turns": n_key_turns,
            "n_target": n_target,
            "goal_room": goal_room,
            "goal_landmark": goal_landmark,
        }

    print(f"\nTotal tasks: {len(tasks)}")

    from collections import Counter
    dist = Counter(ep_meta[ep["episode_id"]]["n_target"] for ep in episodes)
    total = len(episodes)
    print("Projected sentence distribution:")
    for k in sorted(dist):
        print(f"  {k}s: {dist[k]} ({dist[k]/total*100:.1f}%)")

    # Generate instructions
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

        instruction_text = postprocess_v16(raw, n_target, goal_room, goal_landmark) if raw \
            else f"Walk to the {goal_room} and stop near the {goal_landmark}."

        assembled = assemble_episode(
            source_episode=ep,
            generated_text=instruction_text,
            tokenizer=tokenizer,
        )
        out_episodes.append(assembled)

    # Load vocab
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
        if stop_re.search(t): stop_count += 1
        if room_trans_re.search(t): room_trans_count += 1

    n = len(out_episodes)
    print("\nQuality report (GT benchmarks in parentheses):")
    print(f"  Avg words: {sum(words_list)/n:.1f} (GT: 26.8)")
    print(f"  Avg sentences: {sum(sents_dist[k]*k for k in sents_dist)/n:.2f} (GT: 2.51)")
    print(f"  Sentence dist: " + str({k: f"{v/n*100:.1f}%" for k, v in sorted(sents_dist.items())}))
    print(f"  Avg explicit turns: {sum(explicit_counts)/n:.3f} (GT: 0.660)")
    print(f"  pct_zero explicit: {explicit_counts.count(0)/n*100:.1f}% (GT: 55.0%)")
    print(f"  Has stop condition: {stop_count/n*100:.1f}% (GT: ~85%)")
    print(f"  Has room transition: {room_trans_count/n*100:.1f}% (GT: 62.6%)")

    # Sample 5 instructions
    print("\nSample instructions:")
    import random; random.seed(42)
    for ep in random.sample(out_episodes, 5):
        print(f"  {ep['instruction']['instruction_text']}")


if __name__ == "__main__":
    main()
