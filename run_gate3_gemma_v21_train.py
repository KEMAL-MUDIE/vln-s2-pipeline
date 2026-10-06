#!/usr/bin/env python3
"""
Gate3-Gemma v21 — v20 + 3-sentence room-flow for long gentle paths (>=3 transitions)

v20 achieves avg_sentences=1.95 (GT: 2.51). Gap remains 0.56.
The distribution: n=1:18%, n=2:70%, n=3:11%, n=4:1%.
GT distribution: ~n=1:19%, n=2:34%, n=3:31%, n=4:12%.
v20 has too many n=2 and too few n=3.

v21 fix: For paths with n_key_turns==0 AND room_transitions>=3: n_target=3 (not 2).
These 3-sentence paths use ROOM-FLOW language only (no "turn left/right").
Sentence 1: start room → first third; sentence 2: middle rooms; sentence 3: final → stop.

Impact analysis (val_unseen, 1839 eps):
  - ~485 episodes (26.4%) switch from n=2 → n=3 (3-sentence room-flow)
  - avg_sentences: 1.95 → ~2.19 (GT: 2.51, further improvement)
  - pct_zero: maintained ~58.7% (no new "turn left/right" added)
  - avg_words: ~26-28 (three short sentences)

Calibration targets:
  - pct_zero: ~58-59% (GT: 58.7%)
  - avg_explicit: ~0.54 (GT: 0.587)
  - avg_words: ~25-28 (GT: 26.8)
  - avg_sentences: ~2.19 (GT: 2.51, best yet)

SR history: GT=63.77%, v20 pred 69-77%, v21 pred 70-78% (even better spatial structure)

Usage: python3 run_gate3_gemma_v21.py
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

VAL_UNSEEN_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/train/train.json.gz"
VOCAB_SOURCE_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v24.json.gz"
MP3D_DIR = Path("/mnt/nvme0/vln_habitat/habitat_data/scene_datasets/mp3d")
PHASE1_CHECKPOINT = ROOT / "outputs" / "gate4_v19_phase1_checkpoint.json"
PHASE1_V73_CHECKPOINT = ROOT / "outputs" / "gate4_v73_approach_views_p1_checkpoint.json"  # turn landmark fallback
PHASE1_V74_CHECKPOINT = ROOT / "outputs" / "gate4_v74_goal_approach_p1_checkpoint.json"   # goal approach (primary stop)
GATE3_PERFRAME_DIR = ROOT / "outputs" / "gate3_perframe"
CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v21_train_checkpoint.json"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "train_gate3_gemma_v21.json.gz"

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


SHARP_TURN_THRESHOLD_GLOBAL = 81.0  # unified: same for sentence count AND explicit language
ROOM_FLOW_SPLIT_THRESHOLD = 2    # min room transitions for 2-sentence room-flow
ROOM_FLOW_3S_THRESHOLD = 3       # min room transitions for 3-sentence room-flow (v21)

def _is_key_turn(prim: dict) -> bool:
    return prim["type"] in ("left_turn", "right_turn") and prim.get("angle_deg", 0) > SHARP_TURN_THRESHOLD_GLOBAL


def _count_room_transitions(waypoint_rooms: List[str]) -> int:
    """Count how many times the room changes consecutively along the path."""
    if len(waypoint_rooms) < 2:
        return 0
    return sum(1 for j in range(1, len(waypoint_rooms)) if waypoint_rooms[j] != waypoint_rooms[j-1])


def _n_target_sentences(n_key_turns: int, episode_id: int, room_transitions: int = 0) -> int:
    """
    v21: n_target based on both explicit turns AND room transitions.
    - n_key_turns==0 AND room_transitions>=3: force n_target=3 (3-sentence room-flow)
    - n_key_turns==0 AND room_transitions==2: force n_target=2 (2-sentence room-flow)
    - n_key_turns==0 AND room_transitions<=1: n_target=1 (single sentence)
    - Otherwise: use turn-based calibration as before
    GT distribution: 1s=18.9%, 2s=33.8%, 3s=31.0%, 4s=11.7%
    """
    if n_key_turns == 0:
        if room_transitions >= ROOM_FLOW_3S_THRESHOLD:
            return 3  # v21: long gentle paths get 3-sentence coverage
        elif room_transitions >= ROOM_FLOW_SPLIT_THRESHOLD:
            return 2  # multi-room gentle paths get 2-sentence coverage (v20)
        else:
            return 1  # very short or same-room paths: 1 sentence
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


def build_v18_prompt(
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
    has_explicit_turns: bool = True,  # v20: False when n_target=2 from room-flow split
) -> str:
    """
    Build prompt using MP3D-accurate room labels for GT-style navigation instructions.
    v20: room-flow split — n_target=2 for gentle multi-room paths (no explicit turns).
    Total word budget (not per-sentence) reduces runaway instruction lengths.
    """
    SHARP_TURN_THRESHOLD = SHARP_TURN_THRESHOLD_GLOBAL

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

    # v21: four distinct prompt styles:
    # (a) n_target=1: short path (<=1 room transition), 1 sentence room-flow
    # (b) n_target=2, no explicit turns: 2-sentence room-flow split (v20)
    # (c) n_target=3, no explicit turns: 3-sentence room-flow split (v21 NEW)
    # (d) n_target=2+, has_explicit_turns=True: turn-containing examples
    # GT word targets: 0-turn=25.7w, 1-turn=27.0w, 2-turn=30.9w, 3+turn=35+w

    if n_target == 1:
        # Very short path: 0 room transitions or just 1, no explicit turns
        # 1 sentence covering start → goal
        examples = [
            "Exit the bedroom, walk through the hallway, and stop near the framed painting on the wall.",
            "Leave the kitchen and walk into the dining room, stopping near the glass cabinet by the door.",
            "Go up the stairs and walk into the bedroom, stopping near the white double doors.",
            "Exit the bathroom and walk through the entryway, stopping near the potted plant.",
        ]
        no_turn_directive = "\nIMPORTANT: Do NOT use 'turn left' or 'turn right'. MUST mention every room listed in Path context."
        word_target = "16-24 words"
    elif n_target == 2 and not has_explicit_turns:
        # Multi-room gentle path: 2 room-flow sentences, no "turn left/right"
        # Split path at midpoint — sentence 1 covers start half, sentence 2 covers end half + stop
        examples = [
            "Exit the bedroom and walk through the hallway into the living room. Continue past the dining room and stop near the wooden coffee table.",
            "Leave the kitchen and walk through the entryway into the hallway. Continue into the bedroom and wait near the glass cabinet.",
            "Exit the bathroom and walk through the hallway into the living room. Go through the dining room and stop near the large wooden bookshelf.",
            "Walk out of the office and through the hallway into the bedroom. Continue into the closet and stop near the white wardrobe.",
        ]
        no_turn_directive = "\nIMPORTANT: Do NOT use 'turn left' or 'turn right'. MUST mention every room listed in Path context. Write 2 SHORT sentences — first covers path start, second covers path end and stop."
        word_target = "20-28 words"
    elif n_target == 3 and not has_explicit_turns:
        # Long gentle path (>=3 room transitions): 3 room-flow sentences (v21 NEW)
        # Sentence 1: start rooms; sentence 2: middle rooms; sentence 3: final rooms + stop
        examples = [
            "Exit the bedroom and walk through the hallway into the kitchen. Continue into the dining room and walk past the living room. Stop near the wooden sofa by the window.",
            "Leave the bathroom and walk through the entryway into the hallway. Continue past the bedroom into the kitchen. Wait near the wooden kitchen counter.",
            "Exit the living room and walk through the hallway into the bedroom. Continue through the closet and into the bathroom. Stop near the white ceramic bathtub.",
            "Walk out of the office and through the hallway into the living room. Continue past the dining room and into the entryway. Stop near the large wooden front door.",
        ]
        no_turn_directive = "\nIMPORTANT: Do NOT use 'turn left' or 'turn right'. MUST mention every room listed in Path context. Write 3 SHORT sentences — each covers one third of the path, with the last ending at the stop target."
        word_target = "24-36 words"
    else:
        no_turn_directive = ""
        word_target = {2: "24-33 words", 3: "28-38 words", 4: "33-45 words"}.get(n_target, "24-38 words")
        examples = {
            2: [
                "Exit the bedroom and turn left into the hallway at the wooden pillar. Walk into the living room and stop near the wooden rug.",
                "Walk through the kitchen and turn right at the counter. Continue into the dining room and wait near the table.",
                "Go up the stairs and turn left at the landing railing. Walk into the hallway and wait near the wooden door.",
                "Exit the bathroom, walk into the hallway, and turn right at the white pillar. Stop near the door frame.",
            ],
            3: [
                "Walk out of the living room and turn right at the hallway. Turn left past the grey sofa and walk into the dining room. Stop near the wooden table.",
                "Exit the bedroom and walk into the kitchen. Turn left at the counter and walk into the dining room. Wait near the glass stool.",
                "Go up the stairs and turn left at the landing. Walk through the hallway and turn right at the wooden door. Stop near the window.",
                "Exit the closet and turn right through the bedroom. Walk into the bathroom and turn left at the dresser. Stop near the bathtub.",
            ],
            4: [
                "Exit the bedroom and turn right at the hallway. Walk through the living room and turn left at the archway. Continue into the dining room and stop near the table.",
                "Walk out of the kitchen and turn right at the stairs. Go up and turn right into the hallway. Walk into the bedroom and wait near the wooden door.",
                "Exit the bathroom and turn left at the hallway. Walk past the window and turn left into the kitchen. Continue past the counter and stop near the wooden table.",
                "Walk out of the living room and turn left at the wall. Go up the stairs, turn right at the landing, and walk into the bedroom. Stop near the bed.",
            ],
        }.get(n_target, ["Walk through the rooms following the path and stop near the destination."])

    examples_str = "\n".join(f'"{e}"' for e in examples)

    return f"""You are writing indoor navigation instructions for a robot.
Use these room names exactly as given. Do NOT change them.
Style: varied action verbs (exit, walk into, go past, turn, stop near, wait near). Target {word_target} total.{no_turn_directive}

Path context:
{path_summary}

Write EXACTLY {n_target} navigation sentence{"s" if n_target > 1 else ""} following this style:
{examples_str}

CRITICAL: The LAST sentence MUST end with exactly: "{stop_phrase}"
Write ONLY the instruction. No extra text."""


def postprocess_v15(raw: str, n_target: int, goal_room: str, goal_landmark: str) -> str:
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

        # v20: count room transitions for room-flow split
        room_transitions = _count_room_transitions(waypoint_rooms)
        n_target = _n_target_sentences(n_key_turns, eid, room_transitions=room_transitions)
        has_explicit_turns = (n_key_turns > 0)

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

        prompt = build_v18_prompt(
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
            has_explicit_turns=has_explicit_turns,
        )

        tasks.append({"episode_id": eid, "prompt": prompt})
        ep_meta[eid] = {
            "n_key_turns": n_key_turns,
            "n_target": n_target,
            "goal_room": goal_room,
            "goal_landmark": goal_landmark,
            "room_transitions": room_transitions,
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

        instruction_text = postprocess_v15(raw, n_target, goal_room, goal_landmark) if raw \
            else f"Walk to the {goal_room} and stop near the {goal_landmark}."  # postprocess unchanged from v15

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
    print(f"  Avg explicit turns: {sum(explicit_counts)/n:.3f} (GT: 0.587)")
    print(f"  pct_zero explicit: {explicit_counts.count(0)/n*100:.1f}% (GT: 58.7%)")
    print(f"  Has stop condition: {stop_count/n*100:.1f}% (GT: ~85%)")
    print(f"  Has room transition: {room_trans_count/n*100:.1f}% (GT: 62.6%)")

    # Sample 5 instructions
    print("\nSample instructions:")
    import random; random.seed(42)
    for ep in random.sample(out_episodes, 5):
        print(f"  {ep['instruction']['instruction_text']}")


if __name__ == "__main__":
    main()
