#!/usr/bin/env python3
"""
Gate3-Gemma v11 train — same approach as v11 val_unseen
MP3D .house room labels + calibrated explicit turn rate (>75° threshold)

Key difference from val_unseen v11:
- No Phase1 rendered-frame descriptions (not available for train)
- No gate3 perframe landmarks (not available for train)
- Uses MP3D .house for room labels at every waypoint (SAME breakthrough)
- Uses >75° threshold for explicit turn direction (SAME calibration)
- Stop condition from goal waypoint room name only (no landmark VLM data)

Predicted quality: slightly below val_unseen v11 (no visual landmarks)
but same room accuracy and turn calibration — much better than v8/v9 for train.

Usage: python3 run_gate3_gemma_v14_train.py
"""
import asyncio
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

TRAIN_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/train/train.json.gz"
VOCAB_SOURCE_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v24.json.gz"
MP3D_DIR = Path("/mnt/nvme0/vln_habitat/habitat_data/scene_datasets/mp3d")
CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v14_train_checkpoint.json"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "train_gate3_gemma_v14.json.gz"

VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
CONCURRENCY = 20
TEMPERATURE = 0.3

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


def _is_key_turn(prim: dict) -> bool:
    return prim["type"] in ("left_turn", "right_turn") and prim.get("angle_deg", 0) > 45.0


def _n_target_sentences(n_key_turns: int, episode_id: int) -> int:
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


def build_v11_seen_prompt(
    start_room: str,
    waypoints_rooms: List[str],
    turn_directions: List[str],
    turn_angles: List[float],
    goal_room: str,
    n_target: int,
) -> str:
    """Build prompt for train — same as v11 but without Phase1 landmarks.
    Uses richer room description style to compensate for missing visual landmarks."""
    SHARP_TURN_THRESHOLD = 79.0

    path_lines = [f"Start: {start_room}"]

    for i, (direction, angle) in enumerate(zip(turn_directions, turn_angles)):
        dest_room = waypoints_rooms[i + 1] if i + 1 < len(waypoints_rooms) else goal_room
        is_sharp = angle > SHARP_TURN_THRESHOLD
        from_room = waypoints_rooms[i] if i < len(waypoints_rooms) else start_room

        if is_sharp:
            dest_str = f" into {dest_room}" if dest_room != from_room else ""
            path_lines.append(f"  Waypoint {i+1}: turn {direction}{dest_str}")
        else:
            dest_str = f"walk into {dest_room}" if dest_room != from_room else "continue forward"
            path_lines.append(f"  Waypoint {i+1}: {dest_str}")

    path_lines.append(f"Goal: {goal_room}")
    path_summary = "\n".join(path_lines)

    # Longer, richer examples to compensate for missing Phase1 visual landmarks
    examples = {
        1: [
            "Exit the bedroom through the open doorway and stop near the wall at the end of the hallway.",
            "Walk through the spacious dining room past the large table and wait near the sofa by the window.",
            "Go up the carpeted stairs to the upper landing and stop near the railing at the top.",
        ],
        2: [
            "Exit the bedroom and turn left into the wide hallway. Stop near the large rug on the floor.",
            "Walk through the kitchen and into the bright dining room. Wait near the wooden table at the far end.",
            "Go up the stairs and turn left at the top landing. Wait near the closed door straight ahead.",
        ],
        3: [
            "Walk out of the living room and turn right through the hallway opening. Continue through the doorway into the dining room. Stop near the table against the far wall.",
            "Exit the bedroom into the narrow hallway. Walk straight past the bathroom door into the kitchen. Wait near the counter along the far wall.",
            "Go up the carpeted stairs to the second floor. Turn left at the top into the long hallway. Walk into the bedroom on the right and stop near the window.",
        ],
        4: [
            "Exit the bedroom and turn right through the hallway. Walk past the bathroom door into the open living room. Continue through the dining area. Stop near the table at the far end.",
            "Walk out of the kitchen into the hallway. Go up the stairs and turn right at the top landing. Walk into the first bedroom. Wait near the closed door.",
            "Exit the bathroom and walk straight through the hallway. Turn left through the opening into the kitchen. Continue past the counter toward the window. Stop near the far wall.",
        ],
    }.get(n_target, [
        "Walk through the rooms along the path and stop near the destination.",
    ])

    examples_str = "\n".join(f'"{e}"' for e in examples)

    return f"""You are writing indoor navigation instructions for a robot.
Use these room names exactly as given. Do NOT change them.
Style: action verbs (exit, walk into, enter, turn, stop near, wait near), descriptive sentences (12-22 words each), include spatial details.

Path context:
{path_summary}

Write EXACTLY {n_target} navigation sentence{"s" if n_target > 1 else ""} following this style:
{examples_str}

IMPORTANT: The last sentence must end with "Stop near" or "Wait near" or "stop in" or "wait in".
Include descriptive spatial details in each sentence. Write ONLY the instruction. No extra text."""


def postprocess(raw: str, n_target: int, goal_room: str) -> str:
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
    if not stop_re.search(result):
        if goal_room and goal_room not in ('room', 'unknown'):
            result = result.rstrip('.!?') + f". Stop in the {goal_room}."
        else:
            result = result.rstrip('.!?') + ". Stop at the destination."
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

    print("Loading train data...")

    with gzip.open(TRAIN_PATH, "rt") as f:
        raw_data = json.load(f)
    episodes = raw_data["episodes"]
    print(f"  train episodes: {len(episodes)}")

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

        regions = parse_house_regions(scene_id)
        waypoint_rooms = [find_room_at(pos, regions) for pos in ep["reference_path"]]

        pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        prims = pa.get("primitives", [])

        key_turn_prims = [p for p in prims if _is_key_turn(p)]
        n_key_turns = len(key_turn_prims)
        n_target = _n_target_sentences(n_key_turns, eid)

        all_turn_directions = []
        all_turn_angles = []
        wp_idx = 0
        for p in prims:
            if p["type"] == "straight":
                wp_idx += min(1, int(p.get("distance_m", 1) / 1.0))
            elif p["type"] in ("left_turn", "right_turn"):
                direction = "left" if p["type"] == "left_turn" else "right"
                angle = p.get("angle_deg", 90.0)
                all_turn_directions.append(direction)
                all_turn_angles.append(angle)
                wp_idx += 1

        start_room = waypoint_rooms[0] if waypoint_rooms else "room"
        goal_room = waypoint_rooms[-1] if waypoint_rooms else "room"

        prompt = build_v11_seen_prompt(
            start_room=start_room,
            waypoints_rooms=waypoint_rooms,
            turn_directions=all_turn_directions,
            turn_angles=all_turn_angles,
            goal_room=goal_room,
            n_target=n_target,
        )

        tasks.append({"episode_id": eid, "prompt": prompt})
        ep_meta[eid] = {
            "n_target": n_target,
            "goal_room": goal_room,
        }

    print(f"Total tasks: {len(tasks)}")

    from collections import Counter
    dist = Counter(ep_meta[ep["episode_id"]]["n_target"] for ep in episodes)
    total = len(episodes)
    print("Projected sentence distribution:")
    for k in sorted(dist):
        print(f"  {k}s: {dist[k]} ({dist[k]/total*100:.1f}%)")

    results = asyncio.run(generate_all(tasks, checkpoint))

    print("\nAssembling output dataset...")
    out_episodes = []
    stop_re = re.compile(r'\b(stop|wait|halt|stand)\b', re.IGNORECASE)

    for ep in episodes:
        eid = ep["episode_id"]
        raw = results.get(str(eid), "")
        meta = ep_meta.get(eid, {})
        n_target = meta.get("n_target", 2)
        goal_room = meta.get("goal_room", "room")

        instruction_text = postprocess(raw, n_target, goal_room) if raw \
            else f"Walk to the {goal_room} and stop."

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

    import collections as col
    explicit_re = re.compile(r'turn\s+(left|right|around)', re.IGNORECASE)
    sents_dist = col.Counter()
    explicit_counts = []
    words_list = []
    stop_count = 0

    for ep in out_episodes:
        t = ep.get("instruction", {}).get("instruction_text", "")
        sents = len([s for s in re.split(r'[.!?]+', t.strip()) if s.strip()])
        sents_dist[sents] += 1
        explicit_counts.append(len(explicit_re.findall(t)))
        words_list.append(len(t.split()))
        if stop_re.search(t): stop_count += 1

    n = len(out_episodes)
    print("\nQuality report (GT benchmarks in parentheses):")
    print(f"  Avg words: {sum(words_list)/n:.1f} (GT: 26.8)")
    print(f"  Avg explicit turns: {sum(explicit_counts)/n:.3f} (GT: 0.660)")
    print(f"  pct_zero explicit: {explicit_counts.count(0)/n*100:.1f}% (GT: 55.0%)")
    print(f"  Has stop condition: {stop_count/n*100:.1f}%")
    print(f"  Sentence dist: " + str({k: f"{v/n*100:.1f}%" for k, v in sorted(sents_dist.items())}))

    print("\nSample instructions:")
    import random; random.seed(42)
    for ep in random.sample(out_episodes, 5):
        print(f"  {ep['instruction']['instruction_text']}")


if __name__ == "__main__":
    main()
