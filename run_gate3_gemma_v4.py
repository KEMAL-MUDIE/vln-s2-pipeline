#!/usr/bin/env python3
"""
Gate3-Informed Gemma Annotation v4 — refined init_turn threshold + triplet diversity.

Key improvements over v3:
- Skip init_turn context for angles 30-45° (GT annotators often skip these too)
  → Increases pct_zero_explicit from ~17.5% toward ~30%
- Per-triplet temperature variation: same trajectory gets 3 different instructions
  (temp 0.3 / 0.5 / 0.7 for 1st/2nd/3rd in triplet)
- Improved turn priority: prefer turns that cross into NEW rooms over same-room turns

Expected: avg_explicit_turns ~0.7-0.8, pct_zero_explicit ~25-35%, more diverse.

Output: outputs/datasets/val_unseen_gate3_gemma_v4.json.gz
Checkpoint: outputs/gate3_gemma_v4_checkpoint.json
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

GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_gate3_gemma_v4.json.gz"
CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v4_checkpoint.json"
PERFRAME_DIR = ROOT / "outputs" / "gate3_perframe"
LANDMARK_DIR = ROOT / "outputs" / "gate3_landmarks"

from gate2_path.path_analyzer import analyze_path
from openai import AsyncOpenAI
from gate4_instructions.gemma_vllm_backend import (
    generate_batch_async, build_gate3_messages,
    VLLM_BASE_URL, VLLM_MODEL, MAX_NEW_TOKENS
)
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset

# Init-turn angle threshold: below this angle, skip explicit init_turn in context.
# GT annotators skip init_turn for angles 30-45° (visual context suffices).
INIT_TURN_CONTEXT_THRESHOLD = 45.0


def _is_distinctive_landmark(lm: str) -> bool:
    if not lm:
        return False
    generic = {"none visible", "none", "wall", "floor", "ceiling", "door", "window", "light"}
    lm_lower = lm.lower().strip()
    if lm_lower in generic:
        return False
    if len(lm_lower) < 4:
        return False
    return True


def _has_new_room(pf_turn: dict, prev_rooms: set) -> bool:
    """True if this turn enters a room different from any seen so far."""
    r = (pf_turn.get("room") or "").lower()
    if not r or r in prev_rooms:
        return False
    return True


def _is_key_turn_v4(pf_turn: dict, prim: dict, prev_rooms: set) -> Tuple[bool, int]:
    """
    Returns (is_key, priority). Priority: higher = more important.
    - Room transition into new room: priority 3
    - Room transition (same room): priority 2
    - Distinctive landmark + sharp angle: priority 1
    - Otherwise: priority 0
    """
    room_trans = pf_turn.get("room_transition", "").lower()
    is_new_room = _has_new_room(pf_turn, prev_rooms)
    has_trans = room_trans and "none" not in room_trans

    if has_trans and is_new_room:
        return True, 3
    if has_trans:
        return True, 2
    lm = pf_turn.get("main_landmark", "")
    if _is_distinctive_landmark(lm) and prim.get("angle_deg", 0) >= 45:
        return True, 1
    return False, 0


def build_room_sequence(start_room: str, turns_pf: list, goal_room: str) -> str:
    rooms = [start_room]
    for t in turns_pf:
        r = (t.get("room") or "").lower().strip()
        if r and r != rooms[-1] and r != "none":
            rooms.append(r)
    if goal_room and goal_room != rooms[-1] and goal_room != "none":
        rooms.append(goal_room)
    deduped = [rooms[0]]
    for r in rooms[1:]:
        if r != deduped[-1]:
            deduped.append(r)
    if len(deduped) == 1:
        return deduped[0]
    return " → ".join(deduped)


def build_context_v4(episode_id: int, ep: dict, perframe: dict, landmark: dict) -> Optional[dict]:
    """v4 context builder with improved init_turn threshold and turn priority."""
    pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
    summary = pa["summary"]
    turn_prims = [p for p in pa["primitives"] if p["type"] in ("left_turn", "right_turn")]
    initial_turn = pa.get("initial_turn")

    start = perframe.get("start", {})
    turns_pf = perframe.get("turns", [])
    goal = perframe.get("goal", {})
    lm_goal = landmark.get("goal_landmark", {})

    start_room = (start.get("room") or "room").lower()
    start_lm = start.get("main_landmark", "")

    goal_room = (goal.get("room") or lm_goal.get("room_type") or "room").lower()
    stop_lm = (goal.get("stop_landmark") or lm_goal.get("stop_landmark") or "").strip()
    stop_desc = (goal.get("stop_description") or lm_goal.get("direction_hint") or "").strip()

    total_dist = summary["total_distance_m"]
    elevation = summary["elevation_change_m"]

    context_lines = []

    # Initial facing — only include if angle >= 45° (skip small rotations)
    if initial_turn and initial_turn["angle_deg"] >= INIT_TURN_CONTEXT_THRESHOLD:
        it_dir = initial_turn["direction"]
        it_deg = initial_turn["angle_deg"]
        if initial_turn.get("is_around"):
            context_lines.append(f"Initial facing: AROUND (~180°)")
        elif it_deg > 90:
            context_lines.append(f"Initial facing: sharp {it_dir} (~{it_deg:.0f}°)")
        else:
            context_lines.append(f"Initial facing: {it_dir} (~{it_deg:.0f}°)")

    # Room sequence
    room_seq = build_room_sequence(start_room, turns_pf, goal_room)
    context_lines.append(f"Room sequence: {room_seq}")

    # Start landmark
    start_line = f"Start: {start_room}"
    if _is_distinctive_landmark(start_lm):
        start_line += f", near the {start_lm}"
    context_lines.append(start_line)

    # Distance
    dist_str = f"Distance: ~{total_dist:.1f}m"
    if elevation > 0.5:
        dist_str += f" (goes UP {elevation:.1f}m, stairs)"
    elif elevation < -0.5:
        dist_str += f" (goes DOWN {abs(elevation):.1f}m, stairs)"
    context_lines.append(dist_str)

    # Select KEY turns with priority — prefer room-crossing turns, max 2
    turn_candidates = []
    seen_rooms = {start_room}
    for i, prim in enumerate(turn_prims):
        pf_t = turns_pf[i] if i < len(turns_pf) else {}
        is_key, priority = _is_key_turn_v4(pf_t, prim, seen_rooms)
        if is_key:
            direction = "left" if prim["type"] == "left_turn" else "right"
            t_room = (pf_t.get("room") or "").lower()
            t_lm = pf_t.get("main_landmark", "")
            t_trans = pf_t.get("room_transition", "")
            turn_candidates.append({
                "priority": priority,
                "direction": direction,
                "room": t_room,
                "landmark": t_lm if _is_distinctive_landmark(t_lm) else "",
                "room_transition": t_trans if t_trans and "none" not in t_trans.lower() else "",
                "is_new_room": _has_new_room(pf_t, seen_rooms),
            })
            if t_room:
                seen_rooms.add(t_room)

    # Sort by priority (highest first), take top 2
    turn_candidates.sort(key=lambda x: -x["priority"])
    key_turns = turn_candidates[:2]

    if not key_turns:
        n_total = len(turn_prims)
        if n_total > 0:
            context_lines.append(f"Navigation: {n_total} turn(s), follow hallway/rooms")
        else:
            context_lines.append("Navigation: walk straight")
    else:
        wp_strs = []
        for t in key_turns:
            if t["room_transition"]:
                wp = t["room_transition"]
                if t["landmark"]:
                    wp += f" near the {t['landmark']}"
                if t["is_new_room"] and t["room"]:
                    wp += f" (into {t['room']})"
            elif t["landmark"]:
                wp = f"turn {t['direction']} at the {t['landmark']}"
                if t["room"]:
                    wp += f" (into {t['room']})"
            elif t["room"]:
                wp = f"enter the {t['room']}"
            else:
                wp = f"turn {t['direction']}"
            wp_strs.append(wp)
        context_lines.append("Key waypoints: " + " | ".join(wp_strs))

    # Goal
    goal_line = f"Goal: {goal_room}"
    if _is_distinctive_landmark(stop_lm):
        goal_line += f", stop near the {stop_lm}"
    if stop_desc:
        goal_line += f". {stop_desc}"
    context_lines.append(goal_line)

    return {
        "episode_id": episode_id,
        "motion_sequence": "\n".join(context_lines),
        "prompt_type": "gate3",
    }


def find_triplets(episodes: list) -> Dict[int, int]:
    """Map episode_id → triplet position (0, 1, 2) for same-trajectory episodes."""
    from collections import defaultdict
    traj_groups: Dict[str, list] = defaultdict(list)
    for ep in episodes:
        path_key = str(ep["reference_path"])
        traj_groups[path_key].append(ep["episode_id"])
    ep_triplet_pos = {}
    for path_key, ep_ids in traj_groups.items():
        for pos, eid in enumerate(ep_ids):
            ep_triplet_pos[eid] = pos % 3
    return ep_triplet_pos


# Temperature per triplet position for diversity
TRIPLET_TEMPS = [0.3, 0.5, 0.7]


def quality_check(text: str) -> tuple:
    words = text.split()
    failures = []
    if len(words) < 8:
        failures.append(f"short({len(words)}w)")
    if len(words) > 70:
        failures.append(f"long({len(words)}w)")
    stop_words = ["stop", "wait", "halt", "stand", "pause"]
    if not any(w in text.lower() for w in stop_words):
        failures.append("no_stop")
    return len(failures) == 0, failures


def clean_output(raw: str) -> str:
    for prefix in ["Instruction:", "Navigation instruction:", "Here is", "Here's", "Answer:", "Sure", "The agent"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].lstrip(":").strip()
    sentences = re.split(r'(?<=[.!?])\s+', raw.strip())
    clean = " ".join(sentences[:4]).strip()
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean


async def generate_with_temp(
    client, episode_id: int, motion_sequence: str, temperature: float, semaphore: asyncio.Semaphore
) -> Dict:
    """Generate with specific temperature for triplet diversity."""
    async with semaphore:
        messages = build_gate3_messages(motion_sequence)
        try:
            resp = await client.chat.completions.create(
                model=VLLM_MODEL, messages=messages,
                max_tokens=MAX_NEW_TOKENS, temperature=temperature,
            )
            text = resp.choices[0].message.content.strip()
            return {"episode_id": episode_id, "text": text, "ok": True}
        except Exception as e:
            return {"episode_id": episode_id, "text": "", "ok": False, "error": str(e)}


async def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Gate3-Informed Gemma Annotation v4 — val_unseen")
    print("v4: init_turn threshold=45°, triplet temperature diversity, priority turns")
    print("=" * 70)

    with gzip.open(GT_PATH, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]
    print(f"  {len(episodes)} episodes loaded")

    # Map triplet positions
    triplet_pos = find_triplets(episodes)
    temp_dist = [0, 0, 0]
    for pos in triplet_pos.values():
        temp_dist[pos % 3] += 1
    print(f"  Triplet positions: pos0={temp_dist[0]}, pos1={temp_dist[1]}, pos2={temp_dist[2]}")

    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        print(f"  Checkpoint: {len(checkpoint)} already generated")

    print("\n[Context] Building v4 contexts...")
    tasks = []
    missing_gate3 = 0
    for ep in episodes:
        eid = str(ep["episode_id"])
        if eid in checkpoint:
            continue
        pf_path = PERFRAME_DIR / f"episode_{int(eid):06d}.json"
        lm_path = LANDMARK_DIR / f"episode_{int(eid):06d}.json"
        perframe = json.loads(pf_path.read_text()) if pf_path.exists() else {}
        landmark = json.loads(lm_path.read_text()) if lm_path.exists() else {}
        if not perframe:
            missing_gate3 += 1
        ctx = build_context_v4(int(eid), ep, perframe, landmark)
        if ctx:
            ctx["temperature"] = TRIPLET_TEMPS[triplet_pos.get(ep["episode_id"], 0)]
            tasks.append(ctx)

    print(f"  {len(tasks)} to generate, {len(checkpoint)} cached, {missing_gate3} missing gate3")

    # Show sample contexts
    print("\n[Samples] First 3 contexts:")
    for task in tasks[:3]:
        print(f"\n  EP{task['episode_id']} (temp={task['temperature']}):")
        for line in task['motion_sequence'].split('\n'):
            print(f"    {line}")

    if tasks:
        print(f"\n[Gemma] Generating with per-triplet temperature variation (concurrency=16)...")
        t0 = time.time()

        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
        sem = asyncio.Semaphore(16)

        coros = [
            generate_with_temp(client, t["episode_id"], t["motion_sequence"], t["temperature"], sem)
            for t in tasks
        ]

        new_results = {}
        errors = 0
        for i, coro in enumerate(asyncio.as_completed(coros)):
            r = await coro
            if r["ok"]:
                new_results[r["episode_id"]] = r["text"]
            else:
                errors += 1
                new_results[r["episode_id"]] = ""
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(tasks)}] ok={len(new_results)-errors} err={errors} "
                      f"rate={rate:.1f}/s ETA={eta/60:.1f}m")

        elapsed = time.time() - t0
        print(f"  Completed {len(new_results)} in {elapsed:.1f}s ({len(new_results)/elapsed:.1f}/s)")

        checkpoint.update({str(k): v for k, v in new_results.items()})
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f)

    all_generated = {int(k): v for k, v in checkpoint.items()}
    ok = sum(1 for v in all_generated.values() if quality_check(clean_output(v))[0])
    print(f"\n[Quality] {ok}/{len(all_generated)} pass ({100*ok/max(1,len(all_generated)):.1f}%)")

    # Sample comparison
    print("\n[Samples] First 5 episodes:")
    for ep in episodes[:5]:
        eid = ep["episode_id"]
        raw = all_generated.get(eid, "MISSING")
        clean = clean_output(raw) if raw != "MISSING" else "MISSING"
        gt_text = ep.get("instruction", {}).get("instruction_text", "")[:80]
        pos = triplet_pos.get(eid, 0)
        temp = TRIPLET_TEMPS[pos]
        print(f"\n  EP{eid} (triplet_pos={pos}, temp={temp}):")
        print(f"    GT:  {gt_text}")
        print(f"    G3v4: {clean}")

    # Assemble dataset
    print("\n[Assemble] Building dataset...")
    tok = VLNTokenizer(GT_PATH)
    assembled = []
    for ep in episodes:
        eid = ep["episode_id"]
        raw = all_generated.get(eid, "")
        text = clean_output(raw) if raw else ""
        if not text:
            continue
        assembled.append(assemble_episode(ep, text, tok))

    # Quality stats
    texts = [e["instruction"]["instruction_text"] for e in assembled]
    def count_explicit(t): return len(re.findall(r'\bturn (?:left|right|around|slightly)\b', t, re.I))
    explicit_turns = [count_explicit(t) for t in texts]
    avg_explicit = sum(explicit_turns) / len(texts)
    avg_words = sum(len(t.split()) for t in texts) / len(texts)
    pct_3plus = sum(1 for t in explicit_turns if t >= 3) / len(texts) * 100
    pct_zero = sum(1 for t in explicit_turns if t == 0) / len(texts) * 100

    print(f"\n[Stats] Gate3-Gemma v4:")
    print(f"  episodes: {len(assembled)}/{len(episodes)}")
    print(f"  avg_words: {avg_words:.1f}  (GT=26.8, v3=25.2)")
    print(f"  avg_explicit_turns: {avg_explicit:.2f}  (GT=0.66, v3=0.83)")
    print(f"  pct_zero_explicit: {pct_zero:.1f}%  (GT=56%, v3=17.5%)")
    print(f"  pct_3plus_turns: {pct_3plus:.1f}%  (GT=3.8%, v3=0.0%)")

    dataset = {
        "episodes": assembled,
        "instruction_vocab": data.get("instruction_vocab", {}),
        "_generation_meta": {
            "mode": "gate3_gemma_v4",
            "model": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
            "split": "val_unseen",
            "n_episodes": len(assembled),
            "n_skipped": len(episodes) - len(assembled),
            "quality_ok": ok,
            "avg_explicit_turns": avg_explicit,
            "avg_words": avg_words,
            "v4_changes": "init_turn threshold=45deg, triplet diversity, priority key-turns",
        },
    }
    save_dataset(dataset, OUTPUT_PATH)

    NVME_PATH = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_gate3_gemma_v4.json.gz")
    NVME_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(NVME_PATH, "wt") as f:
        json.dump(dataset, f)
    print(f"\nSaved:    {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size//1024} KB)")
    print(f"Deployed: {NVME_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
