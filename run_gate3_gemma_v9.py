#!/usr/bin/env python3
"""
Gate3-Gemma v9 — Room-Transition-Focused Instructions + Rendered-Frame Visual Accuracy

Key insight from v8 analysis (35.6% SR vs v24 40.2% SR vs GT 63.5% SR):
- v8 used gate3 perframe METADATA (room dict, landmark dict) → Gemma generates furniture-heavy instructions
- v24 used rendered-frame VLM DESCRIPTIONS → more accurate visual grounding → higher SR
- GT instructions are ACTION-FOCUSED with ROOM TRANSITIONS ("exit the bedroom", "walk into the kitchen")
  NOT furniture-detail-heavy ("grey fabric lounge chair against white wall")

v9 approach:
1. Use v19 Phase1 rendered-frame VLM descriptions as visual context (accurate)
2. Use gate3 perframe for room names and room_transition fields
3. Use path geometry for turn directions (left/right) — always accurate
4. Generate GT-STYLE instructions: short, room-transition-focused, action verbs
5. Calibrate sentence distribution to GT (1s=18.9%, 2s=33.8%, 3s=31.0%, 4s=11.7%)

v8 actual: 35.6% SR (432/1839), furniture-heavy style, pct_zero=43.8%
v9 target:  >40% SR, room-transition style matching GT distribution

SR history: v24=40.2% (best), v213=36.9%, v8=35.6%, v7=?, v6=?
"""
import asyncio
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

VAL_UNSEEN_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz"
VOCAB_SOURCE_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v24.json.gz"
PHASE1_CHECKPOINT = ROOT / "outputs" / "gate4_v19_phase1_checkpoint.json"
GATE3_PERFRAME_DIR = ROOT / "outputs" / "gate3_perframe"
CHECKPOINT_PATH = ROOT / "outputs" / "gate3_gemma_v9_checkpoint.json"
OUTPUT_PATH = ROOT / "outputs" / "datasets" / "val_unseen_gate3_gemma_v9.json.gz"

VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
CONCURRENCY = 20
TEMPERATURE = 0.3

# Sentence target mapping — calibrated to GT distribution
# GT: 1s=18.9%, 2s=33.8%, 3s=31.0%, 4s=11.7%
# Key turns from data: 0kt=18.9%, 1kt=35.5%, 2kt=29.3%, 3kt=12.4%, 4kt=3.8%


def _n_target_sentences(n_key_turns: int, episode_id: int) -> int:
    """Map key turn count to target sentence count matching GT distribution."""
    if n_key_turns == 0:
        return 1
    elif n_key_turns == 1:
        # 35.5% of episodes: target 2s (GT 33.8%)
        return 2
    elif n_key_turns == 2:
        # Mix of 2s and 3s to hit GT 31.0% for 3s
        h = episode_id % 100
        return 3 if h < 87 else 2  # 87% → 3s
    elif n_key_turns == 3:
        h = episode_id % 100
        return 4 if h < 63 else 3  # 63% → 4s
    else:
        return 4  # 4+ turns → always 4s


def _is_key_turn(prim: dict) -> bool:
    """Sharp turn (>45°) counts as key navigation turn."""
    if prim["type"] not in ("left_turn", "right_turn"):
        return False
    return prim.get("angle_deg", 0) > 45.0


def _extract_room_from_desc(desc: str) -> str:
    """Extract room name from Phase1 VLM description."""
    if not desc:
        return ""
    # Look for "Ahead, a X" pattern which tells us what room we're going into
    ahead_match = re.search(r"ahead[,.]?\s+(?:a\s+)?([a-z ]+?)(?:\s+with|\s+that|\.|,)", desc, re.IGNORECASE)
    if ahead_match:
        room_candidate = ahead_match.group(1).strip().lower()
        # Only use if it sounds like a room
        for room in ["living room", "bedroom", "kitchen", "hallway", "hallway", "dining", "bathroom", "corridor", "office", "study", "library", "foyer", "lobby", "entryway", "stairway"]:
            if room in room_candidate:
                return room_candidate
    # Look for "Starting in a X" pattern
    start_match = re.search(r"starting in (?:a\s+)?([a-z ]+?)(?:\s+with|\s+that|\.|,)", desc, re.IGNORECASE)
    if start_match:
        return start_match.group(1).strip().lower()
    return ""


def _shorten_p1_desc(desc: str) -> str:
    """Extract the key landmark from a Phase1 VLM description (one short phrase)."""
    if not desc:
        return ""
    # "Turn at the X. Ahead, a Y" → extract X and Y
    turn_match = re.search(r"Turn at the ([^.]{5,50})\.", desc)
    ahead_match = re.search(r"Ahead[,.]?\s+(?:a\s+)?([a-z][^.]{5,50})", desc, re.IGNORECASE)
    start_match = re.search(r"Starting in ([^.]{5,50})\.", desc)

    parts = []
    if turn_match:
        landmark = turn_match.group(1).strip()
        # Shorten: keep first 5 words max
        words = landmark.split()[:6]
        parts.append(f"at {' '.join(words)}")
    if ahead_match:
        dest = ahead_match.group(1).strip()
        words = dest.split()[:6]
        parts.append(f"ahead: {' '.join(words)}")
    if start_match and not parts:
        ctx = start_match.group(1).strip()
        words = ctx.split()[:6]
        parts.append(' '.join(words))

    return '; '.join(parts) if parts else desc[:80]


def build_v9_prompt(
    start_room: str,
    start_desc: str,
    turns: List[dict],  # [{"direction": "left/right", "label": "turn_k", "p1_desc": str, "room_trans": str, "landmark": str}]
    goal_room: str,
    goal_desc: str,
    goal_landmark: str,
    n_target: int,
    all_p1_turns: Dict[str, str] = None,  # all Phase1 turn labels → descriptions
) -> str:
    """
    Build a prompt that generates GT-style room-transition navigation instructions.
    Uses Phase1 visual descriptions for accuracy, asks for GT action-focused style.
    """
    # Build path summary using Phase1 descriptions when available
    path_lines = []

    start_room_str = start_room or "indoor area"
    # Shorten start description
    start_context = _shorten_p1_desc(start_desc) if start_desc else ""
    if start_context:
        path_lines.append(f"Start: {start_room_str} ({start_context})")
    else:
        path_lines.append(f"Start: {start_room_str}")

    for i, t in enumerate(turns):
        direction = t.get("direction", "")
        room_trans = t.get("room_trans", "").strip()
        landmark = t.get("landmark", "").strip()
        p1_desc = t.get("p1_desc", "").strip()

        # Build turn context - prefer Phase1 description for visual accuracy
        p1_short = _shorten_p1_desc(p1_desc) if p1_desc else ""

        # Extract room from room_trans if informative
        trans_str = ""
        if room_trans and "none" not in room_trans.lower() and "doorway" not in room_trans.lower() and len(room_trans) > 8:
            trans_str = room_trans

        # Build turn line
        turn_parts = []
        if direction:
            turn_parts.append(f"turn {direction}")
        if trans_str:
            turn_parts.append(trans_str)
        if p1_short:
            turn_parts.append(p1_short)
        elif landmark and not trans_str:
            turn_parts.append(f"landmark: {landmark}")

        path_lines.append(f"  Turn {i+1}: {', '.join(turn_parts) if turn_parts else 'continue forward'}")

    # Goal
    goal_room_str = goal_room or "destination"
    goal_p1_short = _shorten_p1_desc(goal_desc) if goal_desc else ""
    if goal_landmark and goal_landmark not in ("the destination", "destination"):
        if goal_p1_short:
            path_lines.append(f"Goal: {goal_room_str}, near {goal_landmark} ({goal_p1_short})")
        else:
            path_lines.append(f"Goal: {goal_room_str}, near {goal_landmark}")
    elif goal_p1_short:
        path_lines.append(f"Goal: {goal_room_str} ({goal_p1_short})")
    else:
        path_lines.append(f"Goal: {goal_room_str}")

    path_summary = "\n".join(path_lines)

    # Examples calibrated to target sentence count
    examples_by_n = {
        1: [
            "Go past the dining table and stop near the couch.",
            "Exit through the hallway and wait near the entrance.",
            "Turn left down the stairs and stop near the landing.",
        ],
        2: [
            "Exit the bedroom and turn left. Walk straight and stop near the rug.",
            "Walk through the hallway and turn right into the kitchen. Stop near the counter.",
            "Go up the stairs and turn left. Wait near the stair landing.",
        ],
        3: [
            "Walk out of the room and turn right in the hallway. Continue straight into the living room. Stop near the couch.",
            "Exit the bathroom through the door. Turn left and walk into the bedroom. Wait near the bed.",
            "Go forward and turn left at the stairs. Head up the steps. Wait near the landing.",
        ],
        4: [
            "Exit the room through the door. Turn left and walk down the hallway. Enter the bedroom at the end. Stop near the window.",
            "Walk out of the bedroom. Turn right into the hallway. Go straight through the living room. Wait near the sliding door.",
            "Go forward and turn left to the stairs. Head up the steps and turn left again. Go up the second set of steps. Wait near the landing.",
        ],
    }
    examples = "\n".join(f'"{e}"' for e in examples_by_n.get(n_target, examples_by_n[3]))

    prompt = f"""You are writing navigation instructions for a robot navigating indoors.
Style rules:
- Use SHORT sentences (5-15 words each)
- Use room transition verbs: "walk into", "enter", "exit", "go through", "head up/down"
- Name rooms and features: bedroom, living room, kitchen, hallway, stairs, bathroom
- Mention turn direction (left/right) at actual turns
- End with "Stop near" or "Wait near" followed by the goal
- Keep description brief — focus on actions and rooms, not furniture colors

Path context:
{path_summary}

Write EXACTLY {n_target} navigation sentence{"s" if n_target > 1 else ""} following this style:
{examples}

Write ONLY the instruction. No extra text."""

    return prompt


def postprocess_v9(raw: str, n_target: int) -> str:
    """Clean and enforce sentence count."""
    text = raw.strip()

    # Strip common prefixes
    for prefix in ["Here is the navigation instruction:", "Instruction:", "Navigation instruction:", "Instructions:", "Sure,", "Certainly,"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # Remove quotes
    text = text.strip('"\'')

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    # Remove sentence-number prefixes "1. 2."
    sentences = [re.sub(r'^\d+\.\s*', '', s) for s in sentences]

    # If too many sentences, truncate
    if len(sentences) > n_target:
        sentences = sentences[:n_target]
        if not sentences[-1].endswith(('.', '!', '?')):
            sentences[-1] += '.'

    # If too few sentences and only 1 found, that's okay for n_target=1
    if len(sentences) == 0:
        return text

    # Ensure ends with period
    result = ' '.join(sentences)
    if result and result[-1] not in '.!?':
        result += '.'

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
            prompt = task["prompt"]
            msgs = [{"role": "user", "content": prompt}]
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
    ckpt_path = CHECKPOINT_PATH

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
            with open(ckpt_path, "w") as f:
                json.dump(results, f)

    with open(ckpt_path, "w") as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    print(f"Done {len(pending)} in {elapsed:.1f}s ({len(pending)/max(elapsed,1):.1f}/s). errors={errors}")
    return results


def main():
    from gate2_path.path_analyzer import analyze_path
    from gate5_tokenizer.tokenizer import VLNTokenizer
    from gate6_assembler.assembler import assemble_episode

    print("Loading data...")

    # Load GT episodes
    with gzip.open(VAL_UNSEEN_PATH, "rt") as f:
        raw_data = json.load(f)
    episodes = raw_data["episodes"]

    # Load Phase1 rendered-frame descriptions
    p1 = json.load(open(PHASE1_CHECKPOINT))
    print(f"  Phase1 descriptions: {len(p1)} episodes")

    # Load gate3 perframe metadata
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

    # Build tasks
    tasks = []
    ep_meta = {}

    for ep in episodes:
        eid = ep["episode_id"]
        pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
        prims = pa.get("primitives", [])

        # Count key turns
        key_turns = [p for p in prims if _is_key_turn(p)]
        n_key_turns = len(key_turns)
        n_target = _n_target_sentences(n_key_turns, eid)

        # Get Phase1 data for this episode
        p1ep = p1.get(str(eid), {})

        # Get gate3 perframe data
        g3ep = g3.get(eid, {})

        # Start room
        start_room = (g3ep.get("start") or {}).get("room", "")
        if not start_room:
            start_room = _extract_room_from_desc(p1ep.get("start", ""))
        start_desc = p1ep.get("start", "")

        # Build turns list — match path geometry directions to Phase1 descriptions
        turn_dirs = []
        for p in prims:
            if p["type"] == "left_turn":
                turn_dirs.append("left")
            elif p["type"] == "right_turn":
                turn_dirs.append("right")

        g3_turns = g3ep.get("turns", [])
        # Build g3 label → data map for quick lookup
        g3_by_label = {t.get("label", ""): t for t in g3_turns}

        # Get all Phase1 turn keys (turn_1, turn_2, ...) in order
        p1_turn_keys = sorted(
            [k for k in p1ep.keys() if k.startswith("turn_")],
            key=lambda x: int(x.split("_")[1])
        )

        turns = []
        for i, direction in enumerate(turn_dirs):
            # Try to get Phase1 description: first from g3 label, then from p1 turn order
            g3t = g3_turns[i] if i < len(g3_turns) else {}
            label = g3t.get("label", "")

            # Phase1 lookup: by gate3 label, then by position in p1_turn_keys
            if label and label in p1ep:
                p1_desc = p1ep[label]
            elif i < len(p1_turn_keys):
                p1_desc = p1ep[p1_turn_keys[i]]
                label = p1_turn_keys[i]
            else:
                p1_desc = ""

            room_trans = g3_by_label.get(label, g3t).get("room_transition", "")
            landmark = g3_by_label.get(label, g3t).get("main_landmark", "")
            turns.append({
                "direction": direction,
                "label": label,
                "p1_desc": p1_desc,
                "room_trans": room_trans,
                "landmark": landmark,
            })

        # Goal
        g3_goal = g3ep.get("goal") or {}
        goal_room = g3_goal.get("room", "")
        goal_landmark = g3_goal.get("stop_landmark", "") or g3_goal.get("main_landmark", "")
        goal_desc = p1ep.get("goal", "")
        if not goal_room:
            goal_room = _extract_room_from_desc(goal_desc)
        if not goal_landmark:
            goal_landmark = "the destination"

        prompt = build_v9_prompt(
            start_room=start_room,
            start_desc=start_desc,
            turns=turns,
            goal_room=goal_room,
            goal_desc=goal_desc,
            goal_landmark=goal_landmark,
            n_target=n_target,
        )

        tasks.append({"episode_id": eid, "prompt": prompt})
        ep_meta[eid] = {
            "n_key_turns": n_key_turns,
            "n_target": n_target,
            "ep": ep,
        }

    print(f"\nTotal tasks: {len(tasks)}")

    # Sentence distribution preview
    from collections import Counter
    dist = Counter(ep_meta[ep["episode_id"]]["n_target"] for ep in episodes)
    total = len(episodes)
    print("Projected sentence distribution:")
    for k in sorted(dist):
        print(f"  {k}s: {dist[k]} ({dist[k]/total*100:.1f}%)")

    # Run generation
    results = asyncio.run(generate_all(tasks, checkpoint))

    # Assemble output dataset
    print("\nAssembling output dataset...")
    out_episodes = []
    tokenizer = VLNTokenizer()

    stop_re = re.compile(r'\b(stop|wait|halt|stand)\b', re.IGNORECASE)

    for ep in episodes:
        eid = ep["episode_id"]
        raw = results.get(str(eid), "")
        meta = ep_meta.get(eid, {})
        n_target = meta.get("n_target", 2)

        instruction_text = postprocess_v9(raw, n_target) if raw else "Walk to the destination and stop."

        # Patch missing stop condition — critical for model to know when to stop
        if not stop_re.search(instruction_text):
            g3ep = g3.get(eid, {})
            g3_goal = g3ep.get("goal") or {}
            goal_landmark = g3_goal.get("stop_landmark", "") or g3_goal.get("main_landmark", "")
            goal_room = g3_goal.get("room", "")
            if goal_landmark and goal_landmark not in ("the destination", "destination"):
                instruction_text = instruction_text.rstrip(".!?") + f". Stop near the {goal_landmark}."
            elif goal_room:
                instruction_text = instruction_text.rstrip(".!?") + f". Stop in the {goal_room}."
            else:
                instruction_text = instruction_text.rstrip(".!?") + ". Stop at the destination."

        assembled = assemble_episode(
            source_episode=ep,
            generated_text=instruction_text,
            tokenizer=tokenizer,
        )
        out_episodes.append(assembled)

    # Load vocab from canonical source
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

    # Quick quality report
    import collections as col
    explicit_re = re.compile(r'turn\s+(left|right|around)', re.IGNORECASE)
    sents_dist = col.Counter()
    explicit_counts = []
    words = []
    for ep in out_episodes:
        t = ep.get("instruction", {}).get("instruction_text", "")
        sents = len([s for s in re.split(r'[.!?]+', t.strip()) if s.strip()])
        sents_dist[sents] += 1
        explicit_counts.append(len(explicit_re.findall(t)))
        words.append(len(t.split()))

    n = len(out_episodes)
    print("\nQuality report:")
    print(f"  Avg words: {sum(words)/n:.1f} (GT: 26.8)")
    print(f"  Avg sentences: {sum(sents_dist[k]*k for k in sents_dist)/n:.2f} (GT: 2.51)")
    print(f"  Sentence dist: " + str({k: f"{v/n*100:.1f}%" for k,v in sorted(sents_dist.items())}))
    print(f"  Avg explicit turns: {sum(explicit_counts)/n:.3f} (GT: 0.660)")
    print(f"  pct_zero explicit: {explicit_counts.count(0)/n*100:.1f}% (GT: 55.0%)")


if __name__ == "__main__":
    main()
