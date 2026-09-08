#!/usr/bin/env python3
"""
Phase 1 Room-Context Turn Descriptions — v152

Improvement over v150 direction-aware Phase C:
  - v150: VLM knows LEFT/RIGHT turn direction → describes distinctive object at turn point
  - v152: VLM ALSO knows room transition context → prefers room/doorway descriptions (GT-style)

Key insight: GT instructions use room names and doorways as turn anchors (hallway, kitchen, bathroom)
NOT specific objects (brown leather sofa, white wall niche). This matches how people navigate.

GT turn landmarks (most common): bathroom, hallway, doorway, kitchen, couch, stairs, door
Phase C (v150) generates: wall, door frame, table, sofa, mirror, cabinet (too specific)

v152 uses THREE prompt types based on Gate3 room transition data:
  1. DOORWAY prompt (65.5% of turns): "Turn [dir] through a doorway into [room]. Describe doorway."
  2. ROOM-ENTRY prompt (remaining room-change turns): "Turn [dir] from [from] into [to]. Describe entry."
  3. DIRECTION-AWARE prompt (no room change, same as v150): "Turn [dir]. Describe landmark."

Process:
  1. Load v150 Phase 1 checkpoint (direction-aware, 3192 turn descriptions)
  2. Load Gate3 perframe room transition data for each episode
  3. For turns with room transitions: re-generate with room-context prompts
  4. For turns without room transitions: keep v150 descriptions (already good)
  5. Merge into new v152 Phase 1 checkpoint

Output: outputs/gate4_v152_phase1_checkpoint.json
"""

import asyncio
import base64
import gzip
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ── Config ────────────────────────────────────────────────────────────────────

VLLM_BASE_URL = "http://10.77.32.231:8000/v1"
VLLM_MODEL    = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
VLLM_API_KEY  = "token-abc123"

PIPELINE_ROOT   = Path(__file__).parent
RF_DIR          = PIPELINE_ROOT / "outputs" / "rendered_frames"
GT_PATH         = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz")
G3PF_DIR        = PIPELINE_ROOT / "outputs" / "gate3_perframe"
V150_P1_CKPT    = PIPELINE_ROOT / "outputs" / "gate4_v150_phase1_checkpoint.json"
V152_P1_CKPT    = PIPELINE_ROOT / "outputs" / "gate4_v152_phase1_checkpoint.json"


# ── Direction/room-context vision prompts ─────────────────────────────────────

def make_doorway_prompt(direction: str, room_transition: str, to_room: str) -> str:
    """Prompt for turns where the robot passes through a doorway/arch."""
    dir_upper = direction.upper()
    room_str = f" into the {to_room}" if to_room and to_room not in ('unknown', '') else ""
    # Extract doorway type from room_transition string
    doorway_type = "doorway"
    if "arched" in room_transition.lower() or "archway" in room_transition.lower():
        doorway_type = "arched doorway"
    elif "glass" in room_transition.lower():
        doorway_type = "glass doorway"
    elif "wide" in room_transition.lower():
        doorway_type = "wide doorway"

    return (
        f"A robot is navigating indoors and is turning {dir_upper} through a doorway at this location. "
        f"In 1-2 sentences, describe this transition: "
        f"(1) What TYPE of opening or threshold is visible to the {direction} "
        f"(plain doorway, arched doorway, glass door, wide opening, narrow corridor entrance)? "
        f"(2) What room or space is visible just beyond{room_str}? "
        f"Format: 'Turn {direction} through the [doorway type]{room_str}. [Brief description of what's beyond (3-6 words)].'"
        f" Reply with ONLY these 1-2 sentences."
    )


def make_room_entry_prompt(direction: str, from_room: str, to_room: str) -> str:
    """Prompt for turns where room changes but no visible doorway."""
    dir_upper = direction.upper()
    from_str = f" from the {from_room}" if from_room else ""
    to_str = f" into the {to_room}" if to_room else ""

    return (
        f"A robot is turning {dir_upper}{from_str}{to_str} at this location. "
        f"In 1-2 sentences, describe this turn: "
        f"(1) What marks this corner or turn point — a wall edge, structural feature, "
        f"or threshold visible to the {direction}? "
        f"(2) What's visible in the {to_room or 'new area'} just ahead after turning? "
        f"Format: 'Turn {direction} at/past the [landmark]{to_str}. [Brief description of what's ahead (3-6 words)].'"
        f" Reply with ONLY these 1-2 sentences."
    )


def make_direction_aware_prompt(direction: str) -> str:
    """Standard direction-aware landmark prompt (same as v150) for no-transition turns."""
    if direction == "left":
        return (
            "A robot is navigating indoors and is about to turn LEFT at this location. "
            "In 1-2 sentences: (1) name the most distinctive landmark to the LEFT "
            "or slightly ahead-left that a person would use to remember 'turn here' — "
            "use a SPECIFIC object type + color (e.g. 'green stove', 'grey stone pillar', "
            "'brown leather sofa') — avoid using 'wooden' as the ONLY descriptor. "
            "(2) Briefly describe what's visible straight ahead after turning left. "
            "Examples: 'Turn left at the green vintage stove. Ahead, a stone-floored dining area opens up.' "
            "Reply with ONLY these 1-2 sentences, nothing else."
        )
    elif direction == "right":
        return (
            "A robot is navigating indoors and is about to turn RIGHT at this location. "
            "In 1-2 sentences: (1) name the most distinctive landmark to the RIGHT "
            "or slightly ahead-right that a person would use to remember 'turn here' — "
            "use a SPECIFIC object type + color (e.g. 'white marble island', 'blue sectional sofa') "
            "— avoid using 'wooden' as the ONLY descriptor. "
            "(2) Briefly describe what's visible straight ahead after turning right. "
            "Examples: 'Turn right past the white marble kitchen island. A sunlit hallway leads ahead.' "
            "Reply with ONLY these 1-2 sentences, nothing else."
        )
    else:
        return (
            "A robot is navigating indoors and is about to turn at this location. "
            "In 1-2 sentences: (1) describe the most prominent landmark at this turning point. "
            "(2) briefly note what's visible in the direction the robot will go after turning. "
            "Reply with ONLY these 1-2 sentences, nothing else."
        )


# ── Path analysis helpers ─────────────────────────────────────────────────────

def heading_xz(p1: List, p2: List) -> float:
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2]
    return math.degrees(math.atan2(dx, -dz))


def signed_diff(a: float, b: float) -> float:
    return (b - a + 180) % 360 - 180


def get_turn_direction_at_waypoint(path: List, waypoint_idx: int) -> Optional[str]:
    n = len(path)
    i = waypoint_idx
    if i <= 0 or i >= n - 1:
        return None
    heading_before = heading_xz(path[i - 1], path[i])
    heading_after = heading_xz(path[i], path[i + 1])
    delta = signed_diff(heading_before, heading_after)
    if delta < -15:
        return "left"
    elif delta > 15:
        return "right"
    return None


# ── Gate3 perframe data loading ───────────────────────────────────────────────

def load_g3pf_turn_context(episode_id: int) -> Dict[str, Dict]:
    """Load Gate3 perframe room transition context for an episode.
    Returns {turn_label: {from_room, to_room, room_transition, has_doorway}}.
    """
    ep_id_str = f"{episode_id:06d}"
    g3pf_path = G3PF_DIR / f"episode_{ep_id_str}.json"
    if not g3pf_path.exists():
        return {}
    try:
        g3pf = json.load(open(g3pf_path))
    except Exception:
        return {}

    result = {}
    turns = g3pf.get("turns", [])
    start_room = g3pf.get("start", {}).get("room", "")
    prev_room = start_room

    for t in turns:
        label = t.get("label", "")
        if not label:
            continue
        room_transition = t.get("room_transition", "none visible")
        cur_room = t.get("room", "") or ""

        has_doorway = ("doorway" in room_transition.lower() or
                       "archway" in room_transition.lower() or
                       "arch" in room_transition.lower())
        room_changed = (cur_room and cur_room.lower() != prev_room.lower())

        result[label] = {
            "from_room": prev_room,
            "to_room": cur_room,
            "room_transition": room_transition,
            "has_doorway": has_doorway,
            "room_changed": room_changed,
        }
        if cur_room:
            prev_room = cur_room

    return result


# ── Image utilities ───────────────────────────────────────────────────────────

def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_vision_desc(raw: str) -> Optional[str]:
    raw = raw.strip().strip('"\'').strip()
    for prefix in ["Description:", "The landmark is", "I see", "I can see",
                   "In this image", "This image shows", "The image shows"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].strip(" :-").strip()
    sents = re.split(r'(?<=[.!?])\s+', raw)
    raw = " ".join(sents[:2]).strip()
    raw = re.sub(r"\s+", " ", raw).strip()
    words = raw.split()
    if len(words) < 3 or len(words) > 60:
        return None
    return raw


# ── Async VLM call ────────────────────────────────────────────────────────────

async def vision_describe_one(client, eid: str, image_path: Path, label: str,
                               direction: Optional[str], prompt_text: str,
                               prompt_type: str,
                               sem: asyncio.Semaphore,
                               done: List, total: int, t0: float) -> Dict:
    async with sem:
        try:
            b64 = image_to_base64(image_path)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt_text},
                ]
            }]
            resp = await client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                max_tokens=120,
                temperature=0.15,
            )
            raw = resp.choices[0].message.content.strip()
            desc = clean_vision_desc(raw)
            result = {
                "eid": eid, "label": label, "desc": desc, "ok": True,
                "direction": direction, "prompt_type": prompt_type,
            }
        except Exception as e:
            result = {
                "eid": eid, "label": label, "desc": None, "ok": False,
                "error": str(e), "direction": direction, "prompt_type": prompt_type,
            }

    done[0] += 1
    if done[0] % 200 == 0 or done[0] == total:
        elapsed = time.time() - t0
        r = done[0] / elapsed if elapsed > 0 else 0.001
        eta = (total - done[0]) / r if r > 0 else 0
        print(f"  [Phase1-v152 {done[0]}/{total}] {r:.1f}/s  ETA={eta/60:.1f}m", flush=True)
    return result


async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--n-episodes", type=int, default=None)
    ap.add_argument("--force-regen", action="store_true")
    ap.add_argument("--regen-doorway", action="store_true",
                    help="Re-generate only doorway/room-change turns (skip v150 non-transition turns)")
    args = ap.parse_args()

    print("=== Phase 1 v152: Room-Context Turn Descriptions ===")
    print(f"  Strategy: doorway turns → doorway prompt; room-entry turns → room-entry prompt;")
    print(f"            no-transition turns → reuse v150 direction-aware descriptions")
    print(f"  Concurrency: {args.concurrency}")

    # Load GT episodes
    with gzip.open(GT_PATH, "rt") as f:
        gt_data = json.load(f)
    episodes = gt_data["episodes"]
    if args.n_episodes:
        episodes = episodes[:args.n_episodes]
    print(f"  Episodes: {len(episodes)}")

    # Load v150 Phase 1 checkpoint (direction-aware, 3192 turn descriptions)
    v150_p1 = {}
    if V150_P1_CKPT.exists():
        v150_p1 = json.load(open(V150_P1_CKPT))
        print(f"  v150 checkpoint: {len(v150_p1)} episodes")
    else:
        print("  WARNING: v150 checkpoint not found!")

    # Load existing v152 checkpoint (for resuming)
    v152_p1 = {}
    if V152_P1_CKPT.exists() and not args.force_regen:
        v152_p1 = json.load(open(V152_P1_CKPT))
        print(f"  v152 checkpoint (existing): {len(v152_p1)} episodes")

    # Initialize v152 from v150 (inherit start, goal, AND turn descriptions)
    # We'll selectively override doorway/room-change turns
    merged = {}
    for eid_str, v150_descs in v150_p1.items():
        merged[eid_str] = dict(v150_descs)
    # Apply any already-computed v152 overrides
    for eid_str, v152_descs in v152_p1.items():
        if eid_str not in merged:
            merged[eid_str] = {}
        for k, v in v152_descs.items():
            if k.startswith("turn_") and v:
                merged[eid_str][k] = v

    print(f"  Merged checkpoint initialized: {len(merged)} episodes")

    # Build tasks: re-generate turn descriptions for doorway/room-change turns
    tasks = []
    prompt_type_stats = {"doorway": 0, "room_entry": 0, "direction_aware": 0, "skipped_v150": 0}
    FORCE_REGEN_TRANSITION = args.regen_doorway or args.force_regen

    for ep in episodes:
        eid = ep["episode_id"]
        eid_str = str(eid)

        ep_dir = RF_DIR / f"episode_{eid:06d}"
        poses_f = ep_dir / "poses.json"
        if not poses_f.exists():
            continue

        try:
            poses = json.load(open(poses_f))
        except Exception:
            continue

        ref_path = ep.get("reference_path", [])
        # Load Gate3 room transition context
        g3pf_context = load_g3pf_turn_context(eid)

        for frame in poses.get("frames", []):
            label = frame["label"]
            if not label.startswith("turn_"):
                continue

            img_path = ep_dir / frame["path"]
            if not img_path.exists():
                continue

            # Compute direction from path geometry
            waypoint_idx = frame.get("waypoint_idx", 0)
            direction = get_turn_direction_at_waypoint(ref_path, waypoint_idx)
            if direction is None:
                direction = "left"  # fallback

            # Get room context for this turn
            ctx = g3pf_context.get(label, {})
            has_doorway = ctx.get("has_doorway", False)
            room_changed = ctx.get("room_changed", False)
            from_room = ctx.get("from_room", "")
            to_room = ctx.get("to_room", "")
            room_transition = ctx.get("room_transition", "none visible")

            # Determine prompt type and whether to re-generate
            if has_doorway:
                prompt_type = "doorway"
                prompt_text = make_doorway_prompt(direction, room_transition, to_room)
            elif room_changed and to_room:
                prompt_type = "room_entry"
                prompt_text = make_room_entry_prompt(direction, from_room, to_room)
            else:
                prompt_type = "direction_aware"
                prompt_text = make_direction_aware_prompt(direction)

            # Skip re-generation if:
            # - Not forcing regen
            # - Already in v152 checkpoint (previously computed v152 description)
            # - For non-transition turns: keep v150 descriptions
            already_in_v152 = (eid_str in v152_p1 and label in v152_p1[eid_str]
                               and v152_p1[eid_str][label])
            if already_in_v152 and not args.force_regen:
                prompt_type_stats["skipped_v150"] += 1
                continue

            if not FORCE_REGEN_TRANSITION and prompt_type == "direction_aware":
                # Keep v150 description for no-transition turns
                prompt_type_stats["skipped_v150"] += 1
                continue

            prompt_type_stats[prompt_type] += 1
            tasks.append({
                "eid": eid_str,
                "label": label,
                "image_path": img_path,
                "direction": direction,
                "prompt_text": prompt_text,
                "prompt_type": prompt_type,
                "from_room": from_room,
                "to_room": to_room,
            })

    print(f"\n  Turn tasks to generate: {len(tasks)}")
    print(f"  Prompt type breakdown:")
    for pt, cnt in prompt_type_stats.items():
        print(f"    {pt}: {cnt}")

    if not tasks:
        print("  All turn descriptions already computed!")
    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        sem = asyncio.Semaphore(args.concurrency)
        done = [0]
        t0 = time.time()
        total = len(tasks)

        coros = [
            vision_describe_one(
                client, t["eid"], t["image_path"], t["label"],
                t["direction"], t["prompt_text"], t["prompt_type"],
                sem, done, total, t0
            )
            for t in tasks
        ]

        new_results = {}
        for result in await asyncio.gather(*coros):
            eid_str = result["eid"]
            label = result["label"]
            if eid_str not in new_results:
                new_results[eid_str] = {}
            new_results[eid_str][label] = result["desc"]

        elapsed = time.time() - t0
        ok = sum(1 for fd in new_results.values() for d in fd.values() if d)
        fail = sum(1 for fd in new_results.values() for d in fd.values() if not d)
        print(f"\n  Phase 1 v152 done in {elapsed/60:.1f}m: {ok} good, {fail} failed")

        # Merge new descriptions into checkpoint
        for eid_str, frame_descs in new_results.items():
            if eid_str not in merged:
                merged[eid_str] = {}
            for label, desc in frame_descs.items():
                if desc:  # Only override if VLM returned a valid description
                    merged[eid_str][label] = desc

    # Save v152 Phase 1 checkpoint
    V152_P1_CKPT.parent.mkdir(parents=True, exist_ok=True)
    with open(V152_P1_CKPT, "w") as f:
        json.dump(merged, f)
    print(f"\n  Saved: {V152_P1_CKPT}")

    # Stats
    total_eps = len(merged)
    turn_descs = sum(
        1 for fd in merged.values()
        for k, v in fd.items()
        if k.startswith("turn_") and v
    )

    # Analyze turn landmark types in v152
    from collections import Counter
    landmark_types = Counter()
    doorway_count = 0
    room_count = 0
    for fd in merged.values():
        for k, desc in fd.items():
            if not k.startswith("turn_") or not desc:
                continue
            m = re.match(r'Turn\s+(?:left|right)?\s*(?:at|past|through|into)\s+(?:the\s+)?(.+?)[.\n]',
                         desc, re.IGNORECASE)
            if m:
                lm = m.group(1).strip().lower()
                last_word = lm.split()[-1] if lm.split() else ""
                landmark_types[last_word] += 1
                if "doorway" in lm or "arch" in lm or "door" in lm:
                    doorway_count += 1
                if any(room in lm for room in ["hallway", "kitchen", "bedroom", "bathroom",
                                                "living room", "dining", "closet", "office"]):
                    room_count += 1

    print(f"\n=== Phase 1 v152 Stats ===")
    print(f"  Episodes: {total_eps}")
    print(f"  Turn descriptions: {turn_descs}")
    print(f"  Doorway/arch landmarks: {doorway_count} ({100*doorway_count/turn_descs:.1f}%)")
    print(f"  Room-name landmarks: {room_count} ({100*room_count/turn_descs:.1f}%)")
    print(f"\n  Top landmark types (last word):")
    for lm_type, cnt in landmark_types.most_common(12):
        print(f"    {lm_type}: {cnt} ({100*cnt/turn_descs:.1f}%)")

    print(f"\n  Sample v152 descriptions:")
    cnt = 0
    for eid_str, fd in list(merged.items())[:50]:
        for k, desc in fd.items():
            if k.startswith("turn_") and desc:
                if any(kw in desc.lower() for kw in ["doorway", "room", "hallway", "kitchen", "bathroom"]):
                    print(f"    EP{eid_str} [{k}]: {desc[:100]}")
                    cnt += 1
                    if cnt >= 5:
                        break
        if cnt >= 5:
            break


if __name__ == "__main__":
    asyncio.run(main())
