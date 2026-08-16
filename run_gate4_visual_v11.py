#!/usr/bin/env python3
"""
Gate 4 Visual v11 — GT-Style Visual-Anchor Instructions

ROOT CAUSE FIX: InternVLA-N1 was trained on GT instructions that anchor turns
to visual landmarks: "turn left at the refrigerator", "walk past the television".
Our v2-v10 instructions say "turn left" with NO visual anchor → model can't
verify it's at the correct turn → goes off-path → 25pp SR gap vs GT.

v11 KEY INNOVATIONS:
1. Per-turn visual anchors from Gate 3 v2 (per-frame detection)
   - Each turn gets its own detected landmark (not pooled scene context)
   - "turn left at the [TURN_LANDMARK]" / "when you reach the [X], turn right"
2. Spatial preposition injection to match GT distribution
   - GT: through=30.3%, past=22.2%, into=31.7%, towards=12.4%
   - v10: through=2.1%, past=1.2%, into=9.7%, towards=0.0% (massive gap)
3. Structured route template matching GT sentence structure
   - Start: "Walk [through/past/into] the [start_context]..."
   - Turn:  "...turn [dir] at/past the [TURN_LANDMARK]..."
   - Goal:  "Stop [near/in front of/at] the [GOAL_LANDMARK]."
4. Path-matched GT examples (same scene, similar path) as few-shot
   - Shows 8 GT examples using the exact spatial vocabulary for this building

Expected improvement (vs Gen v2 SR=38.44%):
  SR target: > 65% (beat GT 63.89%)
  Mechanism: model recognizes matching visual cues → correct turn confirmation

Output: outputs/datasets/val_unseen_generated_gemma_visual_v11.json.gz

Usage:
  source /home/kemal/VLNav/vlnav_env/bin/activate
  python3 run_gate4_visual_v11.py [--n-episodes N] [--concurrency C]
"""
import asyncio
import gzip
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

GT_PATH     = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
G3PF_DIR    = ROOT / "outputs" / "gate3_perframe"   # Gate 3 v2 per-frame outputs
G3_OLD_DIR  = ROOT / "outputs" / "gate3_landmarks"  # Gate 3 v1 fallback
OUTPUT      = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v11.json.gz"
CKPT        = ROOT / "outputs" / "gate4_visual_v11_checkpoint.json"

from gate2_path.path_analyzer import analyze_path
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset


# ── Path utilities ─────────────────────────────────────────────────────────────

def get_primitives(ep: Dict) -> List[Dict]:
    pa = analyze_path(ep["reference_path"], ep.get("start_rotation"))
    return pa.get("primitives", [])


def extract_path_features(ep: Dict) -> Dict:
    pa    = analyze_path(ep["reference_path"], ep.get("start_rotation"))
    prims = pa.get("primitives", [])
    turns = []
    total_dist = 0.0
    for p in prims:
        if p["type"] == "left_turn":    turns.append("L")
        elif p["type"] == "right_turn": turns.append("R")
        elif p["type"] == "straight":   total_dist += p.get("distance_m", 0)
    summary = pa.get("summary", {})
    return {
        "turns":      turns,
        "n_turns":    len(turns),
        "total_dist": summary.get("total_distance_m", total_dist),
        "n_waypoints": summary.get("n_waypoints", len(ep["reference_path"])),
    }


def turn_edit_dist(t1: List[str], t2: List[str]) -> float:
    m, n = len(t1), len(t2)
    if m == 0 and n == 0: return 0.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = (prev[j-1] if t1[i-1] == t2[j-1]
                     else 1 + min(prev[j], dp[j-1], prev[j-1]))
    return dp[n]


def path_similarity(feat1: Dict, feat2: Dict) -> float:
    t1, t2 = feat1["turns"], feat2["turns"]
    max_t    = max(len(t1), len(t2), 1)
    turn_sim = 1.0 - turn_edit_dist(t1, t2) / max_t
    d1, d2   = feat1["total_dist"], feat2["total_dist"]
    dist_sim = 1.0 - abs(d1 - d2) / max(d1 + d2, 0.01)
    n1, n2   = feat1["n_waypoints"], feat2["n_waypoints"]
    wpt_sim  = 1.0 - abs(n1 - n2) / max(n1, n2, 1)
    return 0.6 * turn_sim + 0.25 * dist_sim + 0.15 * wpt_sim


class ScenePathIndex:
    """Index of GT episodes by scene for path-matched few-shot retrieval."""
    def __init__(self, all_eps: List[Dict]):
        self.by_scene: Dict[str, List[Dict]] = defaultdict(list)
        print("Building scene-path index...", flush=True)
        for ep in all_eps:
            sc    = ep["scene_id"].split("/")[-2]
            instr = (ep["instruction"]["instruction_text"]
                     if isinstance(ep.get("instruction"), dict) else "")
            if not instr.strip():
                continue
            feat = extract_path_features(ep)
            self.by_scene[sc].append({
                "episode_id":  ep["episode_id"],
                "instruction": instr.strip(),
                "feat":        feat,
            })
        total = sum(len(v) for v in self.by_scene.values())
        print(f"Index: {len(self.by_scene)} scenes, {total} GT episodes with instructions")

    def top_k_similar(self, ep: Dict, ep_feat: Dict, k: int = 8) -> List[str]:
        sc   = ep["scene_id"].split("/")[-2]
        pool = [e for e in self.by_scene.get(sc, [])
                if e["episode_id"] != ep["episode_id"]]
        if not pool:
            return []
        scored = sorted(
            [(path_similarity(ep_feat, e["feat"]), e) for e in pool],
            key=lambda x: -x[0]
        )
        # Take top k with sim >= 0.2, then backfill if needed
        top = [e["instruction"] for sim, e in scored[:k] if sim >= 0.2]
        if len(top) < 4:
            top = [e["instruction"] for _, e in scored[:k]]
        return top[:k]


# ── Gate 3 v2 landmark loading ─────────────────────────────────────────────────

def load_perframe_landmarks(g3pf_dir: Path) -> Dict[int, Dict]:
    """Load Gate 3 v2 per-frame landmark results."""
    lm_map = {}
    if not g3pf_dir.exists():
        return lm_map
    for fp in g3pf_dir.glob("episode_*.json"):
        try:
            d   = json.load(open(fp))
            eid = d.get("episode_id") or int(fp.stem.replace("episode_", "").lstrip("0") or "0")
            lm_map[eid] = d
        except Exception:
            pass
    return lm_map


def load_old_landmarks(g3_dir: Path) -> Dict[int, Dict]:
    """Load Gate 3 v1 landmark results (fallback)."""
    lm_map = {}
    for fp in g3_dir.glob("episode_*.json"):
        try:
            d   = json.load(open(fp))
            eid = d.get("episode_id") or int(fp.stem.replace("episode_", "").lstrip("0") or "0")
            lm_map[eid] = d
        except Exception:
            pass
    return lm_map


def get_landmark_info(
    ep: Dict,
    pf_map: Dict[int, Dict],
    old_map: Dict[int, Dict],
    primitives: List[Dict],
) -> Dict:
    """
    Assemble per-turn landmark info for an episode.
    Returns {start_context, turns: [{direction, landmark, room}...], goal_landmark, has_perframe}.
    """
    eid = ep["episode_id"]
    pf  = pf_map.get(eid)
    old = old_map.get(eid, {})

    # Build turn direction sequence from primitives
    turn_directions = []
    for p in primitives:
        if p["type"] == "left_turn":    turn_directions.append("left")
        elif p["type"] == "right_turn": turn_directions.append("right")

    result = {
        "has_perframe": pf is not None,
        "start_context": "",
        "start_room": "",
        "turns": [],
        "goal_landmark": "destination",
        "goal_room": "",
    }

    if pf:
        # Use Gate 3 v2 per-frame data
        start = pf.get("start") or {}
        result["start_room"]    = start.get("room", "")
        start_lms = start.get("landmarks", [])
        start_main = start.get("main_landmark", "")
        if start_main and start_main not in start_lms:
            start_lms = [start_main] + start_lms
        result["start_context"] = ", ".join(start_lms[:3]) if start_lms else "the starting area"

        # Per-turn landmarks — pair with direction from primitives
        pf_turns = pf.get("turns", [])
        for i, turn_det in enumerate(pf_turns):
            direction = turn_directions[i] if i < len(turn_directions) else "left"
            main = turn_det.get("main_landmark") or ""
            others = turn_det.get("landmarks", [])
            room = turn_det.get("room", "")
            room_trans = turn_det.get("room_transition", "")
            # Pick best landmark: main > others[0] > "the area"
            best = main or (others[0] if others else "")
            if not best:
                best = f"the {room}" if room else "the area"
            result["turns"].append({
                "direction":  direction,
                "landmark":   best,
                "room":       room,
                "room_trans": room_trans,
                "all_lms":    ([main] + others)[:3] if main else others[:3],
            })

        # Fill missing turns (primitives may have more turns than rendered frames)
        for i in range(len(pf_turns), len(turn_directions)):
            direction = turn_directions[i]
            result["turns"].append({
                "direction": direction,
                "landmark":  "",
                "room":      "",
                "room_trans": "",
                "all_lms":   [],
            })

        goal = pf.get("goal") or {}
        result["goal_landmark"] = goal.get("stop_landmark", "") or goal.get("main_landmark", "destination")
        result["goal_room"]     = goal.get("room", "")

    else:
        # Fallback to Gate 3 v1 data
        sc = old.get("scene_context") or {}
        gl = old.get("goal_landmark") or {}
        start_lms = sc.get("landmarks", [])[:3]
        result["start_context"] = ", ".join(start_lms) if start_lms else "the starting area"
        result["start_room"]    = sc.get("room_type", "")
        # No per-turn data — use primitives only
        for direction in turn_directions:
            result["turns"].append({
                "direction": direction, "landmark": "", "room": "",
                "room_trans": "", "all_lms": [],
            })
        result["goal_landmark"] = (
            gl.get("stop_landmark") or sc.get("stop_landmark") or "the destination"
        )
        result["goal_room"] = gl.get("room_type", "")

    # Fallback: if goal_landmark is still empty, use a generic
    if not result["goal_landmark"] or result["goal_landmark"] in ("", "destination"):
        result["goal_landmark"] = "the destination"

    return result


# ── Route description builder ──────────────────────────────────────────────────

def build_route_description(primitives: List[Dict], lm_info: Dict) -> str:
    """
    Build a structured route description injecting per-turn landmarks.
    Example: "straight 2m → turn left at [wooden dining table] → straight 4m → turn left at [gray couch] → STOP"
    """
    parts = []
    turn_idx = 0
    turns_info = lm_info.get("turns", [])

    for p in primitives:
        t = p["type"]
        if t == "straight":
            d = p.get("distance_m", 0)
            if d > 0.5:
                parts.append(f"straight {d:.0f}m")
        elif t in ("left_turn", "right_turn"):
            direction = "left" if t == "left_turn" else "right"
            turn_data = turns_info[turn_idx] if turn_idx < len(turns_info) else {}
            landmark  = turn_data.get("landmark", "")
            room      = turn_data.get("room", "")
            room_trans = turn_data.get("room_trans", "")
            if landmark and room_trans and "doorway" in room_trans.lower():
                # e.g. "turn left past the potted plant through the doorway"
                parts.append(f"turn {direction} past [{landmark}] through doorway into [{room}]")
            elif landmark:
                # e.g. "turn left at [white kitchen island]"
                parts.append(f"turn {direction} at [{landmark}]")
            elif room:
                # e.g. "turn left into [living room]"
                parts.append(f"turn {direction} into [{room}]")
            else:
                parts.append(f"turn {direction}")
            turn_idx += 1
        elif t == "elevation":
            dir_str = p.get("direction", "up")
            parts.append(f"{'up' if dir_str == 'up' else 'down'} stairs")
        elif t == "stop":
            goal_lm = lm_info.get("goal_landmark", "destination")
            parts.append(f"STOP at [{goal_lm}]")

    return " → ".join(parts) if parts else "walk forward → STOP"


# ── Starting verb from GT ──────────────────────────────────────────────────────

def gt_start_verb(gt_instr: str) -> str:
    tok = gt_instr.strip().split()
    if not tok: return "Walk"
    w = tok[0].rstrip(".,!?;:")
    return w[0].upper() + w[1:].lower() if w else "Walk"


# ── Prompt builder ─────────────────────────────────────────────────────────────

SYSTEM_INTRO = """You write navigation instructions for an indoor robot following R2R (Room-to-Room) VLN style.

R2R instructions have these key characteristics:
- Start with an action verb (Walk, Go, Exit, Enter, Head, Turn)
- Anchor turns to VISIBLE OBJECTS: "turn left at the refrigerator", "turn right past the wooden dresser"
- Use spatial prepositions: "past", "through", "into", "towards", "near", "in front of"
- Name specific objects (not generic): "gray sectional sofa" not "sofa", "wooden dining table" not "table"
- End with stop condition: "Stop near the [object]" or "Stop in front of the [object]"
- Length: 22-32 words, 1-3 sentences"""


def build_prompt(
    ep: Dict,
    gt_instr: str,
    matched_examples: List[str],
    lm_info: Dict,
    primitives: List[Dict],
) -> Tuple[str, str]:
    sv = gt_start_verb(gt_instr)

    # Route with per-turn landmark cues
    route_desc = build_route_description(primitives, lm_info)

    # Turn summary for clarity
    turns_info = lm_info.get("turns", [])
    turn_parts = []
    for t in turns_info:
        d = t.get("direction", "?")
        lm = t.get("landmark", "")
        if lm:
            turn_parts.append(f"turn {d} at the {lm}")
        else:
            turn_parts.append(f"turn {d}")
    turn_summary = " → ".join(turn_parts) if turn_parts else "no turns"

    # Few-shot examples block
    ex_block = "\n".join(f"  {i+1}. \"{ex}\"" for i, ex in enumerate(matched_examples))
    n_ex = len(matched_examples)

    # Goal
    goal_lm = lm_info.get("goal_landmark", "the destination")
    goal_room = lm_info.get("goal_room", "")
    stop_phrase = f"near the {goal_lm}"
    if goal_room and goal_lm != "the destination":
        stop_phrase = f"near the {goal_lm} in the {goal_room}"

    # Start context
    start_ctx = lm_info.get("start_context", "")
    start_room = lm_info.get("start_room", "")

    # Build per-turn cue lines for explicit injection
    turn_cue_lines = []
    for i, t in enumerate(turns_info, 1):
        d = t.get("direction", "?")
        lm = t.get("landmark", "")
        rm = t.get("room", "")
        rt = t.get("room_trans", "")
        if lm and rt and "doorway" in rt.lower():
            # Suggest through-doorway pattern
            turn_cue_lines.append(f"  Turn {i}: turn {d} past the [{lm}] through the doorway into [{rm}]")
        elif lm:
            turn_cue_lines.append(f"  Turn {i}: turn {d} AT/NEAR/PAST the [{lm}]")
        elif rm:
            turn_cue_lines.append(f"  Turn {i}: turn {d} INTO the [{rm}]")
        else:
            turn_cue_lines.append(f"  Turn {i}: turn {d}")
    turn_cue_block = "\n".join(turn_cue_lines) if turn_cue_lines else "  (no turns)"

    prompt = (
        f"{SYSTEM_INTRO}\n\n"
        f"SAME-BUILDING examples ({n_ex} GT instructions — study their spatial vocabulary "
        f"and how they anchor turns to objects):\n"
        f"{ex_block}\n\n"
        f"Write ONE instruction for THIS SPECIFIC PATH:\n\n"
        f"  Route: {route_desc}\n"
        f"  Turns with visual cues:\n{turn_cue_block}\n"
        f"  Start: {start_room} area, visible: {start_ctx or 'furniture'}\n"
        f"  Stop: {stop_phrase}\n\n"
        f"REQUIREMENTS:\n"
        f"- MUST anchor each turn to a visible object (use ANY of these patterns):\n"
        f"  * \"turn {turns_info[0]['direction'] if turns_info else 'left'} at the [object]\"\n"
        f"  * \"walk past the [object] and turn {turns_info[0]['direction'] if turns_info else 'left'}\"\n"
        f"  * \"turn {turns_info[0]['direction'] if turns_info else 'left'} through/into the [room]\"\n"
        f"- Use spatial prepositions: past, through, into, towards, near, at\n"
        f"- Target 22-30 words (shorter is fine for simple paths)\n"
        f"- End with: \"Stop near/at/in front of the [landmark].\"\n"
        f"- Write ONLY the instruction (no explanation, no quotation marks)\n"
        f"- Start with: \"{sv}\"\n\n"
        f"{sv}"
    )
    return prompt, sv


# ── Async generation ───────────────────────────────────────────────────────────

async def generate(tasks: List[Dict], concurrency: int = 12) -> Dict[int, str]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    sem    = asyncio.Semaphore(concurrency)
    done   = [0]
    t0     = time.time()
    total  = len(tasks)
    results: Dict[int, str] = {}

    async def one(task: Dict):
        eid = task["episode_id"]
        async with sem:
            for attempt in range(3):
                try:
                    resp = await client.chat.completions.create(
                        model=VLLM_MODEL,
                        messages=[{"role": "user", "content": task["prompt"]}],
                        max_tokens=150,
                        temperature=0.25,
                    )
                    results[eid] = resp.choices[0].message.content.strip()
                    break
                except Exception as e:
                    if attempt == 2:
                        results[eid] = f"ERROR: {e}"
                    await asyncio.sleep(0.5 * (attempt + 1))
        done[0] += 1
        if done[0] % 200 == 0 or done[0] == total:
            elapsed = time.time() - t0
            r = done[0] / elapsed if elapsed > 0 else 0.001
            eta = (total - done[0]) / r if r > 0 else 0
            print(f"  [{done[0]}/{total}] {r:.1f}/s  ETA={eta/60:.1f}m", flush=True)

    await asyncio.gather(*(one(t) for t in tasks))
    return results


# ── Post-processing ────────────────────────────────────────────────────────────

STOP_WORDS = {"stop", "wait", "halt", "stand", "pause"}
PREAMBLES  = ["Instruction:", "Navigation:", "Answer:", "Sure,", "Certainly,",
              "Of course,", "Here is", "Here's", "Result:", "Walk:", "The instruction:"]


def clean(raw: str, sv: str) -> str:
    raw = raw.strip()
    # Strip quotes
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1].strip()
    # Strip preamble
    for pre in PREAMBLES:
        if raw.lower().startswith(pre.lower()):
            raw = raw[len(pre):].lstrip(" :\n").strip()
    # Fix double start verb
    sv_l = sv.lower()
    if raw.lower().startswith(sv_l + " " + sv_l):
        raw = raw[len(sv_l):].lstrip()
    # Ensure starts with sv
    if not raw.lower().startswith(sv_l):
        raw = sv + " " + (raw[0].lower() + raw[1:] if raw else "to the destination.")
    # Max 3 sentences
    sents  = re.split(r'(?<=[.!?])\s+', raw)
    result = " ".join(sents[:3]).strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result


def quality_ok(text: str) -> Tuple[bool, str]:
    words = text.split()
    if len(words) < 8:  return False, "too_short"
    if len(words) > 70: return False, "too_long"
    if not any(w in text.lower().split() for w in STOP_WORDS):
        return False, "no_stop"
    return True, "ok"


def fix_no_stop(text: str, goal_lm: str) -> str:
    return text.rstrip(". !") + f". Stop near the {goal_lm}."


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes",  type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--k-similar",   type=int, default=8,
                    help="GT examples per episode (path-matched, same scene)")
    args = ap.parse_args()

    print("=== Gate 4 v11: GT-Style Visual-Anchor Instructions ===")
    print(f"  Model:       {VLLM_MODEL}")
    print(f"  Concurrency: {args.concurrency}  k_similar: {args.k_similar}")
    print(f"  Gate3 v2 dir: {G3PF_DIR}")
    print()

    # Load GT dataset
    with gzip.open(GT_PATH, "rt") as f:
        gt_data = json.load(f)
    all_eps  = gt_data["episodes"]
    episodes = all_eps[:args.n_episodes] if args.n_episodes else all_eps
    print(f"Episodes: {len(episodes)}")

    # Load GT instruction map
    gt_map = {ep["episode_id"]: (ep["instruction"]["instruction_text"]
              if isinstance(ep.get("instruction"), dict) else "")
              for ep in all_eps}

    # Build path index for few-shot retrieval
    sc_path_idx = ScenePathIndex(all_eps)

    # Load per-frame landmarks (Gate 3 v2)
    print("Loading Gate 3 v2 per-frame landmarks...")
    pf_map  = load_perframe_landmarks(G3PF_DIR)
    old_map = load_old_landmarks(G3_OLD_DIR)
    n_pf    = sum(1 for ep in episodes if ep["episode_id"] in pf_map)
    n_old   = sum(1 for ep in episodes if ep["episode_id"] not in pf_map and ep["episode_id"] in old_map)
    print(f"  Gate 3 v2 (per-frame): {n_pf}/{len(episodes)} episodes")
    print(f"  Gate 3 v1 (fallback):  {n_old}/{len(episodes)} episodes")
    print()

    # Load checkpoint
    ckpt: Dict[str, str] = {}
    if CKPT.exists():
        ckpt = json.load(open(CKPT))
        print(f"Checkpoint: {len(ckpt)} episodes done")

    tokenizer = VLNTokenizer(GT_PATH)
    tasks: List[Dict] = []
    task_meta: Dict[int, Dict] = {}
    skipped = 0

    # Compute path features for all episodes
    print("Computing path features for retrieval...")
    ep_feats = {ep["episode_id"]: extract_path_features(ep) for ep in episodes}

    for ep in episodes:
        eid      = ep["episode_id"]
        gt_instr = gt_map.get(eid, "")
        ep_feat  = ep_feats[eid]
        prims    = get_primitives(ep)
        lm_info  = get_landmark_info(ep, pf_map, old_map, prims)

        if str(eid) in ckpt:
            skipped += 1
            task_meta[eid] = {
                "sv":      gt_start_verb(gt_instr),
                "goal_lm": lm_info["goal_landmark"],
            }
            continue

        matched = sc_path_idx.top_k_similar(ep, ep_feat, k=args.k_similar)
        prompt, sv = build_prompt(ep, gt_instr, matched, lm_info, prims)

        task_meta[eid] = {"sv": sv, "goal_lm": lm_info["goal_landmark"]}
        tasks.append({"episode_id": eid, "prompt": prompt, "sv": sv, "goal_lm": lm_info["goal_landmark"]})

    print(f"Tasks: {len(tasks)}  Skipped (cached): {skipped}")

    # Show sample prompt
    if tasks:
        sample = tasks[0]
        print(f"\n--- Sample prompt (ep {sample['episode_id']}) ---")
        print(sample["prompt"][:1000])
        if len(sample["prompt"]) > 1000:
            print("[...truncated...]")
        print("-" * 60)
        print()

    if tasks:
        print("Generating instructions...")
        results = await generate(tasks, args.concurrency)
        ckpt.update({str(k): v for k, v in results.items()})
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        with open(CKPT, "w") as f:
            json.dump(ckpt, f)
        print(f"Checkpoint saved: {len(ckpt)} total")

    # ── Assemble dataset ──────────────────────────────────────────────────────
    print("\nAssembling dataset...")
    ep_map = {ep["episode_id"]: ep for ep in episodes}
    episodes_out: List[Dict] = []
    n_pass = n_fix = n_fail = 0
    retry_eids: List[int] = []

    for ep in episodes:
        eid  = ep["episode_id"]
        raw  = ckpt.get(str(eid), "")
        if not raw or raw.startswith("ERROR"):
            n_fail += 1
            continue

        meta = task_meta.get(eid, {})
        text = clean(raw, meta.get("sv", "Walk"))
        ok, reason = quality_ok(text)

        if not ok:
            if reason == "no_stop":
                text = fix_no_stop(text, meta.get("goal_lm", "the destination"))
                n_fix += 1
                ok = True
            elif reason == "too_short":
                retry_eids.append(eid)
                continue

        if ok:
            episodes_out.append(assemble_episode(ep, text, tokenizer))
            n_pass += 1

    if retry_eids:
        print(f"Retrying {len(retry_eids)} short episodes with simplified prompt...")
        retry_tasks = []
        for eid in retry_eids:
            ep = ep_map.get(eid)
            if not ep: continue
            gt_instr = gt_map.get(eid, "")
            sv = gt_start_verb(gt_instr)
            prims    = get_primitives(ep)
            lm_info  = get_landmark_info(ep, pf_map, old_map, prims)
            goal_lm  = lm_info["goal_landmark"]
            turns_str = " → ".join(
                f"turn {t['direction']} at the {t['landmark']}" if t.get("landmark")
                else f"turn {t['direction']}"
                for t in lm_info["turns"]
            ) or "walk straight"
            prompt = (
                f"Write a 24-30 word indoor navigation instruction in R2R style.\n"
                f"Route: {turns_str}. Stop near the {goal_lm}.\n"
                f"Anchor each turn to a visible landmark.\n"
                f"Start with '{sv}':\n{sv}"
            )
            retry_tasks.append({"episode_id": eid, "prompt": prompt, "sv": sv, "goal_lm": goal_lm})

        retry_results = await generate(retry_tasks, args.concurrency)
        ckpt.update({str(k): v for k, v in retry_results.items()})
        with open(CKPT, "w") as f:
            json.dump(ckpt, f)

        ep_out_map = {eo["episode_id"]: i for i, eo in enumerate(episodes_out)}
        for rt in retry_tasks:
            eid = rt["episode_id"]
            raw = retry_results.get(eid, "")
            ep  = ep_map.get(eid)
            if not raw or not ep: continue
            text = clean(raw, rt["sv"])
            ok, reason = quality_ok(text)
            if reason == "no_stop":
                text = fix_no_stop(text, rt["goal_lm"]); ok = True
            if ok:
                assembled = assemble_episode(ep, text, tokenizer)
                if eid in ep_out_map:
                    episodes_out[ep_out_map[eid]] = assembled
                else:
                    episodes_out.append(assembled); n_pass += 1

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_dataset({
        "episodes": episodes_out,
        "instruction_vocab": gt_data.get("instruction_vocab", {}),
        "_generation_meta": {
            "mode":     "gt_visual_anchor_v11",
            "model":    VLLM_MODEL,
            "n_ep":     len(episodes_out),
            "gate3_v2": f"{n_pf}/{len(episodes)} with per-frame landmarks",
        },
    }, OUTPUT)

    n = len(episodes_out)
    print(f"\n=== v11 Done ===")
    print(f"  Episodes: {n}  Pass: {n_pass}  Fixed: {n_fix}  Failed: {n_fail}")

    # Quality stats
    wc: List[int] = []
    stop_n = 0
    spatial_counts = {"past": 0, "through": 0, "into": 0, "towards": 0, "at the": 0}
    anchor_n = 0  # instructions with "turn X at/past/through the Y"
    for ep_o in episodes_out:
        instr = (ep_o["instruction"]["instruction_text"]
                 if isinstance(ep_o.get("instruction"), dict) else "")
        words = instr.strip().split()
        if words: wc.append(len(words))
        il = instr.lower()
        if any(w in il.split() for w in STOP_WORDS): stop_n += 1
        for sp in spatial_counts:
            if sp in il: spatial_counts[sp] += 1
        if re.search(r"turn (left|right) (at|past|through) the", il):
            anchor_n += 1

    print(f"  avg_words: {sum(wc)/max(len(wc),1):.1f}  (GT=26.8)")
    print(f"  stop%:     {100*stop_n/max(n,1):.1f}%")
    print(f"  turn-anchor%: {100*anchor_n/max(n,1):.1f}%  (turn X at/past/through the Y)")
    print(f"  spatial prepositions:")
    for sp, cnt in spatial_counts.items():
        print(f"    {sp:12s}: {100*cnt/max(n,1):.1f}%  (GT: past=22%, through=30%, into=32%)")
    print(f"\nDataset saved: {OUTPUT}")
    print(f"Run habitat eval:")
    print(f"  bash habitat_eval/scripts/run_eval_gate4_v11.sh")


if __name__ == "__main__":
    asyncio.run(main())
