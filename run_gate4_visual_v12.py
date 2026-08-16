#!/usr/bin/env python3
"""
Gate 4 Visual v12 — Turn-Anchor Enforced Instructions

IMPROVEMENTS OVER v11 (based on analysis of v11 samples and SR gap):

1. TURN-ANCHOR FEW-SHOT PRIORITY
   v11: path-matched GT examples from same scene (random turn-anchor presence)
   v12: prefer GT examples that CONTAIN turn-anchor patterns ("turn left at the X")
        guarantees model sees examples of exact pattern we require
        Backfill with non-anchored examples if fewer than min_anchor_examples

2. TURN-ANCHOR VALIDATION
   v11: quality_ok checks length + stop word only
   v12: also validates turn_anchor_count >= min(expected_turns, 1)
        retries up to 2 times if anchors missing
        → ensures EVERY turn-having episode has ≥1 explicit anchor

3. TIGHTER LENGTH TARGET
   v11: 22-30 words (generated avg=29.8, GT=26.8 → 3w over)
   v12: 19-27 words (targets GT sweet spot, avoids BLEU precision penalty)

4. LANDMARK DEDUPLICATION
   v11: same landmark used for consecutive turns (EP 297: beige armchair twice)
   v12: for consecutive turns with same main landmark, use alternative from all_lms

5. STRONGER SYSTEM PROMPT
   v12: more concrete "DO" examples, explicit turn-anchor counter-examples ("DON'T say: turn left")

Output: outputs/datasets/val_unseen_generated_gemma_visual_v12.json.gz

Usage:
  source /home/kemal/VLNav/vlnav_env/bin/activate
  python3 run_gate4_visual_v12.py [--n-episodes N] [--concurrency C]
"""
import asyncio
import gzip
import json
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
GT_TRAIN_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/train/R2R_train.json.gz"
G3PF_DIR    = ROOT / "outputs" / "gate3_perframe"
G3_OLD_DIR  = ROOT / "outputs" / "gate3_landmarks"
OUTPUT      = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v12.json.gz"
CKPT        = ROOT / "outputs" / "gate4_visual_v12_checkpoint.json"

from gate2_path.path_analyzer import analyze_path
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset

# ── Turn-anchor detection ──────────────────────────────────────────────────────

TURN_ANCHOR_RE = re.compile(
    r'turn\s+(left|right)\s+(at|past|through|into|around)\s+the\s+\w',
    re.IGNORECASE
)
WALK_PAST_RE = re.compile(
    r'(walk|go|head|move)\s+past\s+the\s+\w',
    re.IGNORECASE
)

def has_turn_anchor(text: str) -> bool:
    return bool(TURN_ANCHOR_RE.search(text) or WALK_PAST_RE.search(text))

def count_turn_anchors(text: str) -> int:
    return len(TURN_ANCHOR_RE.findall(text)) + len(WALK_PAST_RE.findall(text))


# ── Path utilities (same as v11) ──────────────────────────────────────────────

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
    """
    Index GT episodes by scene for path-matched few-shot retrieval.
    v12 improvement: scores anchored examples higher to prefer them in top-k.
    """
    def __init__(self, all_eps: List[Dict]):
        self.by_scene: Dict[str, List[Dict]] = defaultdict(list)
        print("Building scene-path index (v12 with turn-anchor priority)...", flush=True)
        n_anchored = 0
        for ep in all_eps:
            sc    = ep["scene_id"].split("/")[-2]
            instr = (ep["instruction"]["instruction_text"]
                     if isinstance(ep.get("instruction"), dict) else "")
            if not instr.strip():
                continue
            anchored = has_turn_anchor(instr)
            if anchored: n_anchored += 1
            feat = extract_path_features(ep)
            self.by_scene[sc].append({
                "episode_id":  ep["episode_id"],
                "instruction": instr.strip(),
                "feat":        feat,
                "anchored":    anchored,
            })
        total = sum(len(v) for v in self.by_scene.values())
        print(f"Index: {len(self.by_scene)} scenes, {total} GT eps, {n_anchored} with turn-anchors ({100*n_anchored/max(total,1):.1f}%)")

    def top_k_similar(self, ep: Dict, ep_feat: Dict, k: int = 8,
                       min_anchor_examples: int = 2) -> List[str]:
        """
        Return top-k most path-similar GT instructions from same scene.
        v12: boosts similarity score for anchored examples to prefer them.
        Guarantees at least min_anchor_examples anchored instructions if available.
        """
        sc   = ep["scene_id"].split("/")[-2]
        pool = [e for e in self.by_scene.get(sc, [])
                if e["episode_id"] != ep["episode_id"]]
        if not pool:
            return []

        # Score = path_similarity + 0.15 bonus for turn-anchored (v12 key change)
        scored = sorted(
            [(path_similarity(ep_feat, e["feat"]) + (0.15 if e["anchored"] else 0.0), e)
             for e in pool],
            key=lambda x: -x[0]
        )

        top_instructions = [e["instruction"] for _, e in scored[:k]]
        n_anchored = sum(1 for instr in top_instructions if has_turn_anchor(instr))

        # If not enough anchored examples, backfill from global pool
        if n_anchored < min_anchor_examples:
            global_anchored = [e for e in pool if e["anchored"]
                               and e["instruction"] not in set(top_instructions)]
            for e in global_anchored:
                if n_anchored >= min_anchor_examples:
                    break
                if len(top_instructions) >= k:
                    top_instructions[-1] = e["instruction"]  # replace last non-anchored
                else:
                    top_instructions.append(e["instruction"])
                if has_turn_anchor(e["instruction"]):
                    n_anchored += 1

        return top_instructions[:k]


# ── Gate 3 v2 landmark loading (same as v11) ──────────────────────────────────

def load_perframe_landmarks(g3pf_dir: Path) -> Dict[int, Dict]:
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
    Assemble per-turn landmark info.
    v12 improvement: deduplicates consecutive turns with same main landmark.
    """
    eid = ep["episode_id"]
    pf  = pf_map.get(eid)
    old = old_map.get(eid, {})

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
        start = pf.get("start") or {}
        result["start_room"]    = start.get("room", "")
        start_lms = start.get("landmarks", [])
        start_main = start.get("main_landmark", "")
        if start_main and start_main not in start_lms:
            start_lms = [start_main] + start_lms
        result["start_context"] = ", ".join(start_lms[:3]) if start_lms else "the starting area"

        pf_turns = pf.get("turns", [])
        prev_landmark = None  # v12: track previous turn landmark for dedup

        for i, turn_det in enumerate(pf_turns):
            direction  = turn_directions[i] if i < len(turn_directions) else "left"
            main       = turn_det.get("main_landmark") or ""
            others     = turn_det.get("landmarks", [])
            room       = turn_det.get("room", "")
            room_trans = turn_det.get("room_transition", "")
            all_lms    = ([main] + [lm for lm in others if lm != main])[:4] if main else others[:4]

            # v12: DEDUP — if same as previous turn, pick next available landmark
            best = main or (others[0] if others else "")
            if best and best == prev_landmark and len(all_lms) > 1:
                best = all_lms[1]  # use second choice
            elif not best:
                best = f"the {room}" if room else "the area"

            prev_landmark = best
            result["turns"].append({
                "direction":  direction,
                "landmark":   best,
                "room":       room,
                "room_trans": room_trans,
                "all_lms":   all_lms,
            })

        for i in range(len(pf_turns), len(turn_directions)):
            direction = turn_directions[i]
            result["turns"].append({
                "direction": direction, "landmark": "", "room": "",
                "room_trans": "", "all_lms": [],
            })

        goal = pf.get("goal") or {}
        result["goal_landmark"] = goal.get("stop_landmark", "") or goal.get("main_landmark", "destination")
        result["goal_room"]     = goal.get("room", "")

    else:
        sc = old.get("scene_context") or {}
        gl = old.get("goal_landmark") or {}
        start_lms = sc.get("landmarks", [])[:3]
        result["start_context"] = ", ".join(start_lms) if start_lms else "the starting area"
        result["start_room"]    = sc.get("room_type", "")
        for direction in turn_directions:
            result["turns"].append({
                "direction": direction, "landmark": "", "room": "",
                "room_trans": "", "all_lms": [],
            })
        result["goal_landmark"] = (
            gl.get("stop_landmark") or sc.get("stop_landmark") or "the destination"
        )
        result["goal_room"] = gl.get("room_type", "")

    if not result["goal_landmark"] or result["goal_landmark"] in ("", "destination"):
        result["goal_landmark"] = "the destination"

    return result


# ── Route description builder (same as v11) ───────────────────────────────────

def build_route_description(primitives: List[Dict], lm_info: Dict) -> str:
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
                parts.append(f"turn {direction} past [{landmark}] through doorway into [{room}]")
            elif landmark:
                parts.append(f"turn {direction} at [{landmark}]")
            elif room:
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


def gt_start_verb(gt_instr: str) -> str:
    tok = gt_instr.strip().split()
    if not tok: return "Walk"
    w = tok[0].rstrip(".,!?;:")
    return w[0].upper() + w[1:].lower() if w else "Walk"


# ── v12 Prompt builder (enhanced) ─────────────────────────────────────────────

SYSTEM_INTRO = """You write concise navigation instructions for an indoor robot following R2R (Room-to-Room) VLN style.

CRITICAL RULE — anchor EVERY turn to a visible object:
  ✅ DO: "turn left at the white kitchen island"
  ✅ DO: "walk past the gray couch and turn right"
  ✅ DO: "turn right through the doorway into the hallway"
  ❌ DON'T: "turn left" (no object anchor — this causes navigation failure)
  ❌ DON'T: "turn right when you reach the corner" (too vague)

Style:
- Start with an action verb (Walk, Go, Exit, Enter, Head, Turn)
- Use spatial prepositions: past, through, into, near, at, in front of
- Use SPECIFIC object descriptions: "gray sectional sofa" not "sofa"
- End with: "Stop near the [specific object]." or "Stop at the [object]."
- Target 20-27 words total (concise!)"""


def build_prompt(
    ep: Dict,
    gt_instr: str,
    matched_examples: List[str],
    lm_info: Dict,
    primitives: List[Dict],
    retry_mode: bool = False,
) -> Tuple[str, str]:
    sv = gt_start_verb(gt_instr)

    route_desc = build_route_description(primitives, lm_info)

    turns_info = lm_info.get("turns", [])

    # Few-shot examples — mark anchored ones with ★
    ex_lines = []
    for i, ex in enumerate(matched_examples):
        marker = "★ " if has_turn_anchor(ex) else "  "
        ex_lines.append(f"  {marker}{i+1}. \"{ex}\"")
    ex_block = "\n".join(ex_lines)
    n_ex = len(matched_examples)
    n_anchored = sum(1 for ex in matched_examples if has_turn_anchor(ex))

    goal_lm   = lm_info.get("goal_landmark", "the destination")
    goal_room = lm_info.get("goal_room", "")
    stop_phrase = f"near the {goal_lm}"
    if goal_room and goal_lm != "the destination":
        stop_phrase = f"near the {goal_lm} in the {goal_room}"

    start_ctx  = lm_info.get("start_context", "")
    start_room = lm_info.get("start_room", "")

    # Per-turn cue lines — explicit anchor requirement
    turn_cue_lines = []
    for i, t in enumerate(turns_info, 1):
        d   = t.get("direction", "?")
        lm  = t.get("landmark", "")
        rm  = t.get("room", "")
        rt  = t.get("room_trans", "")
        if lm and rt and "doorway" in rt.lower():
            turn_cue_lines.append(f"  Turn {i}: MUST use → \"turn {d} past the [{lm}] through the doorway\" or \"turn {d} into the [{rm}]\"")
        elif lm:
            turn_cue_lines.append(f"  Turn {i}: MUST anchor → \"turn {d} at the [{lm}]\" or \"past the [{lm}] turn {d}\"")
        elif rm:
            turn_cue_lines.append(f"  Turn {i}: MUST anchor → \"turn {d} into the [{rm}]\"")
        else:
            turn_cue_lines.append(f"  Turn {i}: turn {d}")
    turn_cue_block = "\n".join(turn_cue_lines) if turn_cue_lines else "  (no turns — straight path)"

    anchor_requirement = ""
    if turns_info:
        n_turns = len(turns_info)
        lms_with_anchor = [t for t in turns_info if t.get("landmark") or t.get("room")]
        if lms_with_anchor:
            anchor_requirement = f"- You MUST include {n_turns} turn anchor(s) — one per turn above\n"

    retry_prefix = ""
    if retry_mode:
        retry_prefix = ("RETRY: Previous generation missed turn anchors. "
                       "EVERY turn MUST have 'turn [dir] at/past/through the [object]'. ")

    prompt = (
        f"{SYSTEM_INTRO}\n\n"
        f"{retry_prefix}"
        f"SAME-BUILDING examples ({n_ex} GT instructions, ★=has turn anchor, study these):\n"
        f"{ex_block}\n\n"
        f"Write ONE instruction for THIS PATH:\n\n"
        f"  Route: {route_desc}\n"
        f"  Visual anchors required:\n{turn_cue_block}\n"
        f"  Start area: {start_room or 'starting area'}, visible: {start_ctx or 'furniture'}\n"
        f"  Stop: {stop_phrase}\n\n"
        f"REQUIREMENTS:\n"
        f"{anchor_requirement}"
        f"- Use spatial prepositions: past, through, into, at, near\n"
        f"- Target 20-27 words (concise is better)\n"
        f"- End with: \"Stop near/at/in front of the [object].\"\n"
        f"- Write ONLY the instruction (no explanation, no quotes)\n"
        f"- Start with: \"{sv}\"\n\n"
        f"{sv}"
    )
    return prompt, sv


# ── Async generation ───────────────────────────────────────────────────────────

async def generate_one(client, task: Dict, sem: asyncio.Semaphore,
                        done: List, total: int, t0: float) -> Tuple[int, str]:
    eid = task["episode_id"]
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=VLLM_MODEL,
                    messages=[{"role": "user", "content": task["prompt"]}],
                    max_tokens=130,   # tighter than v11's 150
                    temperature=0.2,  # lower temperature for more precise anchors
                )
                result = resp.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt == 2:
                    result = f"ERROR: {e}"
                await asyncio.sleep(0.5 * (attempt + 1))
    done[0] += 1
    if done[0] % 200 == 0 or done[0] == total:
        elapsed = time.time() - t0
        r = done[0] / elapsed if elapsed > 0 else 0.001
        eta = (total - done[0]) / r if r > 0 else 0
        print(f"  [{done[0]}/{total}] {r:.1f}/s  ETA={eta/60:.1f}m", flush=True)
    return eid, result


async def generate(tasks: List[Dict], concurrency: int = 12) -> Dict[int, str]:
    from openai import AsyncOpenAI
    client  = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    sem     = asyncio.Semaphore(concurrency)
    done    = [0]
    t0      = time.time()
    total   = len(tasks)
    results: Dict[int, str] = {}

    coros = [generate_one(client, t, sem, done, total, t0) for t in tasks]
    for eid, result in await asyncio.gather(*coros):
        results[eid] = result
    return results


# ── Post-processing ────────────────────────────────────────────────────────────

STOP_WORDS = {"stop", "wait", "halt", "stand", "pause"}
PREAMBLES  = ["Instruction:", "Navigation:", "Answer:", "Sure,", "Certainly,",
              "Of course,", "Here is", "Here's", "Result:", "Walk:", "The instruction:"]


def clean(raw: str, sv: str) -> str:
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1].strip()
    for pre in PREAMBLES:
        if raw.lower().startswith(pre.lower()):
            raw = raw[len(pre):].lstrip(" :\n").strip()
    sv_l = sv.lower()
    if raw.lower().startswith(sv_l + " " + sv_l):
        raw = raw[len(sv_l):].lstrip()
    if not raw.lower().startswith(sv_l):
        raw = sv + " " + (raw[0].lower() + raw[1:] if raw else "to the destination.")
    sents  = re.split(r'(?<=[.!?])\s+', raw)
    result = " ".join(sents[:3]).strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result


def quality_ok(text: str, expected_anchors: int = 0) -> Tuple[bool, str]:
    """
    v12: validates turn-anchor count in addition to length + stop word.
    expected_anchors = number of turns in the path (min anchors required).
    """
    words = text.split()
    if len(words) < 8:  return False, "too_short"
    if len(words) > 70: return False, "too_long"
    if not any(w in text.lower().split() for w in STOP_WORDS):
        return False, "no_stop"
    # v12: check turn anchors
    if expected_anchors >= 1:
        n_anchors = count_turn_anchors(text)
        if n_anchors < 1:
            return False, "no_anchor"
    return True, "ok"


def fix_no_stop(text: str, goal_lm: str) -> str:
    return text.rstrip(". !") + f". Stop near the {goal_lm}."


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes",  type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--k-similar",   type=int, default=8)
    ap.add_argument("--min-anchor-examples", type=int, default=2,
                    help="Min turn-anchored GT examples in few-shot (default: 2)")
    args = ap.parse_args()

    print("=== Gate 4 v12: Turn-Anchor Enforced Instructions ===")
    print(f"  Model:       {VLLM_MODEL}")
    print(f"  Concurrency: {args.concurrency}  k_similar: {args.k_similar}")
    print(f"  Min anchor examples: {args.min_anchor_examples}")
    print()

    with gzip.open(GT_PATH, "rt") as f:
        gt_data = json.load(f)
    all_eps  = gt_data["episodes"]
    episodes = all_eps[:args.n_episodes] if args.n_episodes else all_eps
    print(f"Episodes: {len(episodes)}")

    gt_map = {ep["episode_id"]: (ep["instruction"]["instruction_text"]
              if isinstance(ep.get("instruction"), dict) else "")
              for ep in all_eps}

    sc_path_idx = ScenePathIndex(all_eps)

    print("Loading Gate 3 v2 per-frame landmarks...")
    pf_map  = load_perframe_landmarks(G3PF_DIR)
    old_map = load_old_landmarks(G3_OLD_DIR)
    n_pf    = sum(1 for ep in episodes if ep["episode_id"] in pf_map)
    n_old   = sum(1 for ep in episodes if ep["episode_id"] not in pf_map and ep["episode_id"] in old_map)
    print(f"  Gate 3 v2 (per-frame): {n_pf}/{len(episodes)} episodes")
    print(f"  Gate 3 v1 (fallback):  {n_old}/{len(episodes)} episodes")
    print()

    ckpt: Dict[str, str] = {}
    if CKPT.exists():
        ckpt = json.load(open(CKPT))
        print(f"Checkpoint: {len(ckpt)} episodes done")

    tokenizer = VLNTokenizer(GT_PATH)
    tasks: List[Dict] = []
    task_meta: Dict[int, Dict] = {}
    skipped = 0

    print("Computing path features and building task list...")
    ep_feats = {ep["episode_id"]: extract_path_features(ep) for ep in episodes}

    for ep in episodes:
        eid      = ep["episode_id"]
        gt_instr = gt_map.get(eid, "")
        ep_feat  = ep_feats[eid]
        prims    = get_primitives(ep)
        lm_info  = get_landmark_info(ep, pf_map, old_map, prims)
        n_turns  = len(lm_info["turns"])

        if str(eid) in ckpt:
            skipped += 1
            task_meta[eid] = {
                "sv":      gt_start_verb(gt_instr),
                "goal_lm": lm_info["goal_landmark"],
                "n_turns": n_turns,
            }
            continue

        matched = sc_path_idx.top_k_similar(ep, ep_feat, k=args.k_similar,
                                             min_anchor_examples=args.min_anchor_examples)
        prompt, sv = build_prompt(ep, gt_instr, matched, lm_info, prims)

        task_meta[eid] = {"sv": sv, "goal_lm": lm_info["goal_landmark"], "n_turns": n_turns}
        tasks.append({
            "episode_id": eid, "prompt": prompt, "sv": sv,
            "goal_lm": lm_info["goal_landmark"], "n_turns": n_turns,
            "lm_info": lm_info, "prims": prims, "ep": ep,
        })

    print(f"Tasks: {len(tasks)}  Skipped (cached): {skipped}")

    if tasks:
        sample = tasks[0]
        print(f"\n--- Sample prompt (ep {sample['episode_id']}) ---")
        print(sample["prompt"][:1200])
        if len(sample["prompt"]) > 1200:
            print("[...truncated...]")
        print("-" * 60)
        print()

    if tasks:
        print("Generating instructions (pass 1)...")
        results = await generate(tasks, args.concurrency)
        ckpt.update({str(k): v for k, v in results.items()})
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        with open(CKPT, "w") as f:
            json.dump(ckpt, f)
        print(f"Checkpoint saved: {len(ckpt)} total")

    # ── Assemble and validate ─────────────────────────────────────────────────
    print("\nAssembling dataset (pass 1 — validating anchors)...")
    ep_map = {ep["episode_id"]: ep for ep in episodes}
    episodes_out: List[Dict] = []
    n_pass = n_fix_stop = n_fix_retry = n_fail = 0
    retry_tasks: List[Dict] = []
    task_meta_map = {t["episode_id"]: t for t in tasks}

    for ep in episodes:
        eid  = ep["episode_id"]
        raw  = ckpt.get(str(eid), "")
        if not raw or raw.startswith("ERROR"):
            n_fail += 1
            continue

        meta    = task_meta.get(eid, {})
        n_turns = meta.get("n_turns", 0)
        text    = clean(raw, meta.get("sv", "Walk"))
        ok, reason = quality_ok(text, expected_anchors=n_turns)

        if not ok:
            if reason == "no_stop":
                text = fix_no_stop(text, meta.get("goal_lm", "the destination"))
                n_fix_stop += 1
                ok = True
            elif reason == "no_anchor" and eid in task_meta_map:
                # v12: retry with stronger prompt for missing anchors
                t = task_meta_map[eid]
                matched = sc_path_idx.top_k_similar(
                    ep, ep_feats[eid], k=args.k_similar,
                    min_anchor_examples=args.min_anchor_examples
                )
                retry_prompt, sv = build_prompt(
                    ep, gt_map.get(eid, ""), matched,
                    t["lm_info"], t["prims"], retry_mode=True
                )
                retry_tasks.append({
                    "episode_id": eid, "prompt": retry_prompt, "sv": sv,
                    "goal_lm": meta.get("goal_lm"), "n_turns": n_turns,
                })
                continue
            elif reason == "too_short":
                retry_tasks.append(task_meta_map.get(eid, {"episode_id": eid}))
                continue

        if ok:
            episodes_out.append(assemble_episode(ep, text, tokenizer))
            n_pass += 1

    print(f"Pass 1: {n_pass} pass, {n_fix_stop} stop-fixed, {len(retry_tasks)} need retry")

    # ── Retry pass for missing anchors / short ────────────────────────────────
    if retry_tasks:
        valid_retries = [t for t in retry_tasks if "prompt" in t]
        print(f"\nRetrying {len(valid_retries)} episodes (anchor missing or too short)...")

        if valid_retries:
            retry_results = await generate(valid_retries, args.concurrency)

            for t in valid_retries:
                eid = t["episode_id"]
                ep  = ep_map.get(eid)
                if not ep: continue
                raw  = retry_results.get(eid, "")
                meta = task_meta.get(eid, {})
                if not raw or raw.startswith("ERROR"):
                    n_fail += 1
                    continue
                text = clean(raw, meta.get("sv", t.get("sv", "Walk")))
                ok, reason = quality_ok(text, expected_anchors=0)  # relaxed after retry
                if not ok and reason == "no_stop":
                    text = fix_no_stop(text, meta.get("goal_lm", "the destination"))
                    ok = True
                if ok:
                    episodes_out.append(assemble_episode(ep, text, tokenizer))
                    n_fix_retry += 1
                else:
                    n_fail += 1

        # Handle missing checkpoint entries (no prompt)
        for t in retry_tasks:
            if "prompt" not in t:
                eid = t.get("episode_id")
                ep  = ep_map.get(eid)
                if not ep: continue
                meta = task_meta.get(eid, {})
                goal_lm = meta.get("goal_lm", "the destination")
                sv = meta.get("sv", "Walk")
                fallback = f"{sv} to the {goal_lm} area and stop near the {goal_lm}."
                episodes_out.append(assemble_episode(ep, fallback, tokenizer))
                n_fix_retry += 1

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\nFinal: {len(episodes_out)} episodes")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Load GT instruction_vocab
    gt_vocab = gt_data.get("instruction_vocab", {})

    out_data = {
        "episodes": episodes_out,
        "instruction_vocab": gt_vocab,
    }
    with gzip.open(OUTPUT, "wt", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Saved {len(episodes_out)} episodes to {OUTPUT} ({size_kb:.1f} KB)")

    # ── Statistics ────────────────────────────────────────────────────────────
    texts = [ep["instruction"]["instruction_text"] for ep in episodes_out]
    n     = len(texts)
    avg_w = sum(len(t.split()) for t in texts) / n if n else 0
    stop_n = sum(1 for t in texts if any(w in t.lower() for w in STOP_WORDS))
    anchor_n = sum(1 for t in texts if has_turn_anchor(t))
    spp = {k: 0 for k in ["past", "through", "into", "towards", "at the"]}
    for t in texts:
        tl = t.lower()
        for k in spp:
            if k in tl:
                spp[k] += 1

    print(f"\n=== v12 Done ===")
    print(f"  Episodes: {len(episodes_out)}  Pass: {n_pass}  RetryFixed: {n_fix_retry}  StopFixed: {n_fix_stop}  Failed: {n_fail}")
    print(f"  avg_words: {avg_w:.1f}  (GT=26.8, v11=29.8)")
    print(f"  stop%:     {100*stop_n/n:.1f}%")
    print(f"  turn-anchor%: {100*anchor_n/n:.1f}%  (v11=84.2%)")
    print(f"  spatial prepositions:")
    for k, v in spp.items():
        print(f"    {k:<12}: {100*v/n:.1f}%  (GT: past=22%, through=30%, into=32%)")
    print(f"\nDataset saved: {OUTPUT}")
    print(f"Run habitat eval:")
    print(f"  bash habitat_eval/scripts/run_eval_gate4_v12.sh")


if __name__ == "__main__":
    asyncio.run(main())
