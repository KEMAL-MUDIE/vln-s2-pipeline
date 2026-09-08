#!/usr/bin/env python3
"""
Gate 4 Visual v13 — GT-Distribution Matched Instructions

ROOT CAUSE ANALYSIS (from v11/v12 eval):
  v11 turn-anchor rate: 87.5%  (GT = 16.5%)  ← MASSIVELY OOD
  v12 turn-anchor rate: 95.0%  (GT = 16.5%)  ← EVEN MORE OOD
  v11/v12 stop rate:    100%   (GT = 53%)     ← OOD
  v11/v12 wait rate:    0%     (GT = 30%)     ← OOD

CORE INSIGHT:
  The model (InternVLA-N1) was trained on R2R GT instructions.
  GT has 16.5% turn-anchor rate and 30% "wait" usage.
  v11/v12 at 87-95% turn-anchoring are WAY out-of-distribution.
  The model likely ignores or misinterprets excessive visual anchors.

KEY CHANGES vs v12:
  1. ANCHOR RATE: Remove 0.15 priority bonus in ScenePathIndex (pure similarity)
  2. ANCHOR VALIDATION: Remove quality_ok turn-anchor check — never retry for missing anchors
  3. SYSTEM PROMPT: "Use a visual landmark ONLY when clearly distinctive. Most turns = simple 'turn left'."
  4. STOP/WAIT RANDOMIZATION: Match GT distribution (50% stop / 30% wait / 17% none / 3% both)
  5. SELECTIVE TURN CUES: Landmarks shown as OPTIONAL, not MUST-anchor
  6. LENGTH: 18-30w target (wider range matching GT min=5 max=119 avg=26.8)
  7. VOCABULARY: Prefer common object names (bed, couch, table) over decorative specifics

EXPECTED OUTCOME:
  - Turn-anchor rate: ~15-20% (matching GT)
  - Stop% = 50%, Wait% = 30%, Neither = 17%, Both = 3%
  - avg_words: 24-28 (close to GT=26.8)
  - SR improvement: should approach GT baseline since distribution matches training data

Output: outputs/datasets/val_unseen_generated_gemma_visual_v13.json.gz
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
OUTPUT      = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v13.json.gz"
CKPT        = ROOT / "outputs" / "gate4_visual_v13_checkpoint.json"

from gate2_path.path_analyzer import analyze_path
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset


# ── Turn-anchor detection (same regex as v11/v12) ─────────────────────────────

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


# ── Path utilities (same as v11/v12) ─────────────────────────────────────────

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
    v13: NO anchor priority bonus — pure path similarity (matches GT distribution naturally).
    """
    def __init__(self, all_eps: List[Dict]):
        self.by_scene: Dict[str, List[Dict]] = defaultdict(list)
        print("Building scene-path index (v13 — pure similarity, no anchor priority)...", flush=True)
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

    def top_k_similar(self, ep: Dict, ep_feat: Dict, k: int = 8) -> List[str]:
        """
        Return top-k most path-similar GT instructions from same scene.
        v13: pure path similarity — no anchor boost. Natural GT anchor rate in examples.
        """
        sc   = ep["scene_id"].split("/")[-2]
        pool = [e for e in self.by_scene.get(sc, [])
                if e["episode_id"] != ep["episode_id"]]
        if not pool:
            return []

        # Pure path similarity — no anchor bias (v13 key change vs v12)
        scored = sorted(
            [(path_similarity(ep_feat, e["feat"]), e) for e in pool],
            key=lambda x: -x[0]
        )
        return [e["instruction"] for _, e in scored[:k]]


# ── Gate 3 v2 landmark loading (same as v11/v12) ─────────────────────────────

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
    v13: landmarks are OPTIONAL signals, not requirements.
         Also applies deduplication from v12.
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
        prev_landmark = None  # dedup: track previous turn landmark

        for i, turn_det in enumerate(pf_turns):
            direction  = turn_directions[i] if i < len(turn_directions) else "left"
            main       = turn_det.get("main_landmark") or ""
            others     = turn_det.get("landmarks", [])
            room       = turn_det.get("room", "")
            room_trans = turn_det.get("room_transition", "")
            all_lms    = ([main] + [lm for lm in others if lm != main])[:4] if main else others[:4]

            best = main or (others[0] if others else "")
            if best and best == prev_landmark and len(all_lms) > 1:
                best = all_lms[1]  # dedup consecutive same landmark

            prev_landmark = best or None
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


# ── Route description builder ─────────────────────────────────────────────────

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
            # v13: always show landmark if available (as context, not requirement)
            if landmark:
                parts.append(f"turn {direction} [landmark: {landmark}]")
            elif room:
                parts.append(f"turn {direction} → {room}")
            else:
                parts.append(f"turn {direction}")
            turn_idx += 1
        elif t == "elevation":
            dir_str = p.get("direction", "up")
            parts.append(f"{'up' if dir_str == 'up' else 'down'} stairs")
        elif t == "stop":
            goal_lm = lm_info.get("goal_landmark", "destination")
            parts.append(f"stop at [{goal_lm}]")

    return " → ".join(parts) if parts else "walk forward → stop"


def gt_start_verb(gt_instr: str) -> str:
    tok = gt_instr.strip().split()
    if not tok: return "Walk"
    w = tok[0].rstrip(".,!?;:")
    return w[0].upper() + w[1:].lower() if w else "Walk"


def enforce_stop_phrase(text: str, phrase: str, stop_type: str) -> str:
    """Remove duplicate stop/wait clauses and enforce the single canonical phrase."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    cleaned = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if re.match(r'^(Stop|Wait|Halt|Stand)\b', s, re.IGNORECASE):
            continue
        s = re.sub(
            r'(?:'
            r'(?:[,;]\s*(?:(?:and|then|to|continue|until|once|when)\s+)*(?:stop|wait|halt)s?[^.!?]*)'
            r'|(?:\s+(?:(?:and|then|to|continue|until|once|when)\s+)+(?:\w+\s+)?(?:stop|wait|halt)s?[^.!?]*)'
            r')[.!?]?$',
            '.', s, flags=re.IGNORECASE
        )
        s = re.sub(r'\.{2,}', '.', s).strip()
        if s and s not in ('.', '!', '?'):
            cleaned.append(s)
    text = ' '.join(cleaned).strip()
    text = re.sub(r'\.{2,}', '.', text).strip()
    if stop_type == "none":
        return (text.rstrip('. ') + '.') if text else '.'
    return text.rstrip('. ').rstrip('.') + '. ' + phrase


# ── v13 Stop/Wait phrase generator — matches GT distribution ──────────────────
# GT: stop=50.8%, wait=30.5%, neither=16.5%, both=2.2%

def choose_stop_phrase(goal_lm: str, goal_room: str, eid: int) -> Tuple[str, str]:
    """
    Returns (stop_instruction, stop_type) where stop_type is 'stop'/'wait'/'none'.
    Uses eid-seeded random for deterministic output.
    GT distribution: stop 51% / wait 30% / neither 17% / both 2%.
    """
    rng = random.Random(eid ^ 0xA3B7C5)
    r = rng.random()

    if goal_lm and goal_lm != "the destination":
        lm_ref = f"the {goal_lm}" if not goal_lm.startswith("the ") else goal_lm
    else:
        lm_ref = "the " + (goal_room if goal_room else "destination")

    if r < 0.51:  # stop (matches GT 50.8%)
        stop_patterns = [
            f"Stop near {lm_ref}.",
            f"Stop at {lm_ref}.",
            f"Stop in front of {lm_ref}.",
            f"Stop by {lm_ref}.",
        ]
        if goal_room and not goal_lm or goal_lm == "the destination":
            stop_patterns = [
                f"Stop in the {goal_room}.",
                f"Stop at the entrance to the {goal_room}.",
            ]
        phrase = rng.choice(stop_patterns)
        return phrase, "stop"
    elif r < 0.81:  # wait (matches GT 30.5%)
        wait_patterns = [
            f"Wait near {lm_ref}.",
            f"Wait by {lm_ref}.",
            f"Wait there.",
        ]
        phrase = rng.choice(wait_patterns)
        return phrase, "wait"
    else:  # neither (matches GT 16.5%)
        return "", "none"


# ── v13 System prompt — GT-style guidance ────────────────────────────────────

SYSTEM_INTRO = """You write concise R2R (Room-to-Room) navigation instructions for an indoor robot.

Style guidelines (follow strictly):
- Be CONCISE: 18-30 words total
- Most turns: just say "turn left" or "turn right" — no object needed
- Occasionally (1 in 5 turns): add a visual anchor ONLY if the landmark is large and obvious:
    ✓ "walk past the bed and turn right"
    ✓ "turn left at the kitchen island"
    ✓ "turn right into the hallway"
  NOT for tiny/decorative objects like "silver leaf decoration" or "recessed niche"
- Use natural spatial language: through, into, past, towards, across
- Start with an action verb: Walk, Go, Exit, Enter, Head, Turn, Leave
- Ending is flexible: "Stop near X." OR "Wait near X." OR "Wait there." OR no explicit stop"""


def build_prompt(
    ep: Dict,
    gt_instr: str,
    matched_examples: List[str],
    lm_info: Dict,
    primitives: List[Dict],
    stop_phrase: str,
    stop_type: str,
) -> Tuple[str, str]:
    sv = gt_start_verb(gt_instr)

    route_desc = build_route_description(primitives, lm_info)
    turns_info = lm_info.get("turns", [])

    # Few-shot examples (no markers — don't bias toward anchored ones)
    ex_lines = [f"  {i+1}. \"{ex}\"" for i, ex in enumerate(matched_examples)]
    ex_block = "\n".join(ex_lines)
    n_ex = len(matched_examples)

    goal_lm   = lm_info.get("goal_landmark", "the destination")
    goal_room = lm_info.get("goal_room", "")
    start_ctx  = lm_info.get("start_context", "")
    start_room = lm_info.get("start_room", "")

    # Per-turn info — landmarks are OPTIONAL context, not requirements
    turn_cue_lines = []
    for i, t in enumerate(turns_info, 1):
        d   = t.get("direction", "?")
        lm  = t.get("landmark", "")
        rm  = t.get("room", "")
        rt  = t.get("room_trans", "")
        if lm and rt and "doorway" in rt.lower():
            turn_cue_lines.append(
                f"  Turn {i}: turn {d} [optional landmark visible: {lm}, doorway into {rm}]"
            )
        elif lm:
            turn_cue_lines.append(
                f"  Turn {i}: turn {d} [optional landmark nearby: {lm}]"
            )
        elif rm:
            turn_cue_lines.append(f"  Turn {i}: turn {d} [entering: {rm}]")
        else:
            turn_cue_lines.append(f"  Turn {i}: turn {d}")
    turn_cue_block = "\n".join(turn_cue_lines) if turn_cue_lines else "  (straight path — no turns)"

    # Stop/wait instruction based on GT distribution
    if stop_type == "stop":
        ending_req = f'- End with: "{stop_phrase}"'
    elif stop_type == "wait":
        ending_req = f'- End with: "{stop_phrase}"'
    else:
        ending_req = "- No explicit stop/wait required — natural ending is fine"

    prompt = (
        f"{SYSTEM_INTRO}\n\n"
        f"Same-building GT examples (study style and length):\n"
        f"{ex_block}\n\n"
        f"Write ONE instruction for this path:\n\n"
        f"  Route: {route_desc}\n"
        f"  Turn details (landmarks are OPTIONAL — use only if clearly visible):\n"
        f"{turn_cue_block}\n"
        f"  Start: {start_room or 'starting area'}"
        + (f", visible: {start_ctx}" if start_ctx else "") + "\n"
        f"  Goal: {goal_lm}" + (f" in {goal_room}" if goal_room else "") + "\n\n"
        f"REQUIREMENTS:\n"
        f"- 18-30 words total (concise!)\n"
        f"{ending_req}\n"
        f"- Write ONLY the instruction (no explanation, no quotes)\n"
        f"- Start with: \"{sv}\"\n\n"
        f"{sv}"
    )
    return prompt, sv


# ── Async generation ──────────────────────────────────────────────────────────

async def generate_one(client, task: Dict, sem: asyncio.Semaphore,
                       done: List, total: int, t0: float) -> Tuple[int, str]:
    eid = task["episode_id"]
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=VLLM_MODEL,
                    messages=[{"role": "user", "content": task["prompt"]}],
                    max_tokens=150,    # slightly larger budget for natural language
                    temperature=0.25,  # balanced temperature for variety
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


# ── Post-processing ───────────────────────────────────────────────────────────

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


def quality_ok(text: str) -> Tuple[bool, str]:
    """
    v13: only checks length. No turn-anchor requirement. Stop is optional.
    """
    words = text.split()
    if len(words) < 6:  return False, "too_short"
    if len(words) > 80: return False, "too_long"
    return True, "ok"


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes",  type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--k-similar",   type=int, default=8)
    args = ap.parse_args()

    print("=== Gate 4 v13: GT-Distribution Matched Instructions ===")
    print(f"  Model:       {VLLM_MODEL}")
    print(f"  Concurrency: {args.concurrency}  k_similar: {args.k_similar}")
    print(f"  Key changes: no anchor priority, no anchor validation, stop/wait distribution matches GT")
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

        # Choose stop/wait phrase using GT distribution
        stop_phrase, stop_type = choose_stop_phrase(
            lm_info["goal_landmark"], lm_info["goal_room"], eid
        )

        if str(eid) in ckpt:
            skipped += 1
            task_meta[eid] = {
                "sv":        gt_start_verb(gt_instr),
                "goal_lm":   lm_info["goal_landmark"],
                "n_turns":   n_turns,
                "stop_phrase": stop_phrase,
                "stop_type": stop_type,
            }
            continue

        matched = sc_path_idx.top_k_similar(ep, ep_feat, k=args.k_similar)
        prompt, sv = build_prompt(ep, gt_instr, matched, lm_info, prims, stop_phrase, stop_type)

        task_meta[eid] = {
            "sv": sv, "goal_lm": lm_info["goal_landmark"],
            "n_turns": n_turns, "stop_phrase": stop_phrase, "stop_type": stop_type,
        }
        tasks.append({
            "episode_id": eid, "prompt": prompt, "sv": sv,
            "goal_lm": lm_info["goal_landmark"], "n_turns": n_turns,
        })

    print(f"Tasks: {len(tasks)}  Skipped (cached): {skipped}")

    # Print stop/wait distribution for verification
    stop_types = [task_meta[ep["episode_id"]]["stop_type"] for ep in episodes if ep["episode_id"] in task_meta]
    from collections import Counter
    st_counter = Counter(stop_types)
    n_total = len(stop_types)
    print(f"\nStop/Wait distribution (target: stop=51% wait=30% none=17%):")
    for st, cnt in sorted(st_counter.items()):
        print(f"  {st}: {cnt} ({100*cnt/n_total:.1f}%)")
    print()

    if tasks:
        sample = tasks[0]
        print(f"--- Sample prompt (ep {sample['episode_id']}) ---")
        print(sample["prompt"][:1500])
        if len(sample["prompt"]) > 1500:
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

    # ── Assemble and validate ────────────────────────────────────────────────
    print("\nAssembling dataset...")
    ep_map = {ep["episode_id"]: ep for ep in episodes}
    episodes_out: List[Dict] = []
    n_pass = n_fail = n_fix = 0

    for ep in episodes:
        eid  = ep["episode_id"]
        raw  = ckpt.get(str(eid), "")
        if not raw or raw.startswith("ERROR"):
            n_fail += 1
            continue

        meta = task_meta.get(eid, {})
        sv   = meta.get("sv", "Walk")
        text = clean(raw, sv)
        ok, reason = quality_ok(text)

        if not ok:
            if reason == "too_short":
                # Fallback: use stop phrase or simple instruction
                goal_lm = meta.get("goal_lm", "the destination")
                stop_phrase = meta.get("stop_phrase", "")
                text = f"{sv} to the destination area{'. ' + stop_phrase if stop_phrase else '.'}"
                n_fix += 1
                ok = True
            elif reason == "too_long":
                # Truncate to 3 sentences
                sents = re.split(r'(?<=[.!?])\s+', text)
                text = " ".join(sents[:3]).strip()
                n_fix += 1
                ok = True

        if ok:
            stop_phrase = meta.get("stop_phrase", "")
            stop_type = meta.get("stop_type", "none")
            text = enforce_stop_phrase(text, stop_phrase, stop_type)
            episodes_out.append(assemble_episode(ep, text, tokenizer))
            n_pass += 1
        else:
            n_fail += 1

    # ── Save ─────────────────────────────────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    gt_vocab = gt_data.get("instruction_vocab", {})
    out_data = {"episodes": episodes_out, "instruction_vocab": gt_vocab}
    with gzip.open(OUTPUT, "wt", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Saved {len(episodes_out)} episodes to {OUTPUT} ({size_kb:.1f} KB)")

    # ── Statistics ───────────────────────────────────────────────────────────
    texts    = [ep["instruction"]["instruction_text"] for ep in episodes_out]
    n        = len(texts)
    avg_w    = sum(len(t.split()) for t in texts) / n if n else 0
    stop_n   = sum(1 for t in texts if re.search(r'\bstop\b', t, re.IGNORECASE))
    wait_n   = sum(1 for t in texts if re.search(r'\bwait\b', t, re.IGNORECASE))
    neither_n = sum(1 for t in texts
                    if not re.search(r'\bstop\b', t, re.IGNORECASE)
                    and not re.search(r'\bwait\b', t, re.IGNORECASE))
    anchor_n = sum(1 for t in texts if has_turn_anchor(t))
    spp = {k: 0 for k in ["past", "through", "into", "towards", "at the"]}
    for t in texts:
        tl = t.lower()
        for k in spp: spp[k] += k in tl

    from collections import Counter as _Counter
    first_words = _Counter(t.split()[0].lower() for t in texts if t.split())

    print(f"\n=== v13 Done ===")
    print(f"  Episodes: {n}  Pass: {n_pass}  Fixed: {n_fix}  Failed: {n_fail}")
    print(f"  avg_words:    {avg_w:.1f}  (GT=26.8)")
    print(f"  stop%:        {100*stop_n/n:.1f}%  (GT=50.8%)")
    print(f"  wait%:        {100*wait_n/n:.1f}%  (GT=30.5%)")
    print(f"  neither%:     {100*neither_n/n:.1f}%  (GT=16.5%)")
    print(f"  turn-anchor%: {100*anchor_n/n:.1f}%  (GT=16.5%, v12=95%)")
    print(f"  top start verbs: {first_words.most_common(6)}")
    print(f"  spatial prepositions:")
    for k, v in spp.items():
        print(f"    {k:<12}: {100*v/n:.1f}%  (GT: past=22%, through=30%, into=32%)")
    print(f"\nDataset saved: {OUTPUT}")
    print(f"Run habitat eval:")
    print(f"  bash habitat_eval/scripts/run_eval_gate4_v13.sh")


if __name__ == "__main__":
    asyncio.run(main())
