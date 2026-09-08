#!/usr/bin/env python3
"""
Gate 4 Visual v18 — Vision-Grounded Instructions

ROOT CAUSE OF 25pp GAP (diagnosed from v11-v17 analysis):
  All prior annotators (v11-v17) use TEXT-ONLY landmark names from 3D annotation
  metadata (gate3_perframe). These are semantic category labels, NOT what the
  agent's camera actually sees. Example: metadata says "glass dining table" as
  main_landmark at turn_1, but the actual rendered frame shows "grey structural
  pillar" as the most visually prominent object. The CMA agent sees the pillar;
  the instruction says turn at a table → mismatch degrades SR.

THE FIX (v18): TWO-PHASE VISION-GROUNDED ANNOTATOR
  Phase 1 — Visual description extraction:
    For each episode, for each key frame (all renders from poses.json), send
    the actual RGB image to Gemma-4-31B vision. Ask: what is the most visually
    distinctive landmark at this turn/goal? Save text descriptions.
    Checkpoint: outputs/gate4_v18_phase1_checkpoint.json
    ~6870 total image calls (avg 3.74 frames/episode × 1839 episodes)
    Runtime: ~30-45 min at concurrency=8

  Phase 2 — Instruction generation (using vision descriptions):
    Same structure as v17 (P_ANCHOR=0.21, stop/wait GT-matched distribution,
    room transitions, enforce_stop_phrase). Replace text-only landmark names
    with vision-extracted descriptions for:
      - Turn anchors (~17% episodes, P_ANCHOR=0.21 selection)
      - Goal landmark (100% episodes — used in stop phrase)
    Checkpoint: outputs/gate4_v18_phase2_checkpoint.json
    Output: outputs/datasets/val_unseen_generated_gemma_visual_v18.json.gz
"""
import asyncio
import base64
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

GT_PATH    = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
G3PF_DIR   = ROOT / "outputs" / "gate3_perframe"
G3_OLD_DIR = ROOT / "outputs" / "gate3_landmarks"
RF_DIR     = ROOT / "outputs" / "rendered_frames"
OUTPUT     = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v18.json.gz"
P1_CKPT    = ROOT / "outputs" / "gate4_v18_phase1_checkpoint.json"
P2_CKPT    = ROOT / "outputs" / "gate4_v18_phase2_checkpoint.json"

from gate2_path.path_analyzer import analyze_path
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode

# ── Constants ─────────────────────────────────────────────────────────────────

P_ANCHOR_EPISODE = 0.21   # ~17.5% episodes get one turn anchor (GT=16.5%)
P_STOP = 0.508
P_WAIT = 0.305

# ── Pattern detection ─────────────────────────────────────────────────────────

TURN_ANCHOR_RE = re.compile(
    r'turn\s+(left|right)\s+(at|past|through|into|around)\s+the\s+\w',
    re.IGNORECASE
)
WALK_PAST_RE = re.compile(r'(walk|go|head|move)\s+past\s+the\s+\w', re.IGNORECASE)
STOP_RE = re.compile(r'\b(stop|halt|stand)\b', re.IGNORECASE)
WAIT_RE = re.compile(r'\b(wait|waiting)\b', re.IGNORECASE)


def has_turn_anchor(text: str) -> bool:
    return bool(TURN_ANCHOR_RE.search(text) or WALK_PAST_RE.search(text))


# ── Path utilities ────────────────────────────────────────────────────────────

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
        "turns":       turns,
        "n_turns":     len(turns),
        "total_dist":  summary.get("total_distance_m", total_dist),
        "n_waypoints": summary.get("n_waypoints", len(ep["reference_path"])),
    }


def turn_edit_dist(t1, t2):
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


def path_similarity(f1, f2):
    t1, t2 = f1["turns"], f2["turns"]
    max_t    = max(len(t1), len(t2), 1)
    turn_sim = 1.0 - turn_edit_dist(t1, t2) / max_t
    d1, d2   = f1["total_dist"], f2["total_dist"]
    dist_sim = 1.0 - abs(d1 - d2) / max(d1 + d2, 0.01)
    n1, n2   = f1["n_waypoints"], f2["n_waypoints"]
    wpt_sim  = 1.0 - abs(n1 - n2) / max(n1, n2, 1)
    return 0.6 * turn_sim + 0.25 * dist_sim + 0.15 * wpt_sim


class ScenePathIndex:
    def __init__(self, all_eps):
        self.by_scene = defaultdict(list)
        n_anchor = 0
        for ep in all_eps:
            sc    = ep["scene_id"].split("/")[-2]
            instr = (ep.get("instruction", {}).get("instruction_text", "")
                     if isinstance(ep.get("instruction"), dict) else "")
            if not instr.strip(): continue
            if has_turn_anchor(instr): n_anchor += 1
            self.by_scene[sc].append({
                "episode_id": ep["episode_id"],
                "instruction": instr.strip(),
                "feat": extract_path_features(ep),
            })
        total = sum(len(v) for v in self.by_scene.values())
        print(f"ScenePathIndex: {len(self.by_scene)} scenes, {total} eps, "
              f"{n_anchor} anchored ({100*n_anchor/max(total,1):.1f}%)")

    def top_k(self, ep, ep_feat, k=8):
        sc   = ep["scene_id"].split("/")[-2]
        pool = [e for e in self.by_scene.get(sc, [])
                if e["episode_id"] != ep["episode_id"]]
        if not pool: return []
        scored = sorted(
            [(path_similarity(ep_feat, e["feat"]), e) for e in pool],
            key=lambda x: -x[0]
        )
        return [e["instruction"] for _, e in scored[:k]]


# ── Landmark loading (text fallback) ─────────────────────────────────────────

def load_perframe_landmarks(g3pf_dir):
    lm_map = {}
    if not g3pf_dir.exists(): return lm_map
    for fp in g3pf_dir.glob("episode_*.json"):
        try:
            d = json.load(open(fp))
            eid = d.get("episode_id") or int(fp.stem.replace("episode_","").lstrip("0") or "0")
            lm_map[eid] = d
        except: pass
    return lm_map


def load_old_landmarks(g3_dir):
    lm_map = {}
    for fp in g3_dir.glob("episode_*.json"):
        try:
            d = json.load(open(fp))
            eid = d.get("episode_id") or int(fp.stem.replace("episode_","").lstrip("0") or "0")
            lm_map[eid] = d
        except: pass
    return lm_map


def get_landmark_info_text(ep, pf_map, old_map, primitives) -> Dict:
    """Get text-based landmark info (fallback when vision fails)."""
    eid = ep["episode_id"]
    pf  = pf_map.get(eid)
    old = old_map.get(eid, {})
    turn_dirs = [p["type"].replace("_turn","") for p in primitives
                 if p["type"] in ("left_turn","right_turn")]
    result = {"has_perframe": pf is not None, "start_context": "", "start_room": "",
              "turns": [], "goal_landmark": "the destination", "goal_room": ""}
    if pf:
        start = pf.get("start") or {}
        result["start_room"] = start.get("room","")
        slms = start.get("landmarks",[])[:3]
        result["start_context"] = ", ".join(slms) if slms else ""
        pf_turns = pf.get("turns",[])
        prev_lm = None
        for i, td in enumerate(pf_turns):
            d = turn_dirs[i] if i < len(turn_dirs) else "left"
            main   = td.get("main_landmark","")
            others = td.get("landmarks",[])
            all_lms = ([main] + [l for l in others if l!=main])[:4] if main else others[:4]
            best = main or (others[0] if others else "")
            if best and best == prev_lm and len(all_lms)>1: best = all_lms[1]
            prev_lm = best or None
            result["turns"].append({
                "direction": d, "landmark": best,
                "label": td.get("label", f"turn_{i+1}"),  # actual label from poses.json
                "room": td.get("room",""), "room_trans": td.get("room_transition","")
            })
        for i in range(len(pf_turns), len(turn_dirs)):
            result["turns"].append({"direction":turn_dirs[i],"landmark":"","room":"","room_trans":""})
        goal = pf.get("goal") or {}
        result["goal_landmark"] = (goal.get("stop_landmark","") or
                                   goal.get("main_landmark","the destination"))
        result["goal_room"] = goal.get("room","")
    else:
        sc = old.get("scene_context") or {}
        gl = old.get("goal_landmark") or {}
        result["start_context"] = ", ".join(sc.get("landmarks",[])[:3])
        result["start_room"] = sc.get("room_type","")
        for d in turn_dirs:
            result["turns"].append({"direction":d,"landmark":"","room":"","room_trans":""})
        result["goal_landmark"] = (gl.get("stop_landmark") or
                                   sc.get("stop_landmark") or "the destination")
        result["goal_room"] = gl.get("room_type","")
    if not result["goal_landmark"] or result["goal_landmark"] == "destination":
        result["goal_landmark"] = "the destination"
    return result


# ── Phase 1: Vision description extraction ───────────────────────────────────

VISION_PROMPT_TURN = (
    "A robot is navigating indoors and is about to turn at this location. "
    "Identify the single most visually distinctive object or landmark visible at this turning point. "
    "Describe it in 3-8 words using specific details (color, material, or shape). "
    "Examples: 'grey concrete pillar', 'dark wood bookshelf', 'white marble fireplace', "
    "'glass dining table with orange chairs', 'wooden staircase railing'. "
    "Reply with ONLY the description, nothing else."
)

VISION_PROMPT_GOAL = (
    "A robot has arrived at its navigation destination. "
    "Identify the most visually prominent object that marks this stopping location. "
    "Describe it in 3-8 words using specific details (color, material, or shape). "
    "Examples: 'silver accent chair by windows', 'white reception desk', 'wooden dining table', "
    "'stone fireplace with wooden mantel'. "
    "Reply with ONLY the description, nothing else."
)

VISION_PROMPT_START = (
    "A robot starts navigation from this indoor location. "
    "Identify the most visually distinctive object in this starting area. "
    "Describe it in 3-8 words using specific details (color, material, or shape). "
    "Reply with ONLY the description, nothing else."
)


def get_vision_prompt(label: str) -> str:
    if label == "start":
        return VISION_PROMPT_START
    elif label == "goal":
        return VISION_PROMPT_GOAL
    else:
        return VISION_PROMPT_TURN


def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def clean_vision_desc(raw: str) -> Optional[str]:
    """Post-process raw vision description. Return None if unusable."""
    raw = raw.strip().strip('"\'').strip()
    # Remove common preambles
    for prefix in ["Description:", "The landmark is", "I see", "I can see", "The most", "Based on"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].strip(" :-").strip()
    # Truncate at first sentence boundary if too long
    first_sent = re.split(r'[.!?\n]', raw)[0].strip()
    if first_sent:
        raw = first_sent
    # Clean up
    raw = re.sub(r'\s+', ' ', raw).strip()
    words = raw.split()
    if len(words) < 2 or len(words) > 15:
        return None
    # Filter generic/useless descriptions
    generic = {"a room", "an area", "furniture", "the room", "indoor", "a space", "nothing"}
    if raw.lower() in generic:
        return None
    return raw


async def vision_describe_one(client, eid: str, image_path: Path,
                               label: str, sem: asyncio.Semaphore,
                               done: List, total: int, t0: float) -> Dict:
    """Single async vision call for one frame."""
    async with sem:
        prompt_text = get_vision_prompt(label)
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
                max_tokens=30,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            desc = clean_vision_desc(raw)
            result = {"eid": eid, "label": label, "desc": desc, "raw": raw, "ok": True}
        except Exception as e:
            result = {"eid": eid, "label": label, "desc": None, "raw": "", "ok": False, "error": str(e)}

    done[0] += 1
    if done[0] % 500 == 0 or done[0] == total:
        elapsed = time.time() - t0
        r = done[0] / elapsed if elapsed > 0 else 0.001
        eta = (total - done[0]) / r if r > 0 else 0
        print(f"  [Phase1 {done[0]}/{total}] {r:.1f}/s  ETA={eta/60:.1f}m", flush=True)
    return result


async def run_phase1(episodes: List[Dict], existing: Dict, concurrency: int = 8) -> Dict:
    """
    Phase 1: Extract visual descriptions from rendered frames.
    Returns {episode_id_str: {label: desc_str_or_None}}
    """
    print(f"\n=== Phase 1: Vision Description Extraction ===")
    print(f"  concurrency: {concurrency}  max_tokens_per_call: 30")

    # Build task list — only for episodes/frames not in checkpoint
    tasks = []
    skipped = 0
    for ep in episodes:
        eid = str(ep["episode_id"])
        ep_dir = RF_DIR / f"episode_{int(eid):06d}"
        poses_f = ep_dir / "poses.json"
        if not poses_f.exists():
            continue
        if eid in existing:
            skipped += 1
            continue
        try:
            poses = json.load(open(poses_f))
        except Exception:
            continue
        for frame in poses.get("frames", []):
            img_path = ep_dir / frame["path"]
            if not img_path.exists():
                continue
            tasks.append({
                "eid": eid,
                "label": frame["label"],
                "image_path": img_path,
            })

    print(f"  Frame tasks: {len(tasks)}  Episodes skipped (checkpoint): {skipped}")

    if not tasks:
        print("  All frames already in checkpoint.")
        return existing

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    sem  = asyncio.Semaphore(concurrency)
    done = [0]
    t0   = time.time()
    total = len(tasks)

    coros = [
        vision_describe_one(
            client, t["eid"], t["image_path"], t["label"],
            sem, done, total, t0
        )
        for t in tasks
    ]

    # Group results by episode
    new_results = {}  # eid → {label: desc}
    for result in await asyncio.gather(*coros):
        eid   = result["eid"]
        label = result["label"]
        if eid not in new_results:
            new_results[eid] = {}
        new_results[eid][label] = result["desc"]  # may be None if unusable

    # Merge with existing checkpoint
    merged = dict(existing)
    for eid, frame_descs in new_results.items():
        merged[eid] = frame_descs

    elapsed = time.time() - t0
    ok = sum(1 for fd in new_results.values() for d in fd.values() if d)
    fail = sum(1 for fd in new_results.values() for d in fd.values() if not d)
    print(f"\nPhase 1 done in {elapsed/60:.1f}m: "
          f"{ok} good descriptions, {fail} unusable/failed, {skipped} skipped")

    # Save checkpoint
    P1_CKPT.parent.mkdir(parents=True, exist_ok=True)
    with open(P1_CKPT, "w") as f:
        json.dump(merged, f)
    print(f"Phase 1 checkpoint saved: {P1_CKPT}")
    return merged


# ── v18 landmark info: merge text + vision ────────────────────────────────────

def get_landmark_info_v18(ep, pf_map, old_map, primitives, vision_map: Dict) -> Dict:
    """
    Build landmark info for v18: uses vision descriptions when available,
    falls back to gate3_perframe text labels.
    Room names always come from gate3_perframe (correct, even if landmark names differ).
    """
    eid = ep["episode_id"]
    eid_str = str(eid)
    vis = vision_map.get(eid_str, {})  # {label: desc_or_None}

    # Start with text-based info (provides room names, turn structure)
    info = get_landmark_info_text(ep, pf_map, old_map, primitives)

    # Override goal landmark with vision description
    goal_vis = vis.get("goal")
    if goal_vis:
        info["goal_landmark"] = goal_vis
    elif info["goal_landmark"] == "the destination":
        pass  # keep fallback

    # Override turn landmarks with vision descriptions
    # Use actual label from gate3_perframe (may be turn_2, turn_3 etc. — non-sequential)
    for turn in info["turns"]:
        label = turn.get("label", "")
        if label:
            vis_lm = vis.get(label)
            if vis_lm:
                turn["landmark"] = vis_lm
        # room and room_trans always from gate3_perframe (they're correct)

    # Use vision start description as context if text context is empty
    start_vis = vis.get("start")
    if start_vis and not info["start_context"]:
        info["start_context"] = start_vis

    return info


# ── Episode-level landmark control (same as v17) ─────────────────────────────

def episode_selective_turns(lm_info: Dict, eid: int, p_anchor: float = P_ANCHOR_EPISODE) -> List[Dict]:
    """
    v17/v18: preserve room names for ALL turns where room CHANGES.
    Only P_ANCHOR_EPISODE of episodes get ONE object landmark (vision or text).
    """
    rng = random.Random(eid ^ 0xF5D8_E27A)
    turns = lm_info.get("turns", [])
    start_room = lm_info.get("start_room", "")

    prev_room = start_room
    result = []
    for t in turns:
        t_room = t.get("room", "")
        room_to_show = t_room if (t_room and t_room.lower() != prev_room.lower()) else ""
        result.append({
            "direction": t["direction"],
            "landmark": "",
            "room": room_to_show,
            "room_trans": t.get("room_trans", ""),
        })
        if t_room:
            prev_room = t_room

    if rng.random() < p_anchor:
        turns_with_lm = [(i, t) for i, t in enumerate(turns) if t.get("landmark")]
        if turns_with_lm:
            best_idx, best_turn = max(turns_with_lm,
                                      key=lambda x: len(x[1].get("landmark", "")))
            result[best_idx]["landmark"] = best_turn["landmark"]

    return result


# ── Stop/wait — GT distribution ───────────────────────────────────────────────

def choose_stop(goal_lm: str, goal_room: str, eid: int) -> Tuple[str, str]:
    rng = random.Random(eid ^ 0xA3B7)
    r   = rng.random()
    # Allow up to 8 words; strip trailing articles/prepositions that sound incomplete
    lm_words = goal_lm.split()[:8]
    lm = " ".join(lm_words)
    lm = re.sub(r'\s+(a|an|the|to|in|on|by|at|of|leading)\s*$', '', lm, flags=re.IGNORECASE).strip()
    if not lm.startswith("the "):
        lm = f"the {lm}"
    if goal_lm == "the destination":
        lm = "your destination"

    if r < P_STOP:
        patterns = [f"Stop near {lm}.", f"Stop at {lm}.", f"Stop in front of {lm}."]
        if goal_room and not (goal_lm and goal_lm != "the destination"):
            patterns = [f"Stop in the {goal_room}.", f"Stop at the entrance."]
        return rng.choice(patterns), "stop"
    elif r < P_STOP + P_WAIT:
        return rng.choice([f"Wait near {lm}.", f"Wait by {lm}.", "Wait there."]), "wait"
    else:
        return "", "none"


def enforce_stop_phrase(text: str, phrase: str, stop_type: str) -> str:
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


# ── Route description ─────────────────────────────────────────────────────────

def build_route_desc(primitives, sel_turns):
    parts = []
    turn_idx = 0
    for p in primitives:
        t = p["type"]
        if t == "straight":
            d = p.get("distance_m", 0)
            if d > 0.5: parts.append(f"straight {d:.0f}m")
        elif t in ("left_turn","right_turn"):
            direction = "left" if t=="left_turn" else "right"
            td = sel_turns[turn_idx] if turn_idx < len(sel_turns) else {}
            lm = td.get("landmark","")
            rm = td.get("room","")
            rt = td.get("room_trans","")
            if lm and rt and "doorway" in rt.lower():
                parts.append(f"turn {direction} past [{lm}] into [{rm}]")
            elif lm:
                parts.append(f"turn {direction} at [{lm}]")
            elif rm:
                parts.append(f"turn {direction} → {rm}")
            else:
                parts.append(f"turn {direction}")
            turn_idx += 1
        elif t == "elevation":
            parts.append(f"{'up' if p.get('direction','up')=='up' else 'down'} stairs")
    return " → ".join(parts) or "walk forward"


def gt_start_verb(gt_instr):
    tok = gt_instr.strip().split()
    if not tok: return "Walk"
    w = tok[0].rstrip(".,!?;:")
    return (w[0].upper() + w[1:].lower()) if w else "Walk"


# ── System prompt (same as v17, updated for v18 vision context) ──────────────

SYSTEM_INTRO = """You write concise R2R navigation instructions for an indoor robot.

Style: Natural and brief (22-32 words), like a human R2R annotator.

RULES:
- When a turn has [object] in brackets: use that object as a navigation reference — write it WITHOUT the brackets, naturally: "turn left at the grey pillar", "pass the glass dining table"
- When a turn has (→ room) after the arrow: mention the room transition naturally — "walk into the living room", "head through the hallway" — NOT as a turn anchor
- Turns with no brackets and no arrow: write ONLY "turn left" or "turn right" — no objects, no furniture
- Do NOT invent or add objects not shown in the route
- Endings vary: "Stop near X." / "Wait near X." / no explicit ending
- Start verb is given — use it exactly"""


def build_prompt(ep, gt_instr, examples, lm_info, primitives, sel_turns,
                 stop_phrase, stop_type):
    sv    = gt_start_verb(gt_instr)
    route = build_route_desc(primitives, sel_turns)
    ex_block = "\n".join(f"  {i+1}. \"{ex}\"" for i, ex in enumerate(examples))

    goal_lm    = lm_info.get("goal_landmark","the destination")
    goal_room  = lm_info.get("goal_room","")
    start_ctx  = lm_info.get("start_context","")
    start_room = lm_info.get("start_room","")

    ending_req = (f'End: "{stop_phrase}"' if stop_type in ("stop","wait")
                  else "End naturally — no explicit stop/wait")

    n_bracketed = sum(1 for t in sel_turns if t.get("landmark"))
    n_room_trans = sum(1 for t in sel_turns if t.get("room") and not t.get("landmark"))
    if n_bracketed > 0 and n_room_trans > 0:
        anchor_note = (f"Route has {n_bracketed} bracketed landmark(s) and {n_room_trans} "
                       f"room transition(s) — use both naturally.")
    elif n_bracketed > 0:
        anchor_note = f"Route has {n_bracketed} bracketed landmark(s) — use them naturally."
    elif n_room_trans > 0:
        anchor_note = (f"Route has {n_room_trans} room transition(s) — mention the room(s) "
                       f"naturally. Use 'turn left and walk into the X' NOT 'turn left into the X'.")
    else:
        anchor_note = "Route has NO bracketed landmarks — write ONLY direction words for all turns."

    prompt = (
        f"{SYSTEM_INTRO}\n\n"
        f"Same-building examples:\n{ex_block}\n\n"
        f"Write ONE instruction for:\n"
        f"  Route: {route}\n"
        f"  Start: {start_room or 'room'}"
        + (f" ({start_ctx})" if start_ctx else "")
        + f"\n  Goal: {goal_lm}"
        + (f" in {goal_room}" if goal_room else "")
        + f"\n\n{anchor_note}\n{ending_req}"
        + f"\nStart with: \"{sv}\"\n\n{sv}"
    )
    return prompt, sv


# ── Async generation (Phase 2) ────────────────────────────────────────────────

async def generate_one_p2(client, task, sem, done, total, t0):
    eid = task["episode_id"]
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=VLLM_MODEL,
                    messages=[{"role":"user","content":task["prompt"]}],
                    max_tokens=150,
                    temperature=0.3,
                )
                result = resp.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt == 2: result = f"ERROR: {e}"
                await asyncio.sleep(0.5 * (attempt+1))
    done[0] += 1
    if done[0] % 200 == 0 or done[0] == total:
        elapsed = time.time() - t0
        r = done[0]/elapsed if elapsed>0 else 0.001
        eta = (total-done[0])/r if r>0 else 0
        print(f"  [Phase2 {done[0]}/{total}] {r:.1f}/s  ETA={eta/60:.1f}m", flush=True)
    return eid, result


async def generate_p2(tasks, concurrency=12):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    sem    = asyncio.Semaphore(concurrency)
    done   = [0]; t0 = time.time(); total = len(tasks)
    results = {}
    for eid, result in await asyncio.gather(
        *[generate_one_p2(client, t, sem, done, total, t0) for t in tasks]
    ):
        results[eid] = result
    return results


# ── Post-processing ───────────────────────────────────────────────────────────

PREAMBLES = ["Instruction:","Navigation:","Answer:","Sure,","Certainly,","Of course,",
             "Here is","Here's","Result:","Walk:","The instruction:"]


def clean(raw, sv):
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'): raw = raw[1:-1].strip()
    for pre in PREAMBLES:
        if raw.lower().startswith(pre.lower()):
            raw = raw[len(pre):].lstrip(" :\n").strip()
    sv_l = sv.lower()
    if raw.lower().startswith(sv_l+" "+sv_l): raw = raw[len(sv_l):].lstrip()
    if not raw.lower().startswith(sv_l):
        raw = sv + " " + (raw[0].lower()+raw[1:] if raw else "to the destination.")
    raw = re.sub(r'\[([^\]]+)\]', r'\1', raw)
    raw = re.sub(r'\s*→\s*', ' ', raw)
    raw = re.sub(r'\s+', ' ', raw).strip()
    sents = re.split(r'(?<=[.!?])\s+', raw)
    result = " ".join(sents[:3]).strip()
    if result and result[-1] not in ".!?": result += "."
    return result


def quality_ok(text):
    words = text.split()
    if len(words) < 5:  return False, "too_short"
    if len(words) > 80: return False, "too_long"
    return True, "ok"


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes",        type=int,   default=None)
    ap.add_argument("--concurrency-p1",    type=int,   default=8,
                    help="Concurrency for Phase 1 vision calls (default 8)")
    ap.add_argument("--concurrency-p2",    type=int,   default=12,
                    help="Concurrency for Phase 2 generation calls (default 12)")
    ap.add_argument("--k-similar",         type=int,   default=8)
    ap.add_argument("--p-anchor-episode",  type=float, default=P_ANCHOR_EPISODE)
    ap.add_argument("--skip-phase1",       action="store_true",
                    help="Skip Phase 1 (use existing checkpoint)")
    ap.add_argument("--skip-phase2",       action="store_true",
                    help="Skip Phase 2 generation (assemble from existing checkpoint)")
    args = ap.parse_args()
    p_anchor = args.p_anchor_episode

    print("=== Gate 4 v18: Vision-Grounded Instructions ===")
    print(f"  P_ANCHOR_EPISODE: {p_anchor} → expected ~{p_anchor*0.44*100:.1f}% (GT=16.5%)")
    print(f"  stop/wait: {P_STOP*100:.0f}%/{P_WAIT*100:.0f}%/{(1-P_STOP-P_WAIT)*100:.0f}%")
    print(f"  Phase 1 concurrency: {args.concurrency_p1}")
    print(f"  Phase 2 concurrency: {args.concurrency_p2}")
    print()

    with gzip.open(GT_PATH,"rt") as f: gt_data = json.load(f)
    all_eps  = gt_data["episodes"]
    episodes = all_eps[:args.n_episodes] if args.n_episodes else all_eps
    print(f"Episodes: {len(episodes)}")

    gt_map = {
        ep["episode_id"]: (ep.get("instruction",{}).get("instruction_text","")
                           if isinstance(ep.get("instruction"),dict) else "")
        for ep in all_eps
    }

    sc_idx = ScenePathIndex(all_eps)

    print("Loading text landmarks (fallback)...")
    pf_map  = load_perframe_landmarks(G3PF_DIR)
    old_map = load_old_landmarks(G3_OLD_DIR)
    n_pf = sum(1 for ep in episodes if ep["episode_id"] in pf_map)
    print(f"  Gate 3 perframe: {n_pf}/{len(episodes)} episodes")

    # ── Phase 1: Vision descriptions ─────────────────────────────────────────
    p1_existing = {}
    if P1_CKPT.exists():
        p1_existing = json.load(open(P1_CKPT))
        print(f"Phase 1 checkpoint: {len(p1_existing)} episodes already processed")

    if not args.skip_phase1:
        vision_map = await run_phase1(episodes, p1_existing, args.concurrency_p1)
    else:
        vision_map = p1_existing
        print(f"Phase 1 skipped. Using {len(vision_map)} episodes from checkpoint.")

    # Phase 1 coverage stats
    has_vis = sum(1 for ep in episodes if str(ep["episode_id"]) in vision_map)
    has_goal_vis = sum(1 for ep in episodes
                       if vision_map.get(str(ep["episode_id"]),{}).get("goal"))
    has_turn1_vis = sum(1 for ep in episodes
                        if vision_map.get(str(ep["episode_id"]),{}).get("turn_1"))
    print(f"\nVision coverage:")
    print(f"  Episodes with any vision: {has_vis}/{len(episodes)} ({100*has_vis/len(episodes):.1f}%)")
    print(f"  Episodes with goal vision: {has_goal_vis}/{len(episodes)} ({100*has_goal_vis/len(episodes):.1f}%)")
    print(f"  Episodes with turn_1 vision: {has_turn1_vis}/{len(episodes)} ({100*has_turn1_vis/len(episodes):.1f}%)")
    print()

    # ── Phase 2: Instruction generation ──────────────────────────────────────
    p2_ckpt = {}
    if P2_CKPT.exists():
        p2_ckpt = json.load(open(P2_CKPT))
        print(f"Phase 2 checkpoint: {len(p2_ckpt)} episodes done")

    ep_feats = {ep["episode_id"]: extract_path_features(ep) for ep in episodes}
    tasks     = []
    task_meta = {}
    skipped   = 0

    for ep in episodes:
        eid      = ep["episode_id"]
        gt_instr = gt_map.get(eid,"")
        ep_feat  = ep_feats[eid]
        prims    = get_primitives(ep)

        # v18: use vision descriptions + text fallback
        lm_info  = get_landmark_info_v18(ep, pf_map, old_map, prims, vision_map)

        sel_turns              = episode_selective_turns(lm_info, eid, p_anchor)
        stop_phrase, stop_type = choose_stop(lm_info["goal_landmark"], lm_info["goal_room"], eid)

        if str(eid) in p2_ckpt:
            skipped += 1
            task_meta[eid] = {
                "sv":          gt_start_verb(gt_instr),
                "goal_lm":     lm_info["goal_landmark"],
                "stop_phrase": stop_phrase,
                "stop_type":   stop_type,
            }
            continue

        examples   = sc_idx.top_k(ep, ep_feat, k=args.k_similar)
        prompt, sv = build_prompt(ep, gt_instr, examples, lm_info, prims,
                                   sel_turns, stop_phrase, stop_type)
        task_meta[eid] = {
            "sv":          sv,
            "goal_lm":     lm_info["goal_landmark"],
            "stop_phrase": stop_phrase,
            "stop_type":   stop_type,
        }
        tasks.append({"episode_id": eid, "prompt": prompt, "sv": sv})

    print(f"\n=== Phase 2: Instruction Generation ===")
    print(f"Tasks: {len(tasks)}  Skipped: {skipped}")

    # Pre-generation anchor stats
    n_with_lm = 0
    n_vision_lm = 0
    for ep in episodes:
        eid   = ep["episode_id"]
        prims = get_primitives(ep)
        lm_info = get_landmark_info_v18(ep, pf_map, old_map, prims, vision_map)
        sel = episode_selective_turns(lm_info, eid, p_anchor)
        has_lm = any(t.get("landmark") for t in sel)
        if has_lm:
            n_with_lm += 1
            # Check if any selected turn's landmark came from vision
            vis = vision_map.get(str(eid), {})
            prims_t = get_primitives(ep)
            lm_t = get_landmark_info_v18(ep, pf_map, old_map, prims_t, vision_map)
            if any(vis.get(t.get("label","")) for t in lm_t["turns"] if t.get("landmark")):
                n_vision_lm += 1

    print(f"  Episodes with turn anchor: {n_with_lm}/{len(episodes)} = {100*n_with_lm/len(episodes):.1f}% (GT=16.5%)")
    print(f"  Of those, vision-grounded: {n_vision_lm} ({100*n_vision_lm/max(n_with_lm,1):.0f}%)")

    from collections import Counter
    stop_types = [task_meta[ep["episode_id"]]["stop_type"]
                  for ep in episodes if ep["episode_id"] in task_meta]
    sc = Counter(stop_types)
    n  = len(stop_types)
    print(f"  Stop/Wait/None: {sc.get('stop',0)/n*100:.1f}%/{sc.get('wait',0)/n*100:.1f}%/{sc.get('none',0)/n*100:.1f}%")

    if tasks and not args.skip_phase2:
        sample = tasks[0]
        print(f"\n--- Sample prompt (ep {sample['episode_id']}) ---")
        print(sample["prompt"][:1400])
        print("-"*60)
        print()
        print("Generating...")
        results = await generate_p2(tasks, args.concurrency_p2)
        p2_ckpt.update({str(k):v for k,v in results.items()})
        P2_CKPT.parent.mkdir(parents=True, exist_ok=True)
        with open(P2_CKPT,"w") as f: json.dump(p2_ckpt,f)

    # ── Assemble ──────────────────────────────────────────────────────────────
    print("\nAssembling dataset...")
    tokenizer    = VLNTokenizer(GT_PATH)
    episodes_out = []
    n_pass = n_fix = n_fail = 0

    for ep in episodes:
        eid  = ep["episode_id"]
        raw  = p2_ckpt.get(str(eid),"")
        if not raw or raw.startswith("ERROR"):
            n_fail += 1
            continue

        meta        = task_meta.get(eid,{})
        sv          = meta.get("sv","Walk")
        stop_phrase = meta.get("stop_phrase","")
        stop_type   = meta.get("stop_type","none")
        goal_lm     = meta.get("goal_lm","the destination")

        text = clean(raw, sv)
        ok, _ = quality_ok(text)
        if not ok:
            text = f"{sv} to the destination."
            if stop_phrase: text = text.rstrip('.') + '. ' + stop_phrase
            n_fix += 1

        text = enforce_stop_phrase(text, stop_phrase, stop_type)
        episodes_out.append(assemble_episode(ep, text, tokenizer))
        n_pass += 1

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out_data = {"episodes": episodes_out,
                "instruction_vocab": gt_data.get("instruction_vocab",{})}
    with gzip.open(OUTPUT,"wt",encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False)

    # ── Stats ─────────────────────────────────────────────────────────────────
    texts     = [ep["instruction"]["instruction_text"] for ep in episodes_out]
    n         = len(texts)
    avg_w     = sum(len(t.split()) for t in texts)/n if n else 0
    stop_n    = sum(1 for t in texts if STOP_RE.search(t) and not WAIT_RE.search(t))
    wait_n    = sum(1 for t in texts if WAIT_RE.search(t))
    neither_n = sum(1 for t in texts if not STOP_RE.search(t) and not WAIT_RE.search(t))
    anchor_n  = sum(1 for t in texts if has_turn_anchor(t))
    dup_stop  = sum(1 for t in texts if len(re.findall(r'\b(stop|wait|halt)\b', t, re.I)) >= 2)

    spp = {k:0 for k in ["past","through","into","at the"]}
    for t in texts:
        tl = t.lower()
        for k in spp: spp[k] += k in tl

    print(f"\n=== v18 Done ===")
    print(f"  Episodes:     {n}  Pass: {n_pass}  Fixed: {n_fix}  Failed: {n_fail}")
    print(f"  avg_words:    {avg_w:.1f}  (GT=26.8)")
    print(f"  stop%:        {100*stop_n/n:.1f}%  (GT=50.8%)")
    print(f"  wait%:        {100*wait_n/n:.1f}%  (GT=30.5%)")
    print(f"  neither%:     {100*neither_n/n:.1f}%  (GT=16.5%)")
    print(f"  turn-anchor%: {100*anchor_n/n:.1f}%  (GT=16.5%) ← MUST STAY NEAR 16.5%")
    print(f"  dup-stop:     {dup_stop} (target: 0)")
    print(f"  spatial:")
    for k,v in spp.items():
        print(f"    {k:<12}: {100*v/n:.1f}%")
    print(f"\nSaved: {OUTPUT}")

    anchor_err = abs(anchor_n/n - 0.165)
    stop_err   = abs(stop_n/n  - 0.508)
    wait_err   = abs(wait_n/n  - 0.305)
    match_score = 1.0 - (anchor_err*2 + stop_err + wait_err)
    print(f"\n  GT-match score: {match_score:.3f}  (target ≥ 0.90; v17=0.948)")
    print(f"  Word count: {avg_w:.1f}w  (GT=26.8w)")

    # Vision contribution stats
    vis_goal = sum(1 for ep in episodes_out
                   if any(w in ep["instruction"]["instruction_text"].lower()
                          for w in ["grey", "white", "dark", "wooden", "glass", "silver"]))
    print(f"\n  ~Vision-grounded stop phrases: check 'grey/white/dark/wooden/glass/silver' presence: {vis_goal}/{n} ({100*vis_goal/n:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
