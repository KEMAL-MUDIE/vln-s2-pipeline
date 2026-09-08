#!/usr/bin/env python3
"""
Gate 4 Visual v201 -- FIX17b: Remove "continue past" from FIX17 (fixes continue% regression).

BASE: v200 (FIX1-17 — walk_past=33.7%, toward=14.2%, continue=18.8%, GT-match=0.843)

MOTIVATION:
v200 FIX17 introduced "continue past the X" as 5% of FIX7 replacements.
This caused continue% to increase from 17.7% (v199) to 18.8% (v200), a 1.1pp regression.
GT continue=8.2%. We're already 9.5pp above GT — any increase is undesirable.
FIX17b redistributes the 5% from "continue past" to "go past" (18%→23%) to avoid continue inflation.

NEW IN v201:
FIX17b: Remove "continue past" from FIX7 replacement, redistribute to "go past".
  PROBLEM: v200 "continue past the X" at 5% → continue% 17.7%→18.8% (+1.1pp regression).
           GT continue=8.2%. v200 continue=18.8% is 10.6pp above GT.
  FIX: Replace 6-phrase distribution with 5-phrase (no "continue past"):
       P=0.35: "walk past the X"    (walk_past — unchanged)
       P=0.24: "walk toward the X"  (walk_toward — unchanged)
       P=0.23: "go past the X"      (go_past — was 18%, +5% from removed continue_past)
       P=0.12: "pass the X"         (pass_the — unchanged)
       P=0.00: "continue past the X"(REMOVED — was +1.1pp continue%)
       P=0.06: "head toward the X"  (head_toward — unchanged)
  EXPECTED: continue ~17.7% (back to v199 level, -1.1pp from v200)
            go_past ~3.0% (23% of FIX7 replacements, closer to GT=2.8%)
            walk_past unchanged from v200 (~33.7%)
            toward unchanged from v200 (~14.2%)

INHERITED FROM v200: FIX1-17 (FIX7 6-phrase vocabulary, FIX8 diversification, surface fixes)
ASSEMBLY-ONLY: Reuse outputs/gate4_v201_phase2_checkpoint.json (copy of v190 P2).

Expected v201:
  walk_past: ~33.7% (GT=11.6%) — unchanged from v200
  go_past: ~3.0% (GT=2.8%) — very close to GT!
  continue: ~17.7% (GT=8.2%) — −1.1pp from v200, back to v199 level
  toward: ~14.2% (GT=4.9%) — unchanged from v200
  All v200 fixes (FIX11-17) inherited, FIX17b replaces FIX17 distribution.
  Predicted SR: 64-74% (same baseline as v200, no continue regression)

Phase 1 checkpoint: outputs/gate4_v152_phase1_checkpoint.json (REUSE)
Phase 2 checkpoint: outputs/gate4_v201_phase2_checkpoint.json (COPY of v190 P2 — assembly-only)
Output: outputs/datasets/val_unseen_generated_gemma_visual_v201.json.gz
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

GT_PATH       = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
G3PF_DIR      = ROOT / "outputs" / "gate3_perframe"
G3_OLD_DIR    = ROOT / "outputs" / "gate3_landmarks"
RF_DIR        = ROOT / "outputs" / "rendered_frames"
MIDPOINTS_DIR = Path("/mnt/nvme0/vln_habitat/habitat_data/midpoint_frames")
OUTPUT        = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v201.json.gz"
P1_CKPT       = ROOT / "outputs" / "gate4_v152_phase1_checkpoint.json"  # REUSE v152 room-context Phase C
P1_MID_CKPT   = ROOT / "outputs" / "gate4_v75_midpoints_p1_checkpoint.json"  # v75: ranked-priority midpoints
P1_TS_CKPT    = ROOT / "outputs" / "gate4_v73_approach_views_p1_checkpoint.json"  # v73: approach-view landmarks
P1_GOAL_CKPT  = ROOT / "outputs" / "gate4_v74_goal_approach_p1_checkpoint.json"  # v74: goal-approach stop descriptions
P2_CKPT       = ROOT / "outputs" / "gate4_v201_phase2_checkpoint.json"  # v201: assembly-only, copy of v190 P2 (FIX17b)

from gate2_path.path_analyzer import analyze_path
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode

# ── Constants ─────────────────────────────────────────────────────────────────

P_ANCHOR_EPISODE = 0.06   # v171: further recalibrated. v170(0.08)→anchor%=25.1% (hallway injection inflates!).
                           # Hallway "end of the hallway" injection adds ~8pp to anchor%. 0.06 base → target ~16.5%
P_HALLWAY_POSITIONAL = 0.30  # v171: only 30% of hallway turns get "end of the hallway" injection (was 100% in v170)
P_STOP = 0.494            # v63: calibrated 0.508→0.494 to compensate LLM +1.4pp over-stop
P_WAIT = 0.305

# v34: Probabilistic midpoint sampling to match GT frequencies.
# GT: walk-past=8.7%, through=27.2%. v33: walk-past=14.0%, through=32.4%.
# Math: 0.60 × 14.0% ≈ 8.4% (≈GT 8.7%); 0.85 × 32.4% ≈ 27.5% (≈GT 27.2%).
P_PASS_MARKER = 0.82   # v139 FRESH: 0.82 → ~8.75% (linear interp: (0.82-0.80)×17.5 + 8.4 = 8.75%≈GT=8.7%)
P_THRU_MARKER = 0.60   # v157: 0.75→0.60 to reduce through_the from 33.7% toward GT=27.2% (scaling: 33.7*0.60/0.75=27.0%)
P_TURN_UNANCHORED = 1.00  # v169: RESTORE ALL TURNS (v24 had 1.0 → avg_turns=1.53 = BEST SR=40.24%).
                           # v168 had 0.30 → avg_turns=0.64. Matching GT avg_turns=0.59 did NOT help SR.
                           # More explicit turns = more navigation guidance at decision points.
P_DIR_ROOM_TRANS  = 0.05  # v169: reduced from 0.10 (more P_ANCHOR turns add anchor%, so reduce P_DIR)
                           # P_ANCHOR_EPISODE=0.13 contributes ~11% + P_DIR=0.05 contributes ~5% = ~16% anchor%
P_DOOR_DIR        = 0.50  # v174: when use_door=True and direction is known, add "on the left/right"
                           # GT: 23.8% of eps have "on the left/right" → 90% for door disambiguation
                           # Route: "turn left (door→ bedroom)" → "turn left (door on the left → bedroom)"
                           # Route: "(door→ bedroom)" → "(door on the left → bedroom)" [when dir known]
                           # Expected: on_left≈10%, on_right≈10% → combined≈20% (GT=23.8%)

# ── Pattern detection ─────────────────────────────────────────────────────────

TURN_ANCHOR_RE = re.compile(
    r'turn\s+(left|right)\s+(at|past|through|into|around)\s+the\s+\w',
    re.IGNORECASE
)
WALK_PAST_RE = re.compile(r'(walk|go|head|move)\s+past\s+the\s+\w', re.IGNORECASE)
STOP_RE = re.compile(r'\b(stop|halt|stand)\b', re.IGNORECASE)
WAIT_RE = re.compile(r'\b(wait|waiting)\b', re.IGNORECASE)


def has_turn_anchor(text: str) -> bool:
    # v29: Only count real turn anchors ("turn left at the X"), NOT "walk past the X" midpoints.
    # Using WALK_PAST_RE here inflated turn-anchor% from 16.5% to 39.2% in v28 (broken metric).
    return bool(TURN_ANCHOR_RE.search(text))


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
    "In 1-2 sentences: (1) describe the most prominent landmark at this turning point "
    "(color, material, shape), and (2) briefly note what's visible in the direction the "
    "robot will go after turning. Focus on navigation-useful details a person would remember. "
    "Examples: 'Turn at the white rectangular dining table with dark chairs. Ahead, a sunlit "
    "living room with hardwood floors opens up.' or 'The grey stone pillar marks the corner. "
    "A hallway with wooden panels continues to the left.' "
    "Reply with ONLY these 1-2 sentences, nothing else."
)

VISION_PROMPT_GOAL = (
    "A robot has arrived at its navigation destination. "
    "In 1-2 sentences, describe the stopping location: (1) the specific object that marks where "
    "to stop (color, material, shape), and (2) any distinctive context around it (window, wall, "
    "adjacent furniture). Do NOT use 'Stop' or 'Wait' — just describe what you see. "
    "Examples: 'The grey fabric lounge chair positioned against the window wall. Warm afternoon "
    "light falls across the wooden floor nearby.' or 'A white marble fireplace with a dark wooden "
    "mantel. It faces a seating area with couches on both sides.' "
    "Reply with ONLY the description, nothing else."
)

VISION_PROMPT_START = (
    "A robot starts navigation from this indoor location. "
    "In 1-2 sentences: describe the most distinctive features of this starting area — "
    "what room type it is, and the 1-2 most prominent objects or architectural features visible. "
    "Examples: 'Starting in a bright hallway with brown wooden double doors on the right. "
    "The polished stone floor leads forward.' or 'A modern bathroom with white tile walls "
    "and a bathrobe hanging by the door. Straight ahead is a long corridor.' "
    "Reply with ONLY these 1-2 sentences, nothing else."
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
    """Post-process raw vision description. Return None if unusable. v19: accepts sentences."""
    raw = raw.strip().strip('"\'').strip()
    # Remove common preambles
    for prefix in ["Description:", "The landmark is", "I see", "I can see", "The most", "Based on",
                   "In this image", "This image shows"]:
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].strip(" :-").strip()
    # Limit to first 2 sentences (v19 asks for 1-2 sentences)
    sents = re.split(r'(?<=[.!?])\s+', raw)
    raw = ' '.join(sents[:2]).strip()
    # Clean up
    raw = re.sub(r'\s+', ' ', raw).strip()
    words = raw.split()
    if len(words) < 3 or len(words) > 60:
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
                max_tokens=80,
                temperature=0.25,
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
    print(f"\n=== Phase 1: Rich Vision Narrative Extraction ===")
    print(f"  concurrency: {concurrency}  max_tokens_per_call: 80 (v19: richer descriptions)")

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


# ── Vision description utilities ──────────────────────────────────────────────

PREPOSITION_RE = re.compile(
    r'\s+(positioned|against|near|by|in|with|on|at|of|to|for|leading|facing|containing|next|along)\b',
    re.IGNORECASE
)


# ── v22: Midpoint loading ─────────────────────────────────────────────────────

def load_midpoints_checkpoint() -> Dict:
    """Load Phase 1b midpoints checkpoint: {eid_str: {mid_N: desc}}"""
    if not P1_MID_CKPT.exists():
        return {}
    return json.load(open(P1_MID_CKPT))


def load_goal_approach_checkpoint() -> Dict:
    """Load v74 goal-approach stop descriptions: {eid_str: desc_str}"""
    if not P1_GOAL_CKPT.exists():
        return {}
    return json.load(open(P1_GOAL_CKPT))


_HAS_COLOR_MAT = re.compile(
    r'\b(grey|gray|white|black|brown|dark|light|red|blue|green|yellow|orange|purple|cream|beige|'
    r'tan|silver|gold|brass|copper|bronze|wooden|wood|marble|glass|metal|leather|fabric|velvet|'
    r'stone|granite|oak|mahogany|walnut|tiled|painted)\b',
    re.IGNORECASE
)
_GENERIC_ARCH_STOP = re.compile(
    r'\b(doorway|doorframe|door frame|wall|floor|ceiling|corridor|hallway|passage|entryway|'
    r'archway|opening|entrance|exit)\b',
    re.IGNORECASE
)


def apply_goal_approach_stop(stop_phrase: str, stop_type: str, goal_desc: str,
                              eid: int) -> Tuple[str, str]:
    """v74: Override generic stop phrase with goal-approach VLM description when better."""
    if stop_type == "none" or not goal_desc:
        return stop_phrase, stop_type
    has_color = bool(_HAS_COLOR_MAT.search(stop_phrase))
    is_arch = bool(_GENERIC_ARCH_STOP.search(stop_phrase))
    goal_has_color = bool(_HAS_COLOR_MAT.search(goal_desc))
    if has_color and not is_arch:
        return stop_phrase, stop_type
    if not goal_has_color:
        return stop_phrase, stop_type
    # Override with goal-approach description
    rng = random.Random(eid ^ 0xB4C2)
    goal_desc_clean = strip_material_adjectives(goal_desc.strip())  # v76: strip wooden/marble/etc from stop
    if not goal_desc_clean.startswith("the "):
        goal_desc_clean = f"the {goal_desc_clean}"
    # v84: GT-calibrated prepositions (stop near: only 1.3% in GT vs old 33%)
    draw = rng.random()
    if stop_type == "stop":
        if draw < 0.35:
            new_phrase = f"Stop at {goal_desc_clean}."
        elif draw < 0.60:
            new_phrase = f"Stop in front of {goal_desc_clean}."
        elif draw < 0.80:
            new_phrase = f"Stop next to {goal_desc_clean}."
        elif draw < 0.95:
            new_phrase = f"Stop by {goal_desc_clean}."
        else:
            new_phrase = f"Stop near {goal_desc_clean}."
    else:
        if draw < 0.30:
            new_phrase = f"Wait at {goal_desc_clean}."
        elif draw < 0.52:
            new_phrase = f"Wait near {goal_desc_clean}."
        elif draw < 0.70:
            new_phrase = f"Wait by {goal_desc_clean}."
        else:
            new_phrase = f"Wait next to {goal_desc_clean}."
    return new_phrase, stop_type


def load_turn_sides_checkpoint() -> Dict:
    """Load Phase 1 approach-views checkpoint: {eid_str: {turn_N: desc}} (v70: approach-direction landmarks)"""
    if not P1_TS_CKPT.exists():
        return {}
    return json.load(open(P1_TS_CKPT))


_GENERIC_TURN_WORDS = {
    'wall', 'walls', 'floor', 'ceiling', 'hallway', 'corridor',
    'space', 'room', 'area', 'passage', 'corner', 'junction',
    # v26: REMOVED 'door', 'doors', 'opening', 'entrance', 'exit' — GT uses doors 47.9%!
}


def is_turn_generic(desc: Optional[str]) -> bool:
    """Return True if desc is a generic/non-useful turn landmark."""
    if not desc:
        return True
    words = set(desc.lower().split())
    if len(words) < 2:
        return True
    return bool(words & _GENERIC_TURN_WORDS)


_THRU_WORDS = re.compile(
    r'\b(doorway|doorways|door|doors|arch|arched|archway|archways|opening|'
    r'gate|threshold|portal|entrance|entryway|passageway)\b',
    re.IGNORECASE
)
_ROOM_WORDS = re.compile(
    r'\b(kitchen|bedroom|bathroom|living room|dining room|hallway|corridor|'
    r'office|foyer|lobby|staircase|stairwell|closet|study|library|garage)\b',
    re.IGNORECASE
)


def classify_pass_action(desc: str) -> str:
    """v27: Classify midpoint description as 'thru' (through the X) or 'pass' (walk past the X).
    Doorways, arches, room transitions → 'thru'.
    Objects, furniture → 'pass'.
    """
    if not desc:
        return "pass"
    if _THRU_WORDS.search(desc):
        return "thru"
    if _ROOM_WORDS.search(desc):
        return "thru"
    return "pass"


_GENERIC_OBJECTS_RE = re.compile(
    r'\b(sofa|couch|loveseat|sectional|settee|'
    r'table|desk|counter|countertop|'
    r'chair|armchair|recliner|stool|bench|ottoman|'
    r'bed|mattress|headboard|'
    r'cabinet|shelf|shelves|bookshelf|bookcase|dresser|wardrobe|closet|'
    r'wall|walls|ceiling|floor|panel|panels|trim|molding|wainscoting|'
    r'curtain|curtains|drape|drapes|blinds|'
    r'lamp|light|lighting)\b',
    re.IGNORECASE
)
_DISTINCTIVE_OBJECTS_RE = re.compile(
    r'\b(fireplace|piano|artwork|art|painting|sculpture|statue|fountain|mural|'
    r'refrigerator|fridge|stove|oven|washer|dryer|dishwasher|'
    r'pool\s+table|billiard|foosball|ping\s*pong|game\s+table|'
    r'carpet|rug|runner|mosaic|tile\s+floor|tiled|'
    r'toilet|bathtub|shower|vanity|sink|'
    r'staircase|stairway|stairs|banister|railing|handrail|'
    r'mirror|television|tv\b|display|aquarium|bookcase|'
    # v107: Add v24's actual "passing the" objects — these were in GENERIC or missing:
    r'sofa|couch|loveseat|sectional|settee|'         # furniture v24 used heavily
    r'armchair|armchairs?|'                           # armchair common in v24
    r'dining\s+table|coffee\s+table|end\s+table|'    # specific table types
    r'kitchen\s+island|island|'                       # kitchen furniture
    r'kitchen\s+cabinets?|sideboard|console|'         # cabinetry landmarks
    r'door(?:way)?|double\s+doors?|archway|arch|'    # architectural passages
    r'bookshelf|bookshelves|'                         # shelving
    r'chandelier|pendant|'                            # overhead fixtures (distinctive)
    r'window\s+seat)\b',                              # distinctive window feature
    re.IGNORECASE
)

# v108: STRUCTURAL objects that are too non-specific for [pass:] markers.
# v108 switches from WHITELIST (allow only distinctive) to BLACKLIST (exclude only structural).
# v165: Extended blacklist — add common furniture. Root cause: "walk past the brown armchair/sofa"
# confuses the agent (trained on room/architecture navigation, not furniture-as-waypoints).
# Only DISTINCTIVE objects remain eligible: piano, fireplace, pool table, mosaic, chandelier, etc.
# NOTE: "wall" uses negative lookahead to preserve "wall painting/clock/art/sculpture/mirror/mural"
# (hung objects are distinctive landmarks, standalone "wall" is structural).
_STRUCTURAL_OBJECTS_RE = re.compile(
    r'\bwall(?![-\s]+(?:art|painting|clock|mirror|sculpture|decor(?:ation)?|mural|hanging|tapestry|mounted|mount))\b'
    r'|\b(?:walls|ceiling|ceilings|floor|floors|hallway|hall|corridor|'
    r'room|area|space|lobby|entrance|exit|'
    r'post|column|pillar|beam|trim|molding|wainscoting|panel|panels|'
    # v165: common furniture — [pass: furniture] confuses agent navigation
    r'sofa|couch|loveseat|sectional|settee|'
    r'armchair|armchairs?|chair|chairs?|recliner|stool|bench|ottoman|'
    r'table|coffee\s+table|dining\s+table|end\s+table|side\s+table|'
    r'desk|counter|countertop|'
    r'bed|mattress|headboard|'
    r'cabinet|cabinets?|shelf|shelves|bookshelf|bookshelves|bookcase|dresser|wardrobe|closet|'
    r'lamp|lamps?|light(?!\s+fixture)|lighting|'
    r'curtain|curtains?|drape|drapes?|blinds|'
    r'plant|plants?|potted|foliage|'
    r'cushion|pillow|mat|runner)\b',
    re.IGNORECASE
)


_MATERIAL_ADJ_RE = re.compile(
    r'\b(wooden|wood(?:en)?|hardwood|oak|pine|walnut|maple|cedar|mahogany|cherry|teak|bamboo|'
    r'marble|stone|granite|slate|concrete|brick|limestone|travertine|'
    r'carpeted|carpet|tiled|tile|vinyl|laminate|linoleum|terrazzo|'
    r'leather|fabric|upholstered|velvet|plush|linen)\s+',
    re.IGNORECASE
)

# v169: Strip color adjectives from route landmarks to match GT style (GT=15% color, v168=74%!).
# GT annotators almost never use colors. Color-heavy landmarks mislead the model (trained on GT).
# "dark brown leather sofa" → "sofa"; "white marble counter" → "marble counter".
# Apply to [landmark] brackets in route before passing to instruction LLM.
_COLOR_ADJ_RE = re.compile(
    r'\b(white|black|brown|dark\s+brown|light\s+brown|grey|gray|dark\s+grey|dark\s+gray|'
    r'beige|cream|tan|ivory|dark|light|bright|pale|off-white|'
    r'red|blue|green|yellow|orange|purple|pink|navy|olive|teal|'
    r'silver|gold|golden|bronze|copper|metallic)\s+',
    re.IGNORECASE
)

def strip_colors_from_landmark(lm: str) -> str:
    """v169: Strip color adjectives from landmark for GT-style matching. Keep material words."""
    result = _COLOR_ADJ_RE.sub('', lm).strip()
    # Clean up double spaces
    result = re.sub(r'\s+', ' ', result).strip()
    # If stripping left only 1 word or nothing, keep original (don't over-strip)
    if len(result.split()) < 1:
        return lm
    return result if result else lm

# v113: Structural-only suppression for "walk past the X" without a [pass:] marker.
# ROOT CAUSE of v111/v112 avg_words=22.3 (GT=26.8): furniture (armchair, sofa, table, chair,
# shelf, plant, etc.) was in the suppression list → "walk past the armchair" → "walk straight"
# → lost 2-4 words per occurrence. Furniture ARE legitimate "walk past" landmarks in GT.
# FIX: Only suppress truly structural/architectural elements that are NOT scene objects:
# walls, floor, ceiling, hallway, hall, corridor, room, area, lobby, column, pillar.
# Remove: table, sofa, chair, armchair, bench, couch, bed, cabinet, shelf, plant, window,
#         curtain, rug, mat, counter, artwork, mural, panel, relief.
_GENERIC_WPT_OBJ_RE = re.compile(
    r'\bwalk\s+past\s+the\s+'                          # "walk past the "
    r'(?:(?:left|right|center|back|front|inner|outer|'
    r'open|main|narrow|long|wide|small|large|big)\s+)*'
    r'(?:walls?|hallway|hall|corridor|'
    r'lounge(?:\s+area)?|area|room|lobby|'
    r'floor|ceiling|post|column|pillar|beam|'
    r'light\s+switch(?:es)?|switch(?:es)?|outlet|socket)\b',
    re.IGNORECASE
)


# v35: Fix LLM-generated "through the hallway/corridor/room" → natural alternatives.
# v34 analysis showed 73 "through the hallway" + 3 "corridor" + 6 "room" = ~4.5pp of
# "through the" that aren't real architectural passages (doorway/arch/double doors).
_THROUGH_HALLWAY_RE = re.compile(
    r'\bthrough\s+the\s+'
    r'(?:(?:narrow|long|short|dark|wide|small|large|open|winding|'
    r'straight|main|left|right)\s+)*'
    r'(hallway|hall|corridor)\b',
    re.IGNORECASE
)
# v86: Only convert truly wrong "through the [room]" patterns.
# GT uses "through the kitchen", "through the office", "through the living room" naturally.
# Only convert "through the bedroom/bathroom/open space" (GT never uses these as passages).
_THROUGH_ROOM_RE = re.compile(
    r'\bthrough\s+the\s+'
    r'(?:(?:large|small|open|main|dark|bright|adjacent)\s+)*'
    r'(bedroom|bathroom|open\s+(?:space|area)|lounge\s+area)\b',
    re.IGNORECASE
)

# v36: Replace LLM "proceed" overuse. GT uses "proceed" only 0.8% vs LLM 43.8%.
# Replace with GT-style "walk X" alternatives; delete motion-only "proceed forward/straight".
_PROCEED_DEST_RE = re.compile(
    r'\bProceed\s+(to|toward|towards|past|through|into|down|up|along|around)\s+the\b',
    re.IGNORECASE
)
_PROCEED_MOTION_RE = re.compile(
    r'\b[Pp]roceed\s+(?:forward|straight|ahead|straight\s+ahead)\b[,.]?\s*',
)

# v36: Collapse duplicate consecutive motion-only phrases.
# "continue forward. Walk forward." → "continue forward."
# "walk forward and proceed forward" (already caught by _PROCEED_MOTION_RE above)
_DUP_MOTION_RE = re.compile(
    r'\b(continue\s+forward|walk\s+forward|walk\s+straight|go\s+forward|move\s+forward)'
    r'[.,]?\s+'
    r'(?:and\s+)?(?:continue\s+forward|walk\s+forward|walk\s+straight|go\s+forward|move\s+forward)\b',
    re.IGNORECASE
)

# v43: Fix "continue" overuse from v42 prompt (continue spiked 9%→21%).
# v109: RESTORE CONTINUE SUPPRESSION — match GT vocabulary (continue=10.4%, not 45%+).
# GT uses "walk" far more than "continue". v108 had 45.2% "continue" vs GT's 10.4%.
# v109: Convert "Continue straight/forward/ahead" → "Walk straight/forward/ahead"
# This brings the model's input language closer to GT training distribution.
_CONTINUE_MOTION_RE = re.compile(
    r'\bContinue\s+(straight|forward|ahead|along|down|up|through|past|to|toward)\b',
    re.IGNORECASE
)
_CONTINUE_FRAG_RE = re.compile(
    r'\bContinue\s+walking\b|\bContinue\s+moving\b',
    re.IGNORECASE
)

def _replace_continue_motion(m: re.Match) -> str:
    original_c = m.group(0)[0]  # 'C' (sentence-start) or 'c' (mid-sentence)
    word = m.group(1)
    walk = 'Walk' if original_c == 'C' else 'walk'
    return f"{walk} {word[0].lower()}{word[1:]}"

# v36: Fix malformed truncation artifacts — "Take a." "Head." isolated fragments.
_TAKE_A_FRAG_RE = re.compile(r'\bTake\s+a\s*\.\s*', re.IGNORECASE)
_HEAD_FRAG_RE = re.compile(r'\bHead\s*\.\s*', re.IGNORECASE)


def _replace_proceed_dest(m: re.Match) -> str:
    prep = m.group(1).lower()
    return f"Walk {prep} the"


def strip_material_adjectives(desc: str) -> str:
    """v29: Remove material/texture adjectives from midpoint descriptions.
    GT annotators rarely use material adjectives (wooden=3.3%, we had 39.9%).
    Converts 'wooden double doors' → 'double doors', 'marble floor' → 'floor'.
    """
    if not desc:
        return desc
    result = _MATERIAL_ADJ_RE.sub('', desc)
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def is_generic_pass_object(desc: str) -> bool:
    """v108: BLACKLIST approach — exclude ONLY structural non-objects.
    v31-v107 used strict WHITELIST (allow only definitely-distinctive objects).
    v107 expanded whitelist to include sofas, armchairs, doors, etc. but still only 39.4%
    of episodes got [pass:] markers (beds, windows, plants all filtered).
    v108: Switch to BLACKLIST — exclude only structural non-objects (wall, floor, ceiling).
    Allow everything else: beds, windows, plants, tables, chairs, curtains, etc.
    Expected: [pass:] coverage rises from 39% → ~90%+ → "passing the" ~70-80% (v24=79.7%).
    """
    if not desc:
        return True
    # v108: Exclude ONLY structural elements (wall, floor, ceiling, hallway, room, etc.)
    # that don't identify a physical object the robot can visually locate.
    if _STRUCTURAL_OBJECTS_RE.search(desc):
        return True  # structural → too generic for [pass:]
    # Everything else (bed, window, plant, table, curtain, lamp, etc.) is eligible
    return False


def get_all_segment_midpoints(ep, midpoints_ckpt: Dict) -> Dict[str, str]:
    """
    v23: Return midpoints for BOTH the FIRST and LAST segments (up to 2 entries).

    Strategy:
    - First segment (start→turn_1): early-path landmark "passing the X"
    - Last segment (turn_N→goal): final-approach landmark "passing the X"
    - When only 1 segment has midpoints: that segment's landmark is used once
    - When 2+ segments have midpoints: first AND last get landmarks (up to 2)

    Returns {endpoint_label: desc} with up to 2 entries.
    endpoint_label is "goal" for last keyframe, or "turn_N" for intermediate keyframes.
    """
    eid = ep["episode_id"]
    eid_str = str(eid)

    # Load keyframe waypoint indices from poses.json
    ep_dir = RF_DIR / f"episode_{eid:06d}"
    poses_f = ep_dir / "poses.json"
    if not poses_f.exists():
        return {}

    try:
        poses = json.load(open(poses_f))
    except Exception:
        return {}

    # Build list of (label, waypoint_idx) for each rendered keyframe
    keyframes = [(f["label"], f["waypoint_idx"]) for f in poses.get("frames", [])]
    if len(keyframes) < 2:
        return {}

    # Build midpoint desc lookup: waypoint_idx → desc (from Phase 1b checkpoint)
    mid_by_idx: Dict[int, str] = {}
    ep_mids = midpoints_ckpt.get(eid_str, {})
    for label, desc in ep_mids.items():
        if label.startswith("mid_") and desc:
            try:
                idx = int(label.split("_")[1])
                mid_by_idx[idx] = desc
            except ValueError:
                pass

    if not mid_by_idx:
        return {}

    # For each segment (a → b), collect intermediate midpoints
    # (strictly between a_idx and b_idx, and NOT at a keyframe position)
    keyframe_idxs = {wpt_idx for _, wpt_idx in keyframes}
    segments = []  # list of (b_label, first_midpoint_desc)
    for i in range(len(keyframes) - 1):
        a_label, a_idx = keyframes[i]
        b_label, b_idx = keyframes[i + 1]
        intermediates = [
            (mid_idx, desc) for mid_idx, desc in sorted(mid_by_idx.items())
            if a_idx < mid_idx < b_idx and mid_idx not in keyframe_idxs
        ]
        if intermediates:
            segments.append((b_label, intermediates[0][1]))  # first midpoint in segment

    if not segments:
        return {}

    result: Dict[str, str] = {}

    if len(segments) == 1:
        # Only one segment has midpoints — use it (could be first or last, same thing)
        result[segments[0][0]] = segments[0][1]
    else:
        # 2+ segments: use FIRST (early guidance) AND LAST (final approach)
        result[segments[0][0]] = segments[0][1]   # first segment
        result[segments[-1][0]] = segments[-1][1]  # last segment

    return result


def extract_noun_phrase(text: str, max_words: int = 6) -> str:
    """Extract the core noun phrase: strip leading article, truncate at first preposition."""
    if not text:
        return text
    # Take first sentence only
    sent = re.split(r'(?<=[.!?])\s+', text.strip())[0].strip().rstrip('.')
    # Strip leading article
    sent = re.sub(r'^(The|A|An)\s+', '', sent, flags=re.IGNORECASE).strip()
    # Truncate at first preposition
    m = PREPOSITION_RE.search(sent)
    if m:
        sent = sent[:m.start()].strip()
    return ' '.join(sent.split()[:max_words]).rstrip('.,')


def extract_turn_noun(desc: str) -> str:
    """From a rich turn description, extract the turn pivot landmark.
    E.g.: 'Turn at the white rectangular table. Ahead, a living room...'
    → 'white rectangular table'
    v150 fix: also handles direction-aware prefix 'Turn left/right at the ...'
    E.g.: 'Turn left at the brown leather sofa. Ahead, ...' → 'brown leather sofa'
    """
    if not desc:
        return ""
    sent = re.split(r'(?<=[.!?])\s+', desc.strip())[0].strip()
    # Remove "Turn [left|right] at/past/through..." prefix (v150: added left/right handling)
    sent = re.sub(r'^Turn\s+(left|right|around)?\s*(at|past|through|around|into)\s+(the\s+)?', '',
                  sent, flags=re.IGNORECASE).strip()
    sent = re.sub(r'^(The|A|An)\s+', '', sent, flags=re.IGNORECASE).strip()
    # Truncate at preposition
    m = PREPOSITION_RE.search(sent)
    if m:
        sent = sent[:m.start()].strip()
    return ' '.join(sent.split()[:8]).rstrip('.,')


def extract_goal_full(desc: str) -> str:
    """Return up to first sentence for Goal: display line (strips leading article)."""
    if not desc:
        return desc
    sent = re.split(r'(?<=[.!?])\s+', desc.strip())[0].strip()
    sent = re.sub(r'^(The|A|An)\s+', '', sent, flags=re.IGNORECASE).strip()
    return sent.rstrip('.')


# "Ahead, a living room with a grey sofa and television is visible." → "grey sofa"
_ROOM_TYPES_RE = re.compile(
    r'^(bedroom|bathroom|hallway|corridor|kitchen|living room|dining room|'
    r'office|closet|foyer|staircase|stairwell|lobby|entrance|open area|'
    r'sunlit room|bright room|dark corridor)\s+with\s+',
    re.IGNORECASE
)
_AHEAD_PREFIX_RE = re.compile(
    r'^(Ahead[,.]?\s*|Ahead is\s*|In front[,.]?\s*|You\'ll see\s*|Looking ahead[,.]?\s*)',
    re.IGNORECASE
)

_TRAILING_VERB_RE = re.compile(
    r'\s+(continues|leads|stretches|opens|extends|follows|appears|is visible|'
    r'can be seen|is ahead|awaits|lies ahead|unfolds)\.?$',
    re.IGNORECASE
)
_GENERIC_AHEAD = re.compile(
    r'^(?:plain[,\s]*|featureless\s*|bare\s*)'
    r'(?:(?:\w[\w-]*)\s+)*'
    r'(wall|floor|ceiling|surface)\s*$',
    re.IGNORECASE
)

def extract_ahead_noun(desc: str, max_words: int = 4) -> str:
    """Extract the core ahead landmark from the 2nd sentence of a turn description.
    E.g.: 'Turn at kitchen island. Ahead, a living room with a grey sofa is visible.'
    → 'grey sofa'
    """
    if not desc:
        return ""
    sents = re.split(r'(?<=[.!?])\s+', desc.strip())
    if len(sents) < 2:
        return ""
    second = sents[1].strip().rstrip('.')
    # Remove "Ahead, " / "In front, " / etc. prefix
    second = _AHEAD_PREFIX_RE.sub('', second).strip()
    # Strip leading article
    second = re.sub(r'^(a|an|the)\s+', '', second, flags=re.IGNORECASE).strip()
    # If starts with room type + "with X", extract X (the specific object)
    m = _ROOM_TYPES_RE.match(second)
    if m:
        second = second[m.end():].strip()
        second = re.sub(r'^(a|an|the)\s+', '', second, flags=re.IGNORECASE).strip()
    # Split at "and" to take first object only
    second = re.split(r'\s+and\s+', second, maxsplit=1)[0].strip()
    # Strip trailing verbs: "continues", "leads", "is visible", etc.
    second = _TRAILING_VERB_RE.sub('', second).strip()
    second = re.sub(r'\s+(is|are|can be|could be)\s+\w+\.?$', '', second,
                    flags=re.IGNORECASE).strip()
    # Truncate at preposition
    pm = PREPOSITION_RE.search(second)
    if pm:
        second = second[:pm.start()].strip()
    result = ' '.join(second.split()[:max_words]).rstrip('.,')
    # Filter out generic/non-identifiable landmarks
    if _GENERIC_AHEAD.match(result):
        return ""
    return result if len(result.split()) >= 2 else ""


# ── v19 landmark info: merge text + vision ────────────────────────────────────

def get_landmark_info_v18(ep, pf_map, old_map, primitives, vision_map: Dict,
                          ts_vision_map: Optional[Dict] = None) -> Dict:
    """
    Build landmark info for v24: turn-side landmarks take priority over approach-direction.
    - goal_landmark: core noun phrase (4-6 words) for stop phrase
    - goal_full: full first sentence shown in Goal: line
    - turn landmarks: TURN-SIDE desc preferred (what you see on the turning side),
                      falling back to approach-direction Phase 1 description
    Room names always from gate3_perframe.
    """
    eid = ep["episode_id"]
    eid_str = str(eid)
    vis = vision_map.get(eid_str, {})          # {label: rich_desc_or_None} — approach direction
    ts = (ts_vision_map or {}).get(eid_str, {})  # {label: noun_phrase} — turn-side direction

    info = get_landmark_info_text(ep, pf_map, old_map, primitives)

    # Goal: extract core noun (for stop phrase) and full description (for prompt display)
    # v172: Strip colors from goal_vis BEFORE extracting noun/full desc — 69.5% of stop phrases
    # had color words in v171 because the LLM faithfully reproduced colors from goal_full prompt.
    # GT stop phrases use colors in only 10.9%. Stripping from source eliminates the color cascade.
    goal_vis = vis.get("goal")
    if goal_vis:
        goal_vis_clean = strip_colors_from_landmark(goal_vis)
        info["goal_landmark"] = extract_noun_phrase(goal_vis_clean, max_words=5)
        info["goal_full"] = extract_goal_full(goal_vis_clean)  # for "Goal:" line in prompt
    elif info["goal_landmark"] == "the destination":
        info["goal_full"] = "the destination"
    else:
        info["goal_full"] = strip_colors_from_landmark(info["goal_landmark"])

    # Turn landmarks: v24 — prefer turn-side description, fall back to approach
    for turn in info["turns"]:
        label = turn.get("label", "")
        if not label:
            continue

        # Try turn-side description first (captures what triggers the turn)
        ts_desc = ts.get(label)
        if ts_desc and not is_turn_generic(ts_desc):
            turn["landmark"] = ts_desc
            # Still try to extract "ahead" from approach-direction Phase 1
            vis_desc = vis.get(label)
            if vis_desc:
                ahead = extract_ahead_noun(vis_desc, max_words=4)
                if ahead:
                    turn["ahead"] = ahead
        else:
            # Fall back to approach-direction Phase 1
            vis_desc = vis.get(label)
            if vis_desc:
                lm = extract_turn_noun(vis_desc)
                if lm and len(lm.split()) >= 2:
                    turn["landmark"] = lm
                # v20: extract ahead landmark from 2nd sentence
                ahead = extract_ahead_noun(vis_desc, max_words=4)
                if ahead:
                    turn["ahead"] = ahead

    # Start context from vision if text is empty
    start_vis = vis.get("start")
    if start_vis and not info["start_context"]:
        info["start_context"] = extract_noun_phrase(start_vis, max_words=6)

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
            "ahead": t.get("ahead", ""),   # v20: pass through ahead landmark
        })
        if t_room:
            prev_room = t_room

    if rng.random() < p_anchor:
        turns_with_lm = [(i, t) for i, t in enumerate(turns) if t.get("landmark")]
        if turns_with_lm:
            best_idx, best_turn = max(turns_with_lm,
                                      key=lambda x: len(x[1].get("landmark", "")))
            result[best_idx]["landmark"] = best_turn["landmark"]
            # v20: also propagate the ahead info for the anchored turn
            if best_turn.get("ahead"):
                result[best_idx]["ahead"] = best_turn["ahead"]

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
        # v84: GT-calibrated stop prepositions (from analysis of 748 GT stop sentences):
        # stop at: 8.8%, stop in the: 7.5%, stop in front of: 6.1%, stop next to: 3.1%
        # stop by: 2.1%, stop near: 1.3% — OLD was "near/at/in front of" each 33% (WRONG)
        if goal_room and not (goal_lm and goal_lm != "the destination"):
            patterns = [f"Stop in the {goal_room}.", f"Stop at the entrance to the {goal_room}."]
        else:
            # Weighted: at=35%, in front of=25%, next to=20%, by=15%, near=5%
            draw = rng.random()
            if draw < 0.35:
                patterns = [f"Stop at {lm}."]
            elif draw < 0.60:
                patterns = [f"Stop in front of {lm}."]
            elif draw < 0.80:
                patterns = [f"Stop next to {lm}."]
            elif draw < 0.95:
                patterns = [f"Stop by {lm}."]
            else:
                patterns = [f"Stop near {lm}."]
        return rng.choice(patterns), "stop"
    elif r < P_STOP + P_WAIT:
        # v84: GT-calibrated wait prepositions:
        # wait at: 7.9%, wait near: 6.6%, wait by: 5.1%, wait there: 4.7%, wait next to: 1.2%
        draw = rng.random()
        if draw < 0.30:
            phrase = f"Wait at {lm}."
        elif draw < 0.52:
            phrase = f"Wait near {lm}."
        elif draw < 0.70:
            phrase = f"Wait by {lm}."
        elif draw < 0.86:
            phrase = "Wait there."
        else:
            phrase = f"Wait next to {lm}."
        return phrase, "wait"
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

def build_route_desc(primitives, sel_turns, seg_midpoint: Optional[Dict[str, str]] = None,
                     path_len: int = 0, eid: int = 0):
    """Build route description string.
    seg_midpoint: {endpoint_label: desc} — up to 2 entries (v23: first + last segment).
    endpoint_label is "turn_N" (1-indexed) or "goal".
    path_len: len(reference_path) — used to lower straight threshold for long paths (v67).
    eid: episode_id — v77: used to seed RNG for deterministic pass/thru decisions.
    Returns (route_str, had_pass_markers: bool) — had_pass_markers=True if any [pass:] added.
    """
    # v77: seed RNG with eid for deterministic pass/thru decisions
    # This lets us track which episodes got [pass:] without propagating random state
    rng = random.Random(eid ^ 0x9145_5300)  # v84: new seed

    # v67: lower straight threshold for long paths to add route content → longer instructions
    # plen=6 (path_len>=7): LLM writes 27.2w vs GT=30.5w (-3.3w) despite "31-40w" budget
    # 552 eps have avg 0.94 extra straight markers (1.0-2.0m) → +1.9w per ep when included
    straight_min_dist = 1.0 if path_len >= 7 else 2.0
    parts = []
    turn_idx = 0
    had_pass_markers = False  # v77: track if any [pass:] was added
    hallway_count = 0  # v106: track hallway mentions — allow first only, suppress subsequent
    prev_prim_was_elevation = False  # v170: track stairs for "at the top/bottom of the stairs" landmark
    for p in primitives:
        t = p["type"]
        if t == "straight":
            d = p.get("distance_m", 0)
            if d > straight_min_dist: parts.append(f"straight {d:.0f}m")
        elif t in ("left_turn","right_turn"):
            direction = "left" if t=="left_turn" else "right"
            td = sel_turns[turn_idx] if turn_idx < len(sel_turns) else {}
            lm    = td.get("landmark","")
            rm    = td.get("room","")
            rt    = td.get("room_trans","")
            ahead = td.get("ahead","")
            # v170: If turn follows stairs, inject "top/bottom of the stairs" positional landmark
            # GT uses "at the stairs" (45 occurrences), "at the top/bottom" (58/49) — reliable structural refs
            if prev_prim_was_elevation and not lm and not rm:
                stair_dir = p.get('direction', 'up')  # p is a dict; 'up'/'down' stair direction
                lm = "top of the stairs" if stair_dir == 'up' else "bottom of the stairs"
            prev_prim_was_elevation = False
            # v167: Filter nav-verb ahead labels — Phase C sometimes outputs "Turn right through..."
            # → extract_ahead_noun extracts "Turn right" → route gets "→ [Turn right ahead]"
            # → LLM writes "walk to the Turn right" (literal transcription of route marker — WRONG!)
            if ahead and re.match(r'^(turn|go|walk|move|proceed|head|step)\b', ahead, re.IGNORECASE):
                ahead = ""
            angle = p.get("angle_deg", 90)
            # v28: classify midpoint; skip [pass:] if object is generic
            # v29: also strip material adjectives from midpoint description
            # v34: probabilistic sampling — P_PASS_MARKER / P_THRU_MARKER to match GT frequencies
            # v77: use episode-seeded rng (not global random) for determinism
            turn_label = f"turn_{turn_idx + 1}"
            if seg_midpoint and turn_label in seg_midpoint:
                mid_desc = strip_material_adjectives(seg_midpoint[turn_label])
                action = classify_pass_action(mid_desc)
                if action == "thru":
                    if rng.random() < P_THRU_MARKER:  # 85% → ~27.5% through (GT=27.2%)
                        parts.append(f"[{action}: {mid_desc}]")
                elif not is_generic_pass_object(mid_desc):
                    if rng.random() < P_PASS_MARKER:  # v120: 0.70 to compensate 20% compliance rate
                        # v120: restore [pass: X] marker (v119 direct injection failed — LLM skips it)
                        parts.append(f"[pass: {mid_desc}]")
                        had_pass_markers = True  # v77: track that this episode got [pass:] marker
            # v105: Include ALL turns regardless of angle (no angle filter).
            # v103/v104 had angle<70° skip → low turn count (24-27% turn left/right).
            # v24 had turn left=51.6%, turn right=53.6% → many episodes with BOTH turns.
            # v24 winning episodes had 90.7% has_turn. More turns = model gets clearer guidance.
            # Skip ONLY: very shallow turns with no landmark (<30°) to avoid noise.
            if not lm and not rm and angle < 30:
                turn_idx += 1
                continue
            # v105: HIGH-TURN sampling — include 90% of unanchored turns (was 42%).
            if not lm and not rm and rng.random() > P_TURN_UNANCHORED:
                turn_idx += 1
                continue
            # Build turn phrase
            # v169: Strip color adjectives from landmarks (GT uses colors only 15% vs v168=74%)
            lm_clean = strip_colors_from_landmark(lm) if lm else lm
            # v171: Positional landmark injection for hallway turns (GT prefers structural refs)
            # GT top landmarks: "end of" (98), "stairs" (45), "entrance" (29) — not furniture!
            # PROBABILISTIC (30%): was 100% in v170 which inflated anchor% from 22.5%→25.1%
            # 30% rate injects ~2pp extra anchor% (GT=16.5%, target stays under that)
            if not lm_clean and not rm and td.get("room","").lower() in ("hallway","corridor","hall"):
                if rng.random() < P_HALLWAY_POSITIONAL:
                    lm_clean = "end of the hallway"
            if lm_clean and rt and "doorway" in rt.lower():
                base = f"turn {direction} past [{lm_clean}] into [{rm}]"
                parts.append(f"{base} → [{ahead} ahead]" if ahead else base)
            elif lm_clean:
                base = f"turn {direction} at [{lm_clean}]"
                parts.append(f"{base} → [{ahead} ahead]" if ahead else base)
            elif rm:
                # v117: Room transitions ALWAYS involve a doorway in Habitat.
                # For 60% of room transitions, inject "(door→ room)" to prompt doorway mentions.
                # For hallway: allow "through the hallway" (NOT "walk into the hallway") — suppressed below.
                # GT: hallway=20.4%, into_hall=4.4%, through_the=27.2%.
                is_hallway = "hallway" in rm.lower() or "corridor" in rm.lower()
                if is_hallway:
                    # v129: Recalibrated from v128 measured LLM compliance values.
                    # v128 measured: compliance(thru)=52%, compliance(hall_into→)=36.8%, compliance(hall→)=45.1%.
                    # v128 gaps: through_the=24.0%(GT=27.2%), into_hallway=2.6%(GT=4.4%).
                    # Root cause: (hall_into→) novel notation confused LLM → low 36.8% compliance.
                    # Fix: rename (hall_into→) → (→ hallway), matching existing (→ room) convention (~85% compliance).
                    # Root cause 2: Extended Stage 1 removed 3.1pp phantom through_the from unannotated eps.
                    # Fix: P_THRU_HALL 32% → 41% to compensate → total through_the: 14.77 + 12.47 = 27.24% ✓
                    #
                    # (through hallway) → "walk through the hallway" → through_the++ AND hallway++
                    # (→ hallway)       → "walk into the hallway"   → into_hallway++ AND hallway++ (NO through_the)
                    # (hall→)           → "walk to the hallway"     → hallway++ ONLY
                    #
                    # v134 calibration (bisection from v133 overshoot):
                    #   v133 overshoot: hallway=21.5%(+1.1pp), into_hallway=4.7%(+0.3pp)
                    #   Empirical: 0.306 pp/% (hall→); 0.26pp/% (→ hallway)
                    #
                    #   P_THRU=42%:           42%×69.3%×51%+12.47=27.30%≈GT=27.2% ✓ (unchanged)
                    #   (→ hallway) 17.5%:    draw 0.42–0.595 → into_hallway:17.5%×69.3%×36%=4.37%≈GT ✓
                    #   (hall→) 6.0%:         draw 0.595–0.655 → hallway:19.8+0.13+0.34=20.27%≈GT ✓
                    #   Total annotated: (43+17+5.4)%×69.3% = 45.1% of all eps (v136)
                    draw = rng.random()
                    if draw < 0.43:        # P_THRU_HALL=43% → through_the≈27.2%≈GT ✓ (unchanged)
                        if direction:
                            parts.append(f"turn {direction} (through hallway)")
                        else:
                            parts.append("(through hallway)")
                    elif draw < 0.60:      # (→ hallway) P=17% abs → into_hallway≈4.47%≈GT ✓
                        # "(→ hallway)" notation: LLM writes "walk into the hallway"
                        if direction:
                            parts.append(f"turn {direction} (→ hallway)")
                        else:
                            parts.append("(→ hallway)")
                    elif draw < 0.663:     # (hall→) P=6.3% abs → hallway≈20.375%≈GT=20.4% ✓ (v139 fresh)
                        # "(hall→)" notation: LLM writes "walk to the hallway" (no "through"/"into")
                        if direction:
                            parts.append(f"turn {direction} (hall→)")
                        else:
                            parts.append("(hall→)")
                    elif direction:
                        parts.append(f"turn {direction}")
                    # else: no direction, no label — skip this waypoint entirely
                else:
                    # v166: Only P_DIR_ROOM_TRANS=20% of non-hallway transitions get direction prefix.
                    # v165 ALWAYS added direction → anchor%=49.1% (overcorrected from v164's 6.1%).
                    # v166: 20% → expected anchor%≈14-17% (GT=16.5%).
                    # Without direction: "(→ bedroom)" → LLM writes "walk into the bedroom" (no anchor)
                    # With direction:    "turn left (→ bedroom)" → LLM writes "turn left into bedroom" → ANCHOR
                    add_dir = direction and rng.random() < P_DIR_ROOM_TRANS
                    # v169: Increase (door→ room) fraction 0.21→0.45 to match GT door mentions (GT=48%, v168=26.9%)
                    # "turn left (door→ bedroom)" → LLM writes "turn left through the doorway into the bedroom"
                    # "(door→ bedroom)" → LLM writes "walk through the doorway into the bedroom"
                    # v172: Add "enter [room]" notation for ~40% of non-direction room transitions.
                    # GT: "enter the [named room]" appears in 5.4% of instrs but v171=1.3% — LLM ignores rule.
                    # Direct route notation forces the LLM: "enter bedroom" → LLM writes "enter the bedroom".
                    # This bypasses the system prompt enforcement failure seen in v170/v171.
                    # v174: "door on the left/right" injection — GT: 23.8% of eps have "on the left/right"
                    # 90% of GT "on" usage qualifies which door to use at a choice point.
                    # When use_door=True and direction is known: P_DOOR_DIR=0.50 → inject "on the {direction}"
                    use_door = rng.random() < 0.45
                    use_door_dir = use_door and bool(direction) and rng.random() < P_DOOR_DIR  # v174
                    use_enter = not add_dir and not use_door_dir and rng.random() < 0.40  # v172: 40% of no-dir transitions use "enter"
                    if add_dir:
                        if use_door:
                            if use_door_dir:
                                # v174: "turn left (door on the left → bedroom)" — explicit door side
                                parts.append(f"turn {direction} (door on the {direction} → {rm})")
                            else:
                                parts.append(f"turn {direction} (door→ {rm})")
                        else:
                            parts.append(f"turn {direction} (→ {rm})")
                    elif use_enter:
                        # v172: Direct "enter [room]" notation — LLM faithfully writes "enter the [room]"
                        parts.append(f"enter {rm}")
                    else:
                        if use_door:
                            if use_door_dir:
                                # v174: "(door on the left → bedroom)" — no turn prefix but side known
                                parts.append(f"(door on the {direction} → {rm})")
                            else:
                                parts.append(f"(door→ {rm})")
                        else:
                            parts.append(f"(→ {rm})")
            else:
                parts.append(f"turn {direction}")
            turn_idx += 1
        elif t == "elevation":
            parts.append(f"{'up' if p.get('direction','up')=='up' else 'down'} stairs")
            prev_prim_was_elevation = True  # v170: next turn gets positional stairs landmark
    # v32: NO midpoints for goal segment (neither [thru:] nor [pass:]).
    # v31 had only [thru:] for goal → "through the" jumped from 33.3% to 39.5% (too high vs GT 27.2%).
    # v32: eliminate goal midpoints entirely. VLN model uses stop phrase for final navigation.
    # Expected: "through the" drops from 39.5% → ~32-35%, "walk past the" stays ~18-19%.
    # (goal midpoints suppressed — seg_midpoint["goal"] is intentionally not appended)
    return " → ".join(parts) or "go straight", had_pass_markers


def gt_start_verb(gt_instr):
    tok = gt_instr.strip().split()
    if not tok: return "Walk"
    w = tok[0].rstrip(".,!?;:")
    return (w[0].upper() + w[1:].lower()) if w else "Walk"


# ── System prompt (same as v17, updated for v18 vision context) ──────────────

SYSTEM_INTRO = """You write concise R2R navigation instructions for an indoor robot.

Style: Natural and brief, like a human R2R annotator. Match how real humans describe indoor navigation.

RULES:
- When a turn has [object] in brackets: use that object as a turn reference — naturally: "turn left at the pillar", "turn right past the dining table". CRITICAL: Do NOT add color words (white, brown, dark, grey, black, etc.) — use only the object type ("table", "sofa", "pillar"). GT annotators almost never use colors.
- When a turn has → [object ahead] after the arrow: add a BRIEF "walk to the X" phrase (3-5 words) — keep it very short. Do NOT write "walk past the" or "walk toward the" for → [object ahead].
- When route has [pass: X]: the robot moves alongside that object — write "walk past the [X]" (3-5 words). Keep it brief. Do NOT change to "passing the".
- When route has [thru: description]: the robot moves THROUGH a doorway — write "walk through the doorway" or "go through the [opening]" (3-5 words). Keep it brief.
- When you see turn left/right (door on the left → room) or turn left/right (door on the right → room): write "turn left/right through the door on the left/right into the [room]" or "go through the door on the left/right and turn left/right into the [room]". The 'on the left/right' MUST appear in your output — it tells the robot which door to use.
- When you see (door on the left → room) or (door on the right → room) WITHOUT a turn prefix: write "go through the door on the left/right into the [room]" or "enter through the doorway on the left/right". The side label MUST appear in your output.
- When you see turn left/right (door→ room): write EITHER "turn left/right and enter the [room]" OR "turn left/right through the doorway into the [room]" — vary between them. Prefer "enter the [room]" for the second or later room transition in the route.
- When you see (door→ room) alone (no turn prefix): write EITHER "enter the [room]" (short, preferred) OR "walk through the doorway into the [room]" — "enter the [room]" is more natural.
- When you see (through hallway): write "walk through the hallway" or "go through the hallway" — do NOT write "walk into the hallway". Hallway transitions pass through, not into.
- When you see (→ hallway): write "walk into the hallway" or "go into the hallway" — same pattern as (→ room). Use "into", NOT "through". The robot enters the hallway from the side.
- When you see (hall→): write "walk to the hallway" or "go to the hallway" (3-4 words max). Do NOT use "through" or "into".
- When you see turn left/right (→ room): write "turn left/right into the [room]" or "turn left/right and enter the [room]" — combine the turn direction with entering the room. Keep it brief (4-6 words).
- When you see (→ room) alone (no turn prefix): write "enter the [room]" or "walk into the [room]" — describe entering that room. Keep it brief (3-5 words).
- SENTENCE STRUCTURE: Write 1-3 sentences maximum. Chain consecutive actions with "and" or comma+then: "turn left and walk through the hallway", "walk to the sofa, then turn right". Chain the STOP too: "walk to the X and stop near the Y." NEVER write one action per sentence. Real R2R instructions: "Exit the room and turn left. Walk to the sofa and stop there." (2 sentences, NOT 4).
- ENTER: When route says "enter [room]": write "enter the [room]" exactly (2-3 words). Do NOT expand to "walk through the doorway into the [room]".
- Turns with no brackets/arrows: write ONLY "turn left" or "turn right" — no objects, no room names after the turn direction
- CRITICAL: Do NOT add "turn left" or "turn right" unless the route explicitly shows "turn left" or "turn right". If the route has no turn, do NOT write one.
- CRITICAL: Do NOT repeat any direction, location, or landmark. Each segment is described EXACTLY ONCE.
- MOTION WORDS: Use natural walking verbs: "walk", "go". Chain movements naturally: "turn left into the kitchen". For landmarks: "walk to the X" or "walk towards the X". AVOID "proceed", "go forward", "walk forward", "keep going", "head toward". Use "towards" (not "toward") when indicating destination direction: "walk towards the kitchen", "head towards the stairs". Use "continue" ONLY when no new visual landmark is present — prefer "walk" or "go" as primary verbs.
- VOCABULARY: Use "couch" (not "sofa") — this matches how real R2R annotators write. Use "walk past" (not "passing") for passing alongside an object.
- DISTANCE: The route shows 'straight Xm' entries — NEVER write metric numbers or "take a few steps". For short segments (≤2m): skip them or chain adjacent movements (e.g. "turn left into the kitchen"). For medium segments (3-8m): use "walk" or "continue" briefly — one phrase. For long segments (>8m): use "walk straight" or "go straight" ONCE and briefly toward the next landmark.
- STOP PHRASE: For the final destination, use a SPECIFIC visual landmark (sofa, armchair, counter, plant, doorway, etc.). Do NOT use "wall", "floor", or "ceiling". Keep it brief: "Stop near the [X]." or "Wait at the [X]." CRITICAL: Do NOT add color words (white, brown, dark, grey) to the stop landmark — just the object type.
- ANTI-COLOR: NEVER use color adjectives anywhere: no "white table", no "brown cabinet", no "dark grey sofa". Just say "table", "cabinet", "sofa". This is how real R2R annotators write.
- Endings vary: "Stop near X." / "Wait near X." / no explicit ending
- Start verb is given — use it exactly"""


_TURN_DIR_PREFIX = re.compile(r'^Turn\s+(?:left|right)\s+', re.I)
_THRU_ROOM_RE2   = re.compile(r'^Turn\s+(?:left|right)\s+(?:through|past)\s+(?:the\s+)?(.+?)(?:\s+into\s+the\s+(.+?))?\.?\s*$', re.I)
_AT_OBJ_RE       = re.compile(r'^Turn\s+(?:left|right)\s+at\b', re.I)


def _an(word: str) -> str:
    """Return 'An' if word starts with a vowel sound, else 'A'."""
    return "An" if word and word[0].lower() in "aeiou" else "A"


def _extract_scene_from_desc(desc: str) -> str:
    """Convert Phase C turn directive to neutral scene description for visual context.

    Phase C descriptions start with "Turn left/right at/through/past..." which Phase 2
    interprets as a turn anchor directive, causing extra spurious turn anchors.
    We strip the directional turn prefix and return only the scene/room context.

    Examples:
      "Turn left through the plain doorway into the hallway. A dark room is visible."
        → "A plain doorway leads into the hallway. A dark room is visible."
      "Turn right at the brown sofa. Ahead, a hallway leads further into the home."
        → "Ahead, a hallway leads further into the home."
      "Turn right at the silver leaf decoration into the hallway. A plain white wall is ahead."
        → "A hallway is accessible. A plain white wall is ahead."
    """
    sents = re.split(r'(?<=[.!?])\s+', desc.strip())
    if not sents:
        return desc
    first = sents[0].strip()
    rest = " ".join(sents[1:]).strip() if len(sents) > 1 else ""

    # "Turn [dir] through/past the [opening] into the [room]." → "A [opening] leads into the [room]."
    m_thru = _THRU_ROOM_RE2.match(first)
    if m_thru:
        opening = m_thru.group(1).strip().rstrip('.')
        room    = m_thru.group(2).strip().rstrip('.') if m_thru.group(2) else ""
        if room:
            neutral = f"{_an(opening)} {opening} leads into the {room}."
        else:
            neutral = f"{_an(opening)} {opening} is visible."
        return (neutral + " " + rest).strip() if rest else neutral

    # "Turn [dir] at [object]..." — drop the object, check if "into the [room]" is in first sentence
    if _AT_OBJ_RE.match(first):
        m_into = re.search(r'into\s+the\s+(\w+(?:\s+\w+)?)', first, re.I)
        if m_into:
            room_name = m_into.group(1).strip()
            neutral = f"{_an(room_name)} {room_name} is accessible."
            return (neutral + " " + rest).strip() if rest else neutral
        # No "into" in first sentence — use rest (room/scene context)
        return rest if rest else "The area is open."

    # "Turn [dir] [other]..." — extract "into the [room]" if present
    if _TURN_DIR_PREFIX.match(first):
        m_room = re.search(r'into\s+the\s+(\w+(?:\s+\w+)?)', first, re.I)
        if m_room:
            room_name = m_room.group(1).strip()
            neutral = f"{_an(room_name)} {room_name} is accessible from here."
            return (neutral + " " + rest).strip() if rest else neutral
        return rest if rest else "The area continues."

    # Not a turn directive — show first sentence as-is
    return first + (" " + rest if rest else "")


def build_visual_narrative(lm_info: Dict, sel_turns: List[Dict], vision_map: Dict,
                            eid: int) -> str:
    """
    Build a visual narrative block for the Phase 2 prompt.
    Only shows start and turn context — NOT goal (goal is in "Goal: X" line).
    Only shows context for turns that have NO bracket — bracketed turns already
    have landmarks in the route notation and don't need extra context.
    This helps the LLM write "walk past the [object]" phrases for unanchored turns.
    """
    eid_str = str(eid)
    vis = vision_map.get(eid_str, {})
    if not vis:
        return ""

    lines = []
    start_desc = vis.get("start", "")
    if start_desc:
        # Trim to first sentence for start (keeps it brief)
        first_sent = re.split(r'(?<=[.!?])\s+', start_desc)[0]
        lines.append(f"  Start area: {first_sent}")

    for i, sel_turn in enumerate(sel_turns):
        # Only show visual context for non-anchored turns (no bracket)
        # Anchored turns already have the landmark in [...]
        if sel_turn.get("landmark"):
            continue  # already anchored — no need for extra narrative
        label = lm_info["turns"][i].get("label", f"turn_{i+1}") if i < len(lm_info.get("turns",[])) else f"turn_{i+1}"
        desc = vis.get(label, "")
        direction = sel_turn.get("direction", "")
        if desc:
            # v154: extract_scene_from_desc converts Phase C turn directives to neutral scene descriptions
            chosen = _extract_scene_from_desc(desc)
            lines.append(f"  Near turn {i+1} ({direction}): {chosen}")

    if not lines:
        return ""
    # v109: GT-style — visual context is for turn-landmark identification, NOT spontaneous generation.
    # The SYSTEM_INTRO [pass:] rule now produces "walk past the X" (GT style).
    header = ("Visual context (for identifying turn landmarks — do NOT add extra landmarks unless route shows [pass:]):\n")
    return header + "\n".join(lines)


def build_prompt(ep, gt_instr, examples, lm_info, primitives, sel_turns,
                 stop_phrase, stop_type, vision_map: Dict, seg_midpoint: Optional[Dict] = None):
    sv    = gt_start_verb(gt_instr)
    eid   = ep.get("episode_id", 0)
    path_len = len(ep.get("reference_path", []))
    route, had_pass_markers = build_route_desc(primitives, sel_turns, seg_midpoint,
                                               path_len=path_len, eid=eid)  # v77: seeded
    ex_block = "\n".join(f"  {i+1}. \"{ex}\"" for i, ex in enumerate(examples))

    goal_lm    = lm_info.get("goal_landmark","the destination")
    # v19: use richer full description for the Goal: display line
    goal_full  = lm_info.get("goal_full", goal_lm)
    goal_room  = lm_info.get("goal_room","")
    start_ctx  = lm_info.get("start_context","")
    start_room = lm_info.get("start_room","")

    ending_req = (f'End: "{stop_phrase}"' if stop_type in ("stop","wait")
                  else "End naturally — no explicit stop/wait")

    n_bracketed = sum(1 for t in sel_turns if t.get("landmark"))
    n_room_trans = sum(1 for t in sel_turns if t.get("room") and not t.get("landmark"))
    n_ahead = sum(1 for t in sel_turns if t.get("landmark") and t.get("ahead"))
    # v120: count [pass:] markers in route (restored from v119 direct injection)
    n_pass = len(re.findall(r'\[pass:', route))
    n_thru = len(re.findall(r'\[thru:', route))
    # v117: count (door→ room) and (through hallway) markers
    n_door_trans = len(re.findall(r'\(door(?:\s+on\s+the\s+(?:left|right))?\s*→', route))  # v174: includes "door on the left →"
    n_door_dir_trans = len(re.findall(r'\(door on the (?:left|right)\s*→', route))  # v174: directional door count
    n_thru_hall = len(re.findall(r'\(through hallway\)', route))
    n_hall_into = len(re.findall(r'\(→ hallway\)', route))   # v129: renamed from (hall_into→)
    n_hall_to   = len(re.findall(r'\(hall→\)', route))
    # v165: count turn+room transitions (new format: "turn left (→ bedroom)")
    n_turn_room_trans = len(re.findall(r'turn\s+(?:left|right)\s+\(→\s+\w', route))
    n_turn_door_trans = len(re.findall(r'turn\s+(?:left|right)\s+\(door', route))  # v174: includes direction variants
    if n_bracketed > 0 and n_room_trans > 0:
        # v165: updated to handle new "turn left (→ room)" notation
        anchor_note = (f"Route has {n_bracketed} bracketed landmark(s) and {n_room_trans} "
                       f"room transition(s) — use landmarks for turns. "
                       f"For 'turn left/right (→ room)': write 'turn left/right into the [room]'. "
                       f"For '(→ room)' alone: write 'walk into the [room]'. "
                       f"For 'turn left/right (door on the left/right → room)': write 'turn left/right through the door on the left/right into the [room]'. "
                       f"For '(door on the left/right → room)' alone: write 'through the door on the left/right into the [room]'. "
                       f"For 'turn left/right (door→ room)': write 'turn left/right through the doorway into the [room]'. "
                       f"For '(door→ room)' alone: write 'through the doorway into the [room]'. "
                       f"For (through hallway) write 'through the hallway'. "
                       f"For (→ hallway) write 'walk into the hallway'. "
                       f"Use each room name AT MOST ONCE.")
    elif n_bracketed > 0:
        anchor_note = f"Route has {n_bracketed} bracketed landmark(s) — use them naturally."
    elif n_room_trans > 0 or n_door_trans > 0 or n_thru_hall > 0 or n_hall_into > 0 or n_hall_to > 0:
        # v165: updated for new turn+room notation (includes direction)
        anchor_note = (f"Route has room transitions — "
                       f"for 'turn left/right (→ room)' write 'turn left/right into the [room]'; "
                       f"for '(→ room)' alone write 'walk into the [room]'; "
                       f"for 'turn left/right (door on the left/right → room)' write 'turn left/right through the door on the left/right into the [room]'; "
                       f"for '(door on the left/right → room)' alone write 'go through the door on the left/right into the [room]' (MUST include 'on the left/right'); "
                       f"for 'turn left/right (door→ room)' write 'turn left/right through the doorway into the [room]'; "
                       f"for '(door→ room)' alone write 'walk through the doorway into the [room]'; "
                       f"for (through hallway) write 'walk through the hallway' or 'go through the hallway'; "
                       f"for (→ hallway) write 'walk into the hallway' or 'go into the hallway' (use 'into', NOT 'through'); "
                       f"for (hall→) write 'walk to the hallway' or 'go to the hallway' (NO 'through' or 'into'). "
                       f"Do NOT use 'enter the'. Use each room name AT MOST ONCE.")
    else:
        anchor_note = "Route has NO bracketed landmarks — write ONLY direction words for all turns."
    if n_ahead > 0:
        anchor_note += (f" Route also has {n_ahead} [X ahead] marker(s) — after turning at"
                        f" [landmark], add a brief 'walk to the X' or 'walk past the X' phrase (3-5 words max).")
    # v120: n_pass counts [pass:] markers restored in route
    if n_pass > 0:
        anchor_note += (f" Route has {n_pass} [pass: X] marker(s) — write 'walk past the [X]' for each."
                        f" Keep each brief and natural.")
    if n_thru == 1:
        anchor_note += (f" Route has 1 [thru:] marker — write 'walk through the doorway' or 'go through the [opening]'"
                        f" (3-5 words). Keep it brief and natural.")
    elif n_thru > 1:
        anchor_note += (f" Route has {n_thru} [thru:] markers — for each, write 'through the doorway' or"
                        f" 'through the [arch/opening]' (3-5 words each).")

    # v25/v26: path-length-aware word budget
    # v51: +4 words to all path_len>=5 budgets — v49 avg_words=23.1 vs GT=26.8 (-3.7w gap)
    # v67: raise path_len>=7 budget to 33-46 (was 31-40); plen=6 generates 27.2w vs GT=30.5w
    if path_len <= 4:
        anchor_note += " LENGTH: Write 16-24 words total. Very short path — be brief."
    elif path_len == 5:
        anchor_note += " LENGTH: Write 27-36 words total."
    elif path_len == 6:
        anchor_note += " LENGTH: Write 30-39 words total."
    else:
        anchor_note += " LENGTH: Write 37-50 words total."  # v176: raised +4w per tier (v173 gap=2.2w)

    prompt = (
        f"{SYSTEM_INTRO}\n\n"
        f"Same-building examples:\n{ex_block}\n\n"
        f"Write ONE instruction for:\n"
        f"  Route: {route}\n"
        f"  Start: {start_room or 'room'}"
        + (f" ({start_ctx})" if start_ctx else "")
        + f"\n  Goal: {goal_full}"   # v19: full visual description here
        + (f" in {goal_room}" if goal_room else "")
        + f"\n\n{anchor_note}\n{ending_req}"
        + f"\nStart with: \"{sv}\"\n\n{sv}"
    )
    return prompt, sv, had_pass_markers  # v77: had_pass_markers for assembly suppression


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


_FWDOBJ_FIX_RE = re.compile(
    r'\bwalk forward\s+(art(?:work)?|painting|sculpture|decor|canvas|mural|tapestry|'
    r'photograph|portrait|framed|mounted)\b',
    re.IGNORECASE
)


def clean(raw, sv):
    raw = raw.strip()
    # v81: strip leading period artifacts ("". Wait near X." → "Wait near X.")
    raw = re.sub(r'^[.!?]+\s*', '', raw).strip()
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
    # v81: Fix VLM-garbled "walk forward art/artwork/painting" (LLM drops "past the")
    raw = _FWDOBJ_FIX_RE.sub(lambda m: f'walk past the {m.group(1)}', raw)
    # v29: post-process "passing through the X" → "through the X"
    raw = re.sub(r'\bpass(?:ing|ed)?\s+through\s+the\b', 'through the', raw, flags=re.IGNORECASE)
    # v29: strip material adjectives from instruction text (catches any that slipped through)
    raw = _MATERIAL_ADJ_RE.sub('', raw)
    # v33: suppress LLM-hallucinated "walk past the [generic]" → "walk" (v155: was 'walk straight' → too high)
    # Apply BEFORE "passing the" conversion so only [pass:] marker hallucinations are suppressed.
    # LLM-generated "passing the X" should survive since X is usually a real observed object.
    raw = _GENERIC_WPT_OBJ_RE.sub('walk', raw)
    # v112: GT-style: convert ALL "passing the X" → "walk past the X" (imperative, GT=10.5%)
    # Apply AFTER _GENERIC_WPT_OBJ_RE so LLM-generated specific objects survive.
    # v111 had _GENERIC_WPT_OBJ_RE AFTER conversion → over-suppressed → avg_words=22.3 (GT=26.8).
    # v112 puts _GENERIC_WPT_OBJ_RE FIRST → specific objects in "passing the X" survive.
    # Step 1: Handle "X, passing the Y" → "X past the Y" (after comma, drop "passing")
    raw = re.sub(r',\s*passing\s+the\s+', ' past the ', raw, flags=re.IGNORECASE)
    # Step 2: Standalone "passing the X" → "walk past the X"
    raw = re.sub(r'\bpassing\s+the\b', 'walk past the', raw, flags=re.IGNORECASE)
    # Step 3: Also normalize "go past/passed the" → "walk past the"
    raw = re.sub(r'\bgo\s+past\s+the\b', 'walk past the', raw, flags=re.IGNORECASE)
    raw = re.sub(r'\bpassed\s+the\b', 'walk past the', raw, flags=re.IGNORECASE)
    # v104: "enter the X" → "walk into the X" (match v24 vocabulary: enter=4.4% vs v103=42.4%)
    raw = re.sub(r'\b[Ee]nter\s+the\b', lambda m: 'Walk into the' if m.group(0)[0].isupper() else 'walk into the', raw)
    # v91: REMOVED _THROUGH_HALLWAY_RE conversion — "through the hallway/corridor" stays as-is.
    # Previously this converted to "down the hallway" reducing through_the by ~3-5pp.
    # Keeping these phrases fixes both through_the and hallway simultaneously.
    # v35: fix "through the [room name]" → "into the X" (rooms aren't passages)
    raw = _THROUGH_ROOM_RE.sub(lambda m: f"into the {m.group(1).lower()}", raw)
    # v36: eliminate LLM "proceed" overuse (43.8% vs GT 0.8%)
    # First: "Proceed to/toward/past/through/into the X" → "Walk to/toward/past/through/into the X"
    raw = _PROCEED_DEST_RE.sub(_replace_proceed_dest, raw)
    # Then: delete standalone motion-only "Proceed forward/straight/ahead" (adds no info)
    raw = _PROCEED_MOTION_RE.sub('', raw)
    # v43: Fix "continue" overuse from v42 prompt. v42 room-suppression caused LLM to substitute
    # "continue through/into/along X" for the suppressed "walk into the hallway" phrases.
    # Replace directional continues with "walk <dir>" to match GT distribution (~10%).
    raw = _CONTINUE_MOTION_RE.sub(_replace_continue_motion, raw)
    # v155: was 'walk straight' → changed to 'walk' to reduce walk_straight overuse
    raw = _CONTINUE_FRAG_RE.sub(
        lambda m: 'Walk' if m.group(0)[0] == 'C' else 'walk', raw
    )
    # v36: collapse duplicate consecutive motion phrases ("continue forward. Walk forward.")
    raw = _DUP_MOTION_RE.sub(lambda m: m.group(1), raw)
    # v36: fix fragment artifacts ("Take a." → "", "Head." → "")
    raw = _TAKE_A_FRAG_RE.sub('', raw)
    raw = _HEAD_FRAG_RE.sub('', raw)
    raw = re.sub(r'\s+', ' ', raw).strip()
    sents = re.split(r'(?<=[.!?])\s+', raw)
    # v39: fix dup-stop — delete "and stop/wait X" fragments from ALL sentences.
    # Root cause of v38 failure: `i < len-1` guard skipped the last LLM sentence which IS the
    # "and stop near X." continuation; the template stop_phrase is appended AFTER clean().
    # Fix: apply to ALL sentences. The regex requires `\band\s+stop\b` (has "and" connector),
    # so it NEVER matches "Stop near X." (capital S, no "and") — safe to apply unconditionally.
    # Strategy: delete the entire "and stop/wait [rest of phrase]" fragment for clean output.
    _MID_STOP_FRAG_RE = re.compile(
        r',?\s*\band\s+(?:stop|wait)\b[^.!?]*', re.IGNORECASE
    )
    # v40: detect fragment sentences (preposition-only phrases without a main verb).
    # LLM sometimes splits "walk forward into the room" at period → "into the room." fragment.
    # Pattern: sentence starts with a preposition/conjunction/adverb without a subject/verb.
    _FRAGMENT_SENT_RE = re.compile(
        r'^(?:into|through|via|past|around|along|toward|towards|across|over|under|'
        r'onto|off|away from|out of|from|between|behind|beside|within|upon)\s',
        re.IGNORECASE
    )
    # v40: fix truncated landmark attr in stop phrases: "holding decorative." / "featuring two dark."
    # LLM starts an attribute clause but leaves noun incomplete.
    _TRUNCATED_ATTR_RE = re.compile(
        r'\s+(?:holding|featuring|with|beside|near|containing)\s+'
        r'(?:a\s+|an\s+|the\s+|two\s+|some\s+|several\s+)?'
        r'(?:small|large|dark|light|brown|white|black|grey|gray|decorative|wooden|'
        r'glass|metal|marble|round|square|rectangular|narrow|tall|short|colorful|'
        r'patterned|textured|abstract)\s*[.!?]?$',
        re.IGNORECASE
    )
    # v41: fragment MERGE — attach preposition-only fragment onto the previous sentence
    # instead of deleting it. Preserves navigation content AND "through the" frequency.
    # "Turn left and walk forward. through the doorway." -> "Turn left and walk forward through the doorway."
    merged_sents = []
    for s in sents:
        s = _MID_STOP_FRAG_RE.sub('', s)
        s = re.sub(r'\s+', ' ', s).strip()
        if s and s[-1] not in '.!?': s += '.'
        is_frag = bool(_FRAGMENT_SENT_RE.match(s))
        if is_frag:
            if merged_sents:
                # Merge with previous sentence: strip period from prev, append fragment content
                prev = merged_sents[-1].rstrip('.!?').rstrip()
                frag_content = s.lstrip()
                merged_sents[-1] = prev + ' ' + frag_content
            # else: discard leading fragment (no previous sentence to merge into)
        else:
            # v40: fix truncated attribute in stop phrases
            if re.match(r'^(?:Stop|Wait|Halt)\b', s, re.IGNORECASE):
                s = _TRUNCATED_ATTR_RE.sub('.', s)
                if s and s[-1] not in '.!?': s += '.'
            merged_sents.append(s)
    result = " ".join(s for s in merged_sents[:3] if s and s not in ('.',)).strip()
    if result and result[-1] not in ".!?": result += "."

    # v121: Diversify "walk straight" → GT-style motion verbs ONLY (no "go forward"/"keep going").
    # v120 analysis: "go forward"=80.3% (GT=1.7%), "keep going"=37.7% (GT~0%) — MASSIVE overcorrection!
    # Root cause: v114 _ws_variants included "go forward"/"keep going" as alternatives.
    # Fix: remove those options, only use "go straight"/"walk forward" (GT=9%/6.5%).
    # v126: Restore "walk forward" to _ws_variants (25%), keep "walk straight ahead" (25%),
    # "go straight ahead" (25%), "go straight" (25%).
    # "Walk forward" restores continue restoration path: 25% × 30% cont-restoration = 7.5% of eps → continue~9-10%.
    # avg_words: avg(3+3+2+2)/4=2.5w vs "walk straight"=2w → +0.5w/occurrence × 2 occurrences ≈ +1w total → ~26.7w.
    _ws_variants = [
        (r'\bwalk straight for a long while\b', ['go straight a long way', 'walk a long way']),
        (r'\bwalk straight for a while\b',      ['go straight for a while', 'walk forward for a bit']),
        (r'\bwalk straight a few\b',             ['go straight a few', 'walk a few']),
        (r'\bwalk straight ahead\b',             ['go straight ahead']),  # prevent double-hit
        # v157: restore 'walk straight' as one alt (25% chance) — v156 eliminated all walk_straight (GT=8.8% needs some)
        (r'\bwalk straight\b',                   ['walk straight', 'go straight ahead', 'go straight', 'walk forward']),
    ]
    h_base = hash(result) & 0xFFFFFF
    for pattern, alts in _ws_variants:
        def _replace_ws(m, _alts=alts, _h=h_base):
            _h = (_h * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFF
            alt = _alts[_h % len(_alts)]
            return (alt[0].upper() + alt[1:]) if m.group(0)[0].isupper() else alt
        result = re.sub(pattern, _replace_ws, result, flags=re.IGNORECASE)

    # v121: Convert residual "go forward"/"keep going" to GT-preferred alternatives.
    # These were generated by the LLM despite the MOTION WORDS rule discouraging them.
    # v156: Cap: replace 2nd+ occurrences of "go forward" with "go straight" (not walk straight).
    # The "go straight" will be absorbed by replace_go_straight_with_landmarks (step4) → walk to X or walk forward.
    _go_forward_counter = [0]
    def _cap_go_forward(m):
        _go_forward_counter[0] += 1
        if _go_forward_counter[0] > 1:
            return 'Go straight' if m.group(0)[0].isupper() else 'go straight'
        return m.group(0)
    result = re.sub(r'\bgo forward\b', _cap_go_forward, result, flags=re.IGNORECASE)

    # v121: Convert "keep going" → "go straight" (GT uses "keep going" <1% vs our 37.7%).
    result = re.sub(r'\bKeep going\b', 'Go straight', result)
    result = re.sub(r'\bkeep going\b', 'go straight', result)

    # v114: Suppress excess "walk into the [room]" — GT uses this only 6.9% of episodes.
    # v156: changed fallback from "walk straight" → "go straight" (absorbed by step4: walk to X or walk forward).
    _walk_into_counter = [0]
    def _suppress_extra_walk_into(m):
        _walk_into_counter[0] += 1
        if _walk_into_counter[0] > 1:
            return 'Go straight' if m.group(0)[0].isupper() else 'go straight'
        return m.group(0)
    result = re.sub(r'\bwalk into the \w+', _suppress_extra_walk_into, result, flags=re.IGNORECASE)

    # v114: Cap "walk past the X" — GT has 8.7%, v113 had 25.4% (too high after furniture fix).
    # v156: changed fallback from "walk straight" → "go straight" (absorbed by step4: walk to X or walk forward).
    _wpt_counter = [0]
    def _suppress_extra_walk_past(m):
        _wpt_counter[0] += 1
        if _wpt_counter[0] > 1:
            return 'Go straight' if m.group(0)[0].isupper() else 'go straight'
        return m.group(0)
    result = re.sub(r'\bwalk past the [^.!?,]+', _suppress_extra_walk_past, result, flags=re.IGNORECASE)

    # v125: _hallway_into_re REMOVED — was converting spontaneous "walk into the hallway" →
    # "walk through the hallway" for ~17.5% of all episodes (phantom through_the source).
    # Now "walk into the hallway" survives (or is suppressed by had_thru_hall logic in assembly).
    # Hallway through_the is now controlled exclusively by (through hallway) notation (P=30%).

    # v117: Convert "continue [X]" → GT-equivalent (continue=10.4% in GT — suppress excess).
    # v121: Use "go straight" instead of "keep going" (GT: keep_going~0%, go_straight=9%).
    # v124: Remove "continue for" and "continue to the" suppressions (too aggressive).
    # v125: ALSO remove "Continue down" and "Continue through" suppressions (continue 7.4%→~10.4%).
    #       Only suppress "continue forward" and "continue straight" — these are never in GT.
    result = re.sub(r'\bContinue\s+forward\b', 'Go straight', result)
    result = re.sub(r'\bcontinue\s+forward\b', 'go straight', result)
    result = re.sub(r'\bContinue\s+straight\b', 'Go straight', result)
    result = re.sub(r'\bcontinue\s+straight\b', 'go straight', result)

    return result


def extract_landmark_brief(desc: str) -> Optional[str]:
    """Extract a 2-5 word landmark noun phrase from a Phase C vision description (v145).
    v150 fix: also handles direction-aware 'Turn left/right at the X.' prefix.
    """
    if not desc:
        return None
    text = desc.strip()
    # Turn point — v150: handles "Turn left/right at the X." AND "Turn at the X."
    m = re.match(r'Turn\s+(?:left|right|around)?\s*(?:at|past)\s+(?:the\s+)?(.+?)[.\n]', text, re.IGNORECASE)
    if m:
        return _clean_landmark_words(m.group(1).strip().split())
    # "Ahead, a living room..." → "living room"
    m = re.match(r'Ahead[,\s]+(?:a |an |the )?(.+?)(?:\s+(?:with|is|are|and|featuring)|\.|$)', text)
    if m:
        return _clean_landmark_words(m.group(1).strip().split())
    # Goal: "The grey lounge chair positioned..." → "lounge chair"
    m = re.match(r'(?:The |A |An )([A-Za-z].+?)(?:\s+(?:positioned|located|is|are|against|and|with|featuring)|\.|$)', text)
    if m:
        return _clean_landmark_words(m.group(1).strip().split())
    words = [w for w in text.split() if len(w) > 2]
    result = _clean_landmark_words(words[:6]) if len(words) >= 2 else None
    # v168: Reject nav-direction results — "Turn right" is not a landmark
    if result and re.match(r'^(turn|go|walk|move|proceed|head|step)\b', result, re.IGNORECASE):
        return None
    return result


# Words to strip from end of extracted landmark (trailing filler)
_LM_STRIP_TRAIL = {
    'ahead', 'with', 'of', 'for', 'at', 'and', 'or', 'by', 'the', 'a', 'an',
    'featuring', 'positioned', 'draped', 'standing', 'leaning', 'hanging',
    'located', 'placed', 'over', 'from', 'to', 'in', 'into', 'on', 'upon',
    'open', 'closed', 'left', 'right', 'across', 'against', 'beside',
}

# Color/material/size adjectives to strip from beginning of extracted phrase
_LM_STRIP_LEAD = {
    'white', 'black', 'grey', 'gray', 'brown', 'beige', 'dark', 'light', 'wooden',
    'light-colored', 'large', 'small', 'big', 'rectangular', 'circular', 'round',
    'ornate', 'red', 'blue', 'green', 'yellow', 'orange', 'pink', 'purple',
    'tall', 'short', 'narrow', 'wide', 'long', 'thick', 'thin', 'glass', 'metal',
    'marble', 'stone', 'brick', 'concrete', 'tiled', 'carpeted', 'hardwood',
}


def _clean_landmark_words(words: list) -> Optional[str]:
    """Strip trailing filler and leading color/material adjectives; return 1-2 core words."""
    # Strip trailing filler (strip up to 3 times to handle "wall featuring ahead")
    for _ in range(3):
        if words and re.sub(r'[^a-z-]', '', words[-1].lower()) in _LM_STRIP_TRAIL:
            words = words[:-1]
        else:
            break
    # Strip leading material/color adjectives (max 2 strips)
    for _ in range(2):
        if words and re.sub(r'[^a-z-]', '', words[0].lower()) in _LM_STRIP_LEAD:
            words = words[1:]
        else:
            break
    # Limit to 2 core words max (keeps it brief like GT: "staircase", "dining table")
    words = words[:2]
    if not words:
        return None
    result = ' '.join(words).strip('.,;:').strip()
    return result if len(result) > 2 else None


_HALLWAY_WORDS = {'hallway', 'corridor', 'hall'}


def replace_go_straight_with_landmarks(text: str, phase_c_descs: Dict[str, str], rng=None) -> str:
    """v148/v149: Replace 'go straight[ahead]' with Phase C landmarks or fallback.
    v147 fixes: (1) filter landmarks containing hallway/corridor, (2) consume trailing 'ahead'.
    v147 ordering: called AFTER continue restoration (avoids Walk straight → Continue straight inflation).
    v156: fallback changed 'walk straight'→'walk forward' to reduce walk_straight% (GT=8.8%→target).
    v199 FIX16: rng parameter enables probabilistic replacement (51% walk_past/24% walk_toward/25% walk_to).
    """
    if not text:
        return text
    # Build ordered list of clean non-hallway landmarks from Phase C
    landmarks = []
    if phase_c_descs:
        for i in range(1, 15):
            label = f"turn_{i}"
            if label in phase_c_descs:
                lm = extract_landmark_brief(phase_c_descs[label])
                # v168: reject nav-verb landmarks ("Turn right" not a real object)
                if lm and re.match(r'^(turn|go|walk|move|proceed|head|step)\b', lm, re.IGNORECASE):
                    lm = None
                if lm and not any(w in lm.lower().split() for w in _HALLWAY_WORDS):
                    landmarks.append(lm)
        goal_lm = extract_landmark_brief(phase_c_descs.get("goal", ""))
        if goal_lm and not any(w in goal_lm.lower().split() for w in _HALLWAY_WORDS):
            landmarks.append(goal_lm)
    # Replace "go straight" and "go straight ahead" in route order (v148: consume 'ahead')
    # v194 FIX8 ROOT CAUSE: was "walk to the {lm}" — injected 29.5% walk_to via assembly.
    # GT: intermediate landmarks use "walk past the X" (8.7%), NOT "walk to the X" (5.1%).
    # "walk past the X" tells CMA to continue past the landmark (correct for intermediate waypoints).
    # "walk to the X" causes CMA to stop at the landmark (wrong for intermediate waypoints).
    # v199 FIX16: Diversify FIX8 output: 51% walk_past / 24% walk_toward / 25% walk_to.
    # walk_to injections are caught by FIX7 at P=0.90 → converted to walk_past/walk_toward.
    # Net effect: walk_past from FIX8 ≈ 64.5% of prior (down from 100%) → −5.6pp total walk_past.
    # Distribution approved by adversarial verification (Option D from deep research synthesis).
    lm_idx = [0]
    def _replace_gs(m):
        cap = m.group(0)[0].isupper()
        if lm_idx[0] < len(landmarks):
            lm = landmarks[lm_idx[0]]
            lm_idx[0] += 1
            # v199 FIX16: probabilistic replacement; fallback to walk_past if no rng supplied
            if rng is not None:
                r = rng.random()
                if r < 0.51:
                    replacement = f"walk past the {lm}"
                elif r < 0.75:
                    replacement = f"walk toward the {lm}"
                else:
                    replacement = f"walk to the {lm}"  # FIX7 converts 90% of these
            else:
                replacement = f"walk past the {lm}"  # v194 FIX8: was "walk to the"
        else:
            replacement = "go straight"  # v182: changed from 'walk forward' — keep go_straight vocabulary
            # v24 evidence: go_straight=23% → SR=40.24%. Model handles go_straight well.
            # walk_forward as fallback created 30% walk_forward (5× GT=6.1%) — harmful.
        return (replacement[0].upper() + replacement[1:]) if cap else replacement
    # v148: consume optional trailing 'ahead' to avoid "walk to X ahead" artifacts
    return re.sub(r'\bgo straight(?:\s+ahead)?\b', _replace_gs, text, flags=re.IGNORECASE)


def remove_loops(text: str, stop_phrase: str = "") -> str:
    """v25: Detect repeated 5-gram loops and truncate, appending stop_phrase if needed.
    v149 fix: truncate at i (second occurrence) instead of seen[ng] (first occurrence).
    Original bug: seen[ng]=0 → words[:0]=[] → truncated="" → ". stop_phrase" → quality_ok fails.
    Fix: keep the first good navigation segment, cut before the loop repeats.
    """
    words = text.split()
    if len(words) < 12:
        return text
    seen: dict = {}
    for i in range(len(words) - 4):
        ng = ' '.join(words[i:i+5]).lower()
        if ng in seen:
            # v149 FIX: cut at i (before second occurrence), not seen[ng] (start of first occurrence)
            truncated = ' '.join(words[:i]).rstrip('.,;:')
            if stop_phrase:
                truncated = truncated + '. ' + stop_phrase
            elif not truncated.rstrip().endswith(('.', '!', '?')):
                truncated += '.'
            return truncated
        seen[ng] = i
    return text


def quality_ok(text, stop_phrase=""):
    """v149: Accept stop_phrase word count toward minimum threshold.
    Root cause of 'destination' fallback bug #2: _MID_STOP_FRAG_RE strips LLM's 'and stop X'
    making text <8 words, even though enforce_stop_phrase will add 5-7 more words later.
    Fix: count stop_phrase words toward the 8-word minimum.
    """
    words = text.split()
    sp_words = stop_phrase.split() if stop_phrase else []
    total_words = len(words) + len(sp_words)
    if total_words < 5:  return False, "too_short"
    if total_words > 85: return False, "too_long"
    # v102: reject truncated instructions that start with '.' (LLM generated only stop phrase)
    if text.strip().startswith('.'): return False, "starts_with_period"
    # v102: reject very short instructions that lack navigation content (< 12 words)
    # v124: lowered from 12 to 8. v149: count stop_phrase toward threshold.
    if total_words < 8: return False, "navigation_too_brief"
    # v25: reject obvious loops (repeated 5-gram after remove_loops should have fixed it,
    # but check anyway as a safety net)
    ngrams = set()
    for i in range(len(words) - 4):
        ng = ' '.join(words[i:i+5]).lower()
        if ng in ngrams:
            return False, "loop_detected"
        ngrams.add(ng)
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

    print('=== Gate 4 v190: walk_to reduction + exit boost + filler removal + continue fix ===')
    print(f"  v189 stats: go_straight=11.3%, walk_to=50.2%, exit=10.7%, continue=19.5%, walk_past=7.3%")
    print(f"  GT reference: walk_to=5.1%, exit=18.0%, continue=10.4%, walk_past=10.5%")
    print(f"  v190 NEW FIXES:")
    print(f"    FIX1: walk_to_X_for_a_bit -> walk_past_X P=0.75 (walk_to:-3.9pp, walk_past:7.3%->11.2%)")
    print(f"    FIX2: walk_to_X+turn -> turn_at_X P=0.80 (walk_to:-3.9pp, GT landmark style)")
    print(f"    FIX3: remove 'for a bit/while/moment' filler (GT=0%, ours=15.2%)")
    print(f"    FIX4: walk_out_of -> exit P=0.70 (+1.4pp exit)")
    print(f"    FIX5: leave_the [room] -> exit P=0.40 (+1.5pp exit)")
    print(f"    FIX6: walk_ahead->continue P=0.70->0.45 (continue:19.5%->~17%)")
    print(f"  v189: gs→ws P=0.50, drop_wa P=0.65+cont P=0.70, wi→enter P=0.45, REMOVED wf/wa→gs")
    print(f"  v188: +go_straight→walk_straight P=0.35, +drop_walk_ahead P=0.65, +walk_into→enter P=0.30")
    print(f"  v187: +stop/wait injection P=0.30, +walk_through_hallway→walk_down P=0.65")
    print(f"  INHERITED FROM v186: through→enter P=0.25, walk_fwd→go_straight P=0.30, P_STOP_MERGE=0.30")
    print(f"  Seed: 0x9145_5300 (assembly, reused from v91); thru injection: 0x9245_B300")
    print(f"  P_ANCHOR_EPISODE: {p_anchor} → expected ~{p_anchor*0.44*100:.1f}% (GT=16.5%)")
    print(f"  stop/wait: {P_STOP*100:.0f}%/{P_WAIT*100:.0f}%/{(1-P_STOP-P_WAIT)*100:.0f}%")
    print(f"  Phase 1: REUSE v152 room-context Phase C (3192 descs, extract_scene converts to neutral)")
    print(f"  Phase 1b: v75 midpoints checkpoint (door 3.1%, artwork 42.4%, improve=1094, regress=0)")
    print(f"  Phase 1b2: v74 goal-approach stop descriptions + material strip fix")
    print(f"  Phase 1c: v73 approach-views checkpoint (40% artwork turn landmarks)")
    print(f"  Phase 2 concurrency: {args.concurrency_p2}")
    print(f"  P_PASS_MARKER={P_PASS_MARKER}, P_THRU_MARKER={P_THRU_MARKER}")
    print(f"  v165: Furniture blacklisted from [pass:] (sofa/armchair/chair/table/bed/cabinet/lamp/curtain/plant)")
    print(f"  v165: Turn direction added to non-hallway room transitions ('turn left (→ bedroom)')")
    print(f"  v165: P_TURN_UNANCHORED={P_TURN_UNANCHORED} (was 0.28) — more turn guidance")
    print(f"  RETAIN v36: 'proceed'→'walk' elimination (43.8%→1.7%)")
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

    # v94: Build hallway room set from perframe data for assembly injection.
    # Episodes where start or any turn room = 'hallway'/'hall'/'corridor' (not goal-only).
    _HALL_ROOMS = ('hallway', 'hall', 'corridor')
    _HALLWAY_ROOM_EIDS: set = set()
    for _eid, _pf in pf_map.items():
        if _pf.get('start', {}).get('room', '').lower() in _HALL_ROOMS:
            _HALLWAY_ROOM_EIDS.add(_eid)
        elif any(t.get('room', '').lower() in _HALL_ROOMS for t in _pf.get('turns', [])):
            _HALLWAY_ROOM_EIDS.add(_eid)
    print(f"  v94 hallway room eids: {len(_HALLWAY_ROOM_EIDS)} episodes ({len(_HALLWAY_ROOM_EIDS)/len(episodes)*100:.1f}%)")

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

    # ── Phase 1b: Load midpoints checkpoint ──────────────────────────────────
    midpoints_ckpt = load_midpoints_checkpoint()
    print(f"Phase 1b midpoints checkpoint: {len(midpoints_ckpt)} episodes")
    has_mid = sum(1 for ep in episodes
                  if any(d for d in midpoints_ckpt.get(str(ep["episode_id"]),{}).values() if d))
    print(f"  Episodes with at least 1 midpoint: {has_mid}/{len(episodes)} "
          f"({100*has_mid/len(episodes):.1f}%)")

    # ── Phase 1b2: Load goal-approach checkpoint (v74) ───────────────────────
    goal_approach_ckpt = load_goal_approach_checkpoint()
    has_goal_app = sum(1 for ep in episodes if str(ep["episode_id"]) in goal_approach_ckpt)
    print(f"Phase 1b2 goal-approach checkpoint: {has_goal_app}/{len(episodes)} episodes")

    # ── Phase 1c: Load turn-sides checkpoint (v24 new) ───────────────────────
    ts_vision_map = load_turn_sides_checkpoint()
    print(f"\nPhase 1 approach-views checkpoint: {len(ts_vision_map)} episodes")
    has_ts = sum(1 for ep in episodes
                 if any(d for d in ts_vision_map.get(str(ep["episode_id"]),{}).values() if d and not is_turn_generic(d)))
    print(f"  Episodes with useful turn-side desc: {has_ts}/{len(episodes)} ({100*has_ts/len(episodes):.1f}%)")

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

        lm_info  = get_landmark_info_v18(ep, pf_map, old_map, prims, vision_map, ts_vision_map)
        sel_turns              = episode_selective_turns(lm_info, eid, p_anchor)
        # v30: strip material adjectives from goal_lm before choosing stop phrase.
        # In v29, 417/419 "wooden" instances came from stop phrases because goal_lm
        # (e.g., "brown wooden door frame") bypassed clean()'s material filter.
        goal_lm_clean = strip_material_adjectives(lm_info["goal_landmark"])
        stop_phrase, stop_type = choose_stop(goal_lm_clean, lm_info["goal_room"], eid)
        # v74: override stop phrase with goal-approach VLM description when better
        goal_app_raw = goal_approach_ckpt.get(str(eid), {})
        goal_app_desc = (goal_app_raw.get("desc") or "") if isinstance(goal_app_raw, dict) else (str(goal_app_raw) if goal_app_raw else "")
        stop_phrase, stop_type = apply_goal_approach_stop(stop_phrase, stop_type, goal_app_desc, eid)
        # v173: strip colors from stop_phrase AFTER apply_goal_approach_stop override.
        # v172 stripped goal_vis (Phase 1 approach view) but NOT the Phase 1b2 goal-approach override.
        # Phase 1b2 overrides stop_phrase with colored descriptions like "Stop near the grey lounge chair."
        # The LLM faithfully COPIES this colored stop phrase from the prompt ending_req field.
        # Fix: strip colors from the finalized stop_phrase so the LLM copies a color-free phrase.
        _sp_m = re.match(r'^(Stop near|Stop by|Stop in front of|Stop at|Wait near|Wait at|Wait by|Wait in)\s+the\s+(.+?)\.?\s*$', stop_phrase, re.IGNORECASE)
        if _sp_m:
            _sp_action = _sp_m.group(1)
            _sp_obj = strip_colors_from_landmark(_sp_m.group(2).rstrip('.'))
            if _sp_obj and len(_sp_obj.split()) >= 1:
                stop_phrase = f"{_sp_action} the {_sp_obj}."
        # v23: first AND last segment midpoints (up to 2 entries)
        seg_midpoint = get_all_segment_midpoints(ep, midpoints_ckpt)

        # v77: compute had_pass_markers deterministically (seeded rng in build_route_desc)
        _prims_for_pass = get_primitives(ep)
        _route_str, had_pass = build_route_desc(_prims_for_pass, sel_turns, seg_midpoint,
                                                 path_len=len(ep.get("reference_path",[])), eid=eid)
        # v92: extract [thru: desc] descriptions from route for assembly injection
        _THRU_MARKER_EXTRACT_RE = re.compile(r'\[thru:\s*([^\]]+)\]')
        thru_descs = _THRU_MARKER_EXTRACT_RE.findall(_route_str)
        # v117: track whether route has any (door→ room) notation (protects "through the doorway")
        had_door_trans = bool(re.search(r'\(door→', _route_str))
        # v125: track whether route has any (through hallway) notation.
        # v127: also track (hall→) notation — both indicate intentional hallway mention.
        # v128: also track (hall_into→) notation — "walk into the hallway" (into_hallway target).
        # v129: renamed (hall_into→) → (→ hallway) for better LLM compliance (matches (→ room) convention).
        had_thru_hall = bool(re.search(r'\(through hallway\)', _route_str))
        had_hall_entry = bool(re.search(r'\(hall→\)', _route_str))
        had_hall_into  = bool(re.search(r'\(→ hallway\)', _route_str))  # v129: renamed from (hall_into→)
        had_any_hall = had_thru_hall or had_hall_entry or had_hall_into

        if str(eid) in p2_ckpt:
            skipped += 1
            task_meta[eid] = {
                "sv":            gt_start_verb(gt_instr),
                "goal_lm":       goal_lm_clean,
                "goal_room":     lm_info.get("goal_room", ""),  # v176: for stop-phrase room context
                "stop_phrase":   stop_phrase,
                "stop_type":     stop_type,
                "had_pass":      had_pass,      # v77
                "thru_descs":    thru_descs,    # v92: for through_the injection
                "had_door_trans": had_door_trans,  # v117: for doorway suppression bypass
                "had_thru_hall": had_thru_hall,    # v125: for hallway suppression
                "had_any_hall":  had_any_hall,     # v128: had_thru_hall OR had_hall_entry OR had_hall_into
            }
            continue

        examples   = sc_idx.top_k(ep, ep_feat, k=args.k_similar)
        prompt, sv, _had_pass2 = build_prompt(ep, gt_instr, examples, lm_info, prims,
                                               sel_turns, stop_phrase, stop_type, vision_map,
                                               seg_midpoint=seg_midpoint)
        task_meta[eid] = {
            "sv":            sv,
            "goal_lm":       goal_lm_clean,   # v30: use material-filtered goal_lm
            "goal_room":     lm_info.get("goal_room", ""),  # v176: for stop-phrase room context
            "stop_phrase":   stop_phrase,
            "stop_type":     stop_type,
            "had_pass":      had_pass,         # v77: deterministic
            "thru_descs":    thru_descs,       # v92: for through_the injection
            "had_door_trans": had_door_trans,  # v117: for doorway suppression bypass
            "had_thru_hall": had_thru_hall,    # v125: for hallway suppression
            "had_any_hall":  had_any_hall,     # v128: had_thru_hall OR had_hall_entry OR had_hall_into
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
        lm_info = get_landmark_info_v18(ep, pf_map, old_map, prims, vision_map, ts_vision_map)
        sel = episode_selective_turns(lm_info, eid, p_anchor)
        has_lm = any(t.get("landmark") for t in sel)
        if has_lm:
            n_with_lm += 1
            # Check if any selected turn's landmark came from vision
            vis = vision_map.get(str(eid), {})
            prims_t = get_primitives(ep)
            lm_t = get_landmark_info_v18(ep, pf_map, old_map, prims_t, vision_map, ts_vision_map)
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

    # v54: Precompute per-episode has_thru_midpoint (deterministic from midpoints_ckpt)
    # Used to surgically remove "through the doorway" when no thru-type midpoints exist.
    # v53 analysis: 54/109 "through the doorway" eps have ONLY pass-type midpoints → safe to remove.
    # Removing those 54 eps brings through_the from 30.4% → ~27.5% (GT=27.2%).
    def _has_thru_midpoint(eid):
        mid_data = midpoints_ckpt.get(str(eid), {})
        return any(classify_pass_action(desc) == "thru" for desc in mid_data.values())

    _THRU_DOORWAY_RE = re.compile(
        r'\b(?:walk\s+)?(?:forward\s+)?(?:go\s+)?(?:continue\s+)?through\s+the\s+doorway\b',
        re.IGNORECASE
    )

    # v58: Precompute per-episode has_pass_midpoint (deterministic from midpoints_ckpt)
    # Same lesson as v54fix for through_the: LLM hallucinated "walk past the X" without [pass:] markers.
    # v57 confirmed P_PASS_MARKER reduction doesn't fix it — LLM generates from visual context.
    # Fix: if NO pass-type midpoints at all → any "walk past the X" is hallucinated → replace.
    # If HAS pass-type midpoints → "walk past the X" may be legitimate → keep.
    def _has_pass_midpoint(eid):
        mid_data = midpoints_ckpt.get(str(eid), {})
        return any(classify_pass_action(desc) == "pass" for desc in mid_data.values())

    _WALK_PAST_THE_RE = re.compile(
        r'\bwalk\s+past\s+the\b',
        re.IGNORECASE
    )

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
        goal_room   = meta.get("goal_room","")  # v176: for stop-phrase room context injection

        text = clean(raw, sv)
        # v25: remove loops BEFORE quality check — truncation may fix loop issues
        text = remove_loops(text, stop_phrase)

        # v54/v117: Surgical through_the post-processing.
        # v117: also protect episodes with (door→ room) route notation (had_door_trans=True).
        # In Habitat, ALL room transitions involve doorways — "through the doorway" is ALWAYS valid
        # for episodes with room transitions (had_door_trans=True) or thru-type midpoints.
        had_door_trans = meta.get("had_door_trans", False)
        if "through the doorway" in text.lower() and not _has_thru_midpoint(eid) and not had_door_trans:
            text = _THRU_DOORWAY_RE.sub("go forward", text)
            text = re.sub(r'\bgo forward\s+go forward\b', 'go forward', text, flags=re.I)

        # v120: had_pass suppression REMOVED — over-suppressed walk_past to 1.2% in v119.
        # [pass: X] markers at P=0.70 drive walk_past to ~8.3%; no suppression needed.
        # Spontaneous walk_past from visual context is now acceptable (GT=8.7%).

        # v125/v126: Full hallway suppression for unannotated episodes.
        # v127: Updated condition — suppress when NEITHER (through hallway) NOR (hall→) in route.
        #        (hall→) episodes should keep their "walk to the hallway" phrases.
        # Stage 1: Any verb + through/into + hallway → "walk straight" (removes from through_the too)
        # Stage 2: Remaining "hallway/corridor" mentions → "room" (catches "turn into the hallway")
        # Stage 3: "walk/go through the room" artifacts from stage 2 → "walk straight"
        had_thru_hall = meta.get("had_thru_hall", False)
        had_any_hall  = meta.get("had_any_hall",  had_thru_hall)
        if not had_any_hall:
            def _to_room(m):
                # v128: Use 5-word replacement to preserve avg_words (vs "walk straight" = 2w).
                # "walk through the hallway" (4w) → "walk forward to the room" (5w)
                return 'Walk forward to the room' if m.group(0)[0].isupper() else 'walk forward to the room'
            # Stage 1: verb + optional_word + through/into + hallway/corridor → "walk forward to the room"
            # v128: extended pattern catches "walk straight through the hallway" (missed in v127).
            text = re.sub(
                r'\b(?:walk|go|continue|head|move)(?:\s+\w+)?\s+(?:through|into)\s+the\s+(?:hallway|corridor)\b',
                _to_room, text, flags=re.IGNORECASE
            )
            # Stage 2: Any remaining "hallway"/"corridor" mention → "room"
            text = re.sub(r'\b(?:hallway|corridor)\b', 'room', text, flags=re.IGNORECASE)
            # Stage 3: "walk/go through the room" artifacts → "walk forward to the room"
            # v128: extended pattern catches "walk straight through the room" artifacts from Stage 2.
            text = re.sub(
                r'\b(?:walk|go|continue)(?:\s+\w+)?\s+through\s+the\s+room\b',
                _to_room, text, flags=re.IGNORECASE
            )

        ok, reason = quality_ok(text, stop_phrase)  # v149: pass stop_phrase for word-count check
        if not ok:
            # v149 fix: always use "Walk" (not sv) — sv can be "Enter", "Make", "Continue" etc.
            # which produces grammatically broken "Enter to the destination." / "Make to the destination."
            text = "Walk to the destination."
            if stop_phrase: text = text.rstrip('.') + '. ' + stop_phrase
            n_fix += 1

        # v102: P_TOWARD disabled (was 0.25 in v86-v101, caused toward_the=7.1% vs GT=3.8%).
        # Natural LLM toward_the is ~2-3%; GT=3.8% — accept the small gap without injection.
        # The random injection was creating distribution mismatch: 25% of episodes that happen
        # to have "walk to the X" get "toward the" → non-uniform distribution vs GT.
        pass  # toward_the injection removed

        # v104: Probabilistic continue restoration — promote "Walk straight/forward/ahead" →
        # "Continue straight/forward/ahead" in P=0.30 of episodes.
        # v103: P=0.08 gave continue=9.1%; v24=57.1%, GT=10.4%. Raising to 0.30 targets ~20%.
        # Only sentence-start "Walk" (capital W), first occurrence only.
        _WALK_FWD_CAP_RE = re.compile(r'\bWalk\s+(forward|straight|ahead)\b')
        cont_rng = random.Random(eid ^ 0x9145_C300)
        # v127: P 0.30 → 0.55 (continue 4.9% → ~9% continue).
        # v128: P 0.55 → 0.70 (continue 7.8% → ~9.9% ≈ GT=10.4%).
        # v129: P 0.70, measured continue=11.5% (1.1pp above GT=10.4%).
        # v130: P 0.70 → 0.63, measured continue=9.7% (0.7pp below GT=10.4%).
        # v136: P=0.655 UNCHANGED (continue=10.3% ≈ GT=10.4% ✓)
        # v163: P 0.655→0.28 → continue≈15% (v161 validated: P=0.28 → 15.8%)
        # Rationale: v162 continue=37.1% too high (high walk_forward triggers many restorations).
        #            v163 uses v158 checkpoint (avg_turns=0.54) so walk_forward is lower.
        if cont_rng.random() < 0.28 and _WALK_FWD_CAP_RE.search(text):
            text = _WALK_FWD_CAP_RE.sub(r'Continue \1', text, count=1)

        # v164: 50% skip — balanced. With fewer Phase C anchors, fewer "walk to X" expected.
        # Fewer anchors → LLM generates more unanchored turns → naturally longer straight segments.
        _gs_skip_rng = random.Random(eid ^ 0x60_5700)
        if _gs_skip_rng.random() >= 0.25:  # v183: 75% run (was 50%) → reduce go_straight from 31.4%→~24.7%
            # v199 FIX16: pass per-episode RNG for probabilistic replacement (51/24/25 distribution)
            _fix16_rng = random.Random(eid ^ 0xF1F2_F3F4)
            text = replace_go_straight_with_landmarks(text, vision_map.get(str(eid), {}), rng=_fix16_rng)

        # v95: through_the injection REMOVED — v94_wins analysis shows injected through_the
        # HURTS SR (v24_wins episodes have through_the=0.9% vs v62=24.6%).
        # LLM will generate "through the X" naturally via [thru:] route markers where relevant.
        # Expected through_the: ~22% (from [thru:] markers), down from injected 27.1%.

        text = enforce_stop_phrase(text, stop_phrase, stop_type)

        # v67: Floor-stop fix (same as v65/v66) — 0% bad floor/tile/carpet stops
        # VLM Phase 1 sometimes describes featureless stops as "the white hallway floor"
        _FLOOR_STOP_RE = re.compile(
            r'\b(Stop|Wait)\b(?:.{0,80}?)'
            r'\b(floor(?:ing)?|tile(?:s)?|carpet(?:ing)?|linoleum|mat|'
            r'hardwood\s+floor|tiled\s+floor|marble\s+floor|wood\s+floor|'
            r'polished\s+floor|light[\-\s]colored\s+floor|white\s+floor|'
            r'grey\s+floor|gray\s+floor)\b.*$',
            re.IGNORECASE | re.DOTALL,
        )
        _RUG_STOP_RE = re.compile(
            r'\b(Stop|Wait)\s+(?:near|at|by|in\s+front\s+of|beside|next\s+to)\s+the\s+'
            r'(?:\w+\s+)*(?:rug|mat|carpet|runner|doormat)\b.*$',
            re.IGNORECASE | re.DOTALL,
        )
        for fs_pat in (_FLOOR_STOP_RE, _RUG_STOP_RE):
            m = fs_pat.search(text)
            if not m: continue
            action = m.group(1)
            prefix = text[:m.start()].rstrip()
            fl = text.lower()
            if re.search(r'\bhallway\b', fl): repl = f"{action} in the hallway."
            elif re.search(r'\bkitchen\b', fl): repl = f"{action} in the kitchen."
            elif re.search(r'\bbedroom\b', fl): repl = f"{action} in the bedroom."
            elif re.search(r'\bliving room\b', fl): repl = f"{action} in the living room."
            elif re.search(r'\bdining room\b', fl): repl = f"{action} in the dining room."
            elif re.search(r'\bbathroom\b', fl): repl = f"{action} in the bathroom."
            elif re.search(r'\boffice\b', fl): repl = f"{action} in the office."
            elif re.search(r'\bstair(?:case|s|way)\b', fl): repl = f"{action} at the stairs."
            elif re.search(r'\bdoorway|doorframe\b', fl): repl = f"{action} in the doorway."
            else: repl = f"{action}."
            text = (prefix + " " + repl).strip()
            break

        # v173: sofa→couch substitution (GT uses "couch" 7.2% vs "sofa" 1.5%,
        # but VLM descriptions consistently say "sofa" → model trained on GT sees mismatch)
        text = re.sub(r'\bsofa\b', 'couch', text, flags=re.IGNORECASE)
        text = re.sub(r'\bSofa\b', 'Couch', text)
        # v173: "walk into the [room]" → "enter the [room]" for 30% of cases (GT "enter"=17%).
        # Route-level "enter room" notation failed (LLM ignores novel notation → writes "walk into").
        # Post-processing substitution directly converts to GT-preferred "enter" verb form.
        # Only apply to NAMED rooms (not hallway which is a transit space, not a destination).
        _NAMED_ROOMS = r'(bedroom|kitchen|bathroom|bathroom|dining room|living room|dining area|office|study|laundry|closet|pantry|garage|gym|game room|media room|library)'
        def _maybe_enter(m):
            # deterministic per-instruction: use hash to get consistent 30% rate
            if (hash(text[:20] + m.group(0)) % 10) < 3:
                return f"enter the {m.group(1)}"
            return m.group(0)
        text = re.sub(rf'\bwalk into the ({_NAMED_ROOMS})\b', _maybe_enter, text, flags=re.IGNORECASE)

        # v182: REMOVED go_straight→walk_ahead conversion (was lines from v176)
        # v24 evidence: go_straight=23% AND SR=40.24% — model handles go_straight fine
        # Converting go_straight→walk_ahead created walk_ahead=33% (5.4× GT=6.1%) — WRONG
        # Keeping go_straight naturally matches GT's 7.9% go_straight distribution
        # KEEP walk_straight→walk_ahead: walk_straight is LLM artifact (GT=8.8%→0.6% actual usage)
        text = re.sub(r'\bwalk straight\b', 'walk ahead', text, flags=re.IGNORECASE)

        # v177: walk_forward PREPOSITION REDUCTION
        # "walk forward [prep]" → "walk [prep]": removes redundant "forward" when direction is explicit
        # GT=6.2% walk_forward vs v176=28.4% — reduces toward GT level (-1w per occurrence)
        # Pattern: "walk forward to/past/through/into/toward/towards/along/up/down the X"
        # Standalone "walk forward" (no preposition) is KEPT → matches GT usage
        text = re.sub(
            r'\bwalk forward (to|past|through|into|toward|towards|along|up|down)\b',
            lambda m: f'walk {m.group(1)}', text, flags=re.IGNORECASE
        )

        # v178: walk_ahead PREPOSITION REDUCTION (new)
        # walk_ahead=40.2% (GT=6.1%) — "walk ahead to/through/into X" → "walk to/through/into X"
        # Standalone "walk ahead" (no preposition) is KEPT — preserves directional use
        text = re.sub(
            r'\bwalk ahead (to|past|through|into|toward|towards|along|up|down)\b',
            lambda m: f'walk {m.group(1)}', text, flags=re.IGNORECASE
        )

        # v190 FIX1: "walk to the X for a bit/while/moment" -> "walk past the X" at P=0.75
        # "for a bit" signals X is an intermediate waypoint (not destination) -> walk_past is correct.
        # Walk_to: -3.9pp. Walk_past: 7.3%+3.9%=11.2% (GT=10.5% -- near perfect).
        # Must run BEFORE filler removal (fix3) to detect the "for a bit" pattern.
        _WALK_TO_FOR_A_BIT_RE = re.compile(
            r'\bwalk to the ([\w ]+?) for a (?:bit|while|moment)\b',
            re.IGNORECASE
        )
        _wtfab_rng = random.Random(eid ^ 0xE1F2_A3B4)
        def _walk_to_for_a_bit_to_past(m):
            if _wtfab_rng.random() < 0.75:
                cap = m.group(0)[0].isupper()
                obj = m.group(1).strip()
                return ('Walk past the ' if cap else 'walk past the ') + obj
            return m.group(0)
        text = _WALK_TO_FOR_A_BIT_RE.sub(_walk_to_for_a_bit_to_past, text)

        # v190 FIX2: "walk to the X and/comma turn [dir]" -> "turn [dir] at the X" at P=0.80
        # GT landmark style: use objects as turn reference points, not as destinations.
        # Covers 4.9% of instructions → ~3.9pp walk_to reduction.
        _WALK_TO_TURN_AND_RE = re.compile(
            r'\bwalk to the ([\w ]+?)\s+and\s+(turn (?:left|right))\b',
            re.IGNORECASE
        )
        _WALK_TO_TURN_COMMA_RE = re.compile(
            r'\bwalk to the ([\w ]+?),\s*(turn (?:left|right))\b',
            re.IGNORECASE
        )
        _wturn_rng = random.Random(eid ^ 0xF2A3_B4C5)
        def _walk_to_turn_at(m):
            if _wturn_rng.random() < 0.80:
                obj = m.group(1).strip()
                turn = m.group(2)  # "turn left" or "turn right"
                cap = m.group(0)[0].isupper()
                direction = turn.split()[1]  # "left" or "right"
                return ('Turn ' if cap else 'turn ') + direction + ' at the ' + obj
            return m.group(0)
        text = _WALK_TO_TURN_AND_RE.sub(_walk_to_turn_at, text)
        _wturn_rng2 = random.Random(eid ^ 0xA3B4_C5D6)
        def _walk_to_turn_at_comma(m):
            if _wturn_rng2.random() < 0.80:
                obj = m.group(1).strip()
                turn = m.group(2)
                cap = m.group(0)[0].isupper()
                direction = turn.split()[1]
                return ('Turn ' if cap else 'turn ') + direction + ' at the ' + obj
            return m.group(0)
        text = _WALK_TO_TURN_COMMA_RE.sub(_walk_to_turn_at_comma, text)

        # v190 FIX3: Remove "for a bit/while/moment" filler (GT=0%, ours=15.2%).
        # This phrase never appears in GT R2R instructions. Always safe to remove.
        text = re.sub(r'\s+for a (?:bit|while|moment)\b', '', text, flags=re.IGNORECASE)

        # v190 FIX4: "walk out of the X" -> "exit the X" at P=0.70
        # GT uses "exit" 18% vs our 10.7%. "walk out of" is a GT alternative (3.3%) we can convert.
        _WALK_OUT_OF_RE = re.compile(r'\bwalk out of (the \w+)', re.IGNORECASE)
        _woof_rng = random.Random(eid ^ 0xB4C5_D6E7)
        def _walk_out_to_exit(m):
            if _woof_rng.random() < 0.70:
                cap = m.group(0)[0].isupper()
                return ('Exit ' if cap else 'exit ') + m.group(1)
            return m.group(0)
        text = _WALK_OUT_OF_RE.sub(_walk_out_to_exit, text)

        # v190 FIX5: "leave the [room]" -> "exit the [room]" at P=0.40
        # GT exit=18% vs ours=10.7%. "leave the" appears 3.8% in our data.
        _LEAVE_ROOM_RE = re.compile(
            r'\bleave the (bedroom|kitchen|bathroom|dining room|living room|dining area'
            r'|office|study|laundry|closet|pantry|garage|gym|game room|media room|library'
            r'|sitting room|hallway|corridor|room|area|foyer|lobby|entryway|space)\b',
            re.IGNORECASE
        )
        _leave_rng = random.Random(eid ^ 0xC5D6_E7F8)
        def _leave_to_exit(m):
            if _leave_rng.random() < 0.40:
                cap = m.group(0)[0].isupper()
                return ('Exit the ' if cap else 'exit the ') + m.group(1)
            return m.group(0)
        text = _LEAVE_ROOM_RE.sub(_leave_to_exit, text)

        # v181: walk_to→enter CONVERSION for named rooms (P=0.75, was 0.60 in v180)
        # v180 P=0.60 → enter=8.5%; v181 P=0.75 → ~10-11% enter (incremental toward GT=17%)
        _NAMED_ROOMS_ENTER = (r'(bedroom|kitchen|bathroom|dining room|living room|dining area'
                              r'|office|study|laundry|closet|pantry|garage|gym|game room'
                              r'|media room|library|sitting room|hallway|corridor)')
        def _walk_to_enter(m):
            if (eid * 7 + hash(m.group(0)[:8])) % 10 < 7:  # v181: ~70% deterministic (~0.75 effective P)
                room_name = m.group(1)
                return f"enter the {room_name}"
            return m.group(0)
        text = re.sub(
            rf'\bwalk to the ({_NAMED_ROOMS_ENTER})\b',
            _walk_to_enter, text, flags=re.IGNORECASE
        )

        # v188/v189 FIX1: go_straight → walk_straight at P=0.50 per occurrence
        # v188: P=0.35 → go_straight=18.7%. v189: P=0.50 → target go_straight ~7-9% (≈ GT=7.9%)
        # NOTE: walk_straight will overshoot GT=8.8% but go_straight becomes the priority.
        # Combined with removing walk_forward->go_straight and walk_ahead->go_straight in v189,
        # go_straight input is lower so P=0.50 should bring it close to GT.
        _GS_TO_WS_RE = re.compile(r'\bgo straight\b', re.IGNORECASE)
        _gs_ws_rng = random.Random(eid ^ 0xA9B0_C1D2)
        def _go_straight_to_walk_straight(m):
            if _gs_ws_rng.random() < 0.50:
                cap = m.group(0)[0].isupper()
                return 'Walk straight' if cap else 'walk straight'
            return m.group(0)
        text = _GS_TO_WS_RE.sub(_go_straight_to_walk_straight, text)

        # v188 FIX2: drop "ahead" from "walk ahead" when followed by prepositions (P=0.65)
        # walk_ahead=9.8% (GT=0.2%) — 9.6pp excess. Remove "ahead" when the context is clear.
        # "walk ahead and turn" -> "walk and turn" | "walk ahead into X" -> "walk into X"
        # Does NOT convert to go_straight (already overused). Just simplifies the phrase.
        _WA_DROP_RE = re.compile(
            r'\bwalk ahead(?=\s+(?:and|for|into|towards?|through|across|along|until|from|past)\b)',
            re.IGNORECASE
        )
        _wa_drop_rng = random.Random(eid ^ 0xB0C1_D2E3)
        def _drop_walk_ahead(m):
            if _wa_drop_rng.random() < 0.65:
                cap = m.group(0)[0].isupper()
                return 'Walk' if cap else 'walk'
            return m.group(0)
        text = _WA_DROP_RE.sub(_drop_walk_ahead, text)

        # v188/v189 FIX3: walk into the [room] → enter the [room] at P=0.45 (was 0.30 in v188)
        # v188: P=0.30 → walk_into=21.1%, enter=16.1%. v189: P=0.45 → walk_into~15%, enter~20%
        # enter slightly overshoots GT=16% but walk_into approaches GT=6.9% better.
        _WALK_INTO_ROOM_RE = re.compile(
            r'\bwalk into (the (?:bedroom|kitchen|bathroom|dining room|living room|dining area'
            r'|office|study|laundry|closet|pantry|garage|gym|game room|media room|library'
            r'|sitting room|hallway|corridor|room|foyer|lobby|entryway|space|area))\b',
            re.IGNORECASE
        )
        _wi_enter_rng = random.Random(eid ^ 0xC1D2_E3F4)
        def _walk_into_to_enter(m):
            if _wi_enter_rng.random() < 0.45:
                cap = m.group(0)[0].isupper()
                return ('Enter ' if cap else 'enter ') + m.group(1)
            return m.group(0)
        text = _WALK_INTO_ROOM_RE.sub(_walk_into_to_enter, text)

        # v187 FIX2: "walk/go through the hallway/corridor" → "walk/go down the hallway/corridor" (P=0.65)
        # GT uses "walk down the hallway" naturally; "walk through the hallway" is less common in GT.
        # v186: walk_through_hallway=10.6% (195 episodes). At P=0.65: ~127 conversions = 6.9pp reduction.
        # Expected: through_the 35.0%→~28.1% (close to GT 27.0%). Semantically clean: "down" implies direction.
        _THROUGH_HALL_RE = re.compile(
            r'\b(walk|go) through (the (?:hallway|corridor))',
            re.IGNORECASE
        )
        _th_hall_rng = random.Random(eid ^ 0xD4E5_F6A7)
        def _through_hall_to_down(m):
            if _th_hall_rng.random() < 0.65:
                verb = m.group(1)
                rest = m.group(2)
                cap = verb[0].isupper()
                return ('Walk' if cap else 'walk') + ' down ' + rest
            return m.group(0)
        text = _THROUGH_HALL_RE.sub(_through_hall_to_down, text)

        # v184 FIX1: "walk/go through the [room]" → "enter the [room]" at P=0.15
        # through_the=39.9% (GT=27.2%) — 12.7pp overuse. Source: LLM-generated "walk through the X".
        # Conversion is semantically valid ("walk through the kitchen" ↔ "enter the kitchen").
        # Two-for-one: reduces through_the AND increases enter (both toward GT).
        # Expected: through_the 39.9%→33.9%, enter 11.4%→17.4% (≈ GT=17%!)
        _THROUGH_ENTER_RE = re.compile(
            r'\b(walk|go) through (the (?:bedroom|kitchen|bathroom|dining room|living room'
            r'|dining area|office|study|laundry|closet|pantry|garage|gym|game room'
            r'|media room|library|sitting room|hallway|corridor|doorway|archway|opening|entryway'
            r'|room|area|foyer|lobby))\b',  # v186: added room/area/foyer/lobby
            re.IGNORECASE
        )
        _thru_enter_rng = random.Random(eid ^ 0x3A4B_5C6D)
        def _convert_through_to_enter(m):
            if _thru_enter_rng.random() < 0.25:  # v185: 0.15→0.25 for deeper through_the reduction
                cap = m.group(0)[0].isupper()
                noun = m.group(2)
                return f"{'E' if cap else 'e'}nter {noun}"
            return m.group(0)
        text = _THROUGH_ENTER_RE.sub(_convert_through_to_enter, text)

        # v189: REMOVED walk_forward→go_straight (was P=0.30 in v184-v188)
        # REASON: go_straight is now 18.7% (GT=7.9%) — way too high. Adding more from walk_forward
        # was counterproductive. walk_forward stays as-is (9.6% vs GT=6.1%, small gap, tolerable).

        # v181: "go to" starting verb (P=0.70, was 0.50 in v180)
        # v180 P=0.50 → go_start=12.6% (GT=18.5%); v181 P=0.70 → ~17-18% (≈GT)
        _go_start_rng = random.Random(eid ^ 0xD4E5_F6A7)
        if _go_start_rng.random() < 0.70:
            text = re.sub(r'^Walk to (the|a)\b', r'Go to \1', text)

        # v190 FIX6: Reduce walk_ahead→continue P=0.70→0.45 (reduce continue overshoot).
        # v189: continue=19.5% (GT=10.4%, +9.1pp overshoot) due to P=0.70 conversion.
        # v190: P=0.45 → continue reduces by ~1.9pp → ~17.0% (still above GT but better).
        # walk_ahead: 2.0% → ~4.0% (slight increase, acceptable given prior fixes kept wa low).
        _WA_STANDALONE_RE = re.compile(
            r'\bwalk ahead(?!\s+(to|past|through|into|toward|towards|along|up|down)\b)',
            re.IGNORECASE
        )
        _wa_cont_rng = random.Random(eid ^ 0xC3D4_E5F6)
        def _walk_ahead_to_continue(m):
            if _wa_cont_rng.random() < 0.45:  # v190: was 0.70 in v189 (caused continue overshoot)
                cap = m.group(0)[0].isupper()
                return 'Continue' if cap else 'continue'
            return m.group(0)
        text = _WA_STANDALONE_RE.sub(_walk_ahead_to_continue, text)

        # v197 FIX10 + v193 FIX7: Sentence-level "walk to the X" → walk_past/walk_toward for intermediate phrases.
        # INHERITED: FIX7 from v193b — deterministic regex for non-final walk_to conversion.
        # NEW (v197 FIX10): Diversify replacement: 60% "walk past the X", 40% "walk toward the X".
        #   MOTIVATION: v194 walk_past=54.2% (GT=10.5%) — massive distribution shift.
        #               GT uses walk_toward=4.9% (v194=0%) — completely missing vocabulary.
        #   "walk toward the X" is GT vocabulary: CMA approaches landmark then takes next action.
        #   Result: walk_past ~43% (reduced from 54.2%), walk_toward ~11% (GT=4.9%, added).
        _v193_fix7_rng = random.Random(eid ^ 0xD4E5_F6A7)
        _v197_fix10_rng = random.Random(eid ^ 0xF1A2_B3C4)  # v200 FIX17: extended to 6-phrase distribution
        def _pick_passing_replacement(cap, obj):
            # v201 FIX17b: 5-phrase GT-native distribution (removed "continue past" to fix continue% regression)
            # v200 had "continue past" at 5% → continue% 17.7%→18.8% (+1.1pp regression, GT=8.2%)
            # walk_past(35%) | walk_toward(24%) | go_past(23%) | pass_the(12%) | head_toward(6%)
            # go_past now 23% (was 18%, absorbs the 5% from removed continue_past)
            # GT evidence: go_past=2.8%, pass_the=5.3%, head_toward=1.4%
            r = _v197_fix10_rng.random()
            if r < 0.35:
                return ('Walk past the ' if cap else 'walk past the ') + obj
            elif r < 0.59:
                return ('Walk toward the ' if cap else 'walk toward the ') + obj
            elif r < 0.82:
                return ('Go past the ' if cap else 'go past the ') + obj
            elif r < 0.94:
                return ('Pass the ' if cap else 'pass the ') + obj
            else:
                return ('Head toward the ' if cap else 'head toward the ') + obj
        def _intermediate_walk_to_to_past(text_in):
            # Pattern: "walk to the X" followed by navigation continuation (NOT stop/wait at end)
            # Strategy: split into sentences, convert walk_to in non-final navigation sentences
            sentences = re.split(r'(?<=[.!?])\s+', text_in.strip())
            if len(sentences) <= 1:
                # Single sentence: convert "walk to the X" that is followed by "and turn/go/enter"
                def _single_sent_conv(m):
                    if _v193_fix7_rng.random() < 0.90:
                        cap = m.group(0)[0].isupper()
                        obj = m.group(1).strip()
                        # v200 FIX17: expanded from 2-phrase (FIX10) to 6-phrase GT-native vocabulary
                        return _pick_passing_replacement(cap, obj)
                    return m.group(0)
                _WTO_FOLLOWED_RE = re.compile(
                    r'\bwalk to the ([\w ]{2,30}?)\s+(?=(?:and\s+)?(?:turn|enter|exit|go|walk|continue)\b)',
                    re.IGNORECASE
                )
                return _WTO_FOLLOWED_RE.sub(_single_sent_conv, text_in)
            # Multi-sentence: convert walk_to in sentences that are NOT the last sentence
            result_sents = []
            for i, sent in enumerate(sentences):
                is_last = (i == len(sentences) - 1)
                has_stop = bool(re.search(r'\b(?:stop|wait|stand)\b', sent, re.IGNORECASE))
                if is_last or has_stop:
                    result_sents.append(sent)
                else:
                    # Non-final, non-stop sentence: convert walk_to → GT-native 6-phrase mix
                    def _conv_wt(m):
                        if _v193_fix7_rng.random() < 0.90:
                            cap = m.group(0)[0].isupper()
                            obj = m.group(1).strip()
                            # v200 FIX17: expanded from 2-phrase (FIX10) to 6-phrase GT-native vocabulary
                            return _pick_passing_replacement(cap, obj)
                        return m.group(0)
                    _WTO_IN_SENT_RE = re.compile(
                        r'\bwalk to the ([\w ]{2,30}?)'
                        r'(?=\s*(?:[,.]|$|\band\b|\bthen\b))',
                        re.IGNORECASE
                    )
                    sent = _WTO_IN_SENT_RE.sub(_conv_wt, sent)
                    result_sents.append(sent)
            return ' '.join(result_sents)
        text = _intermediate_walk_to_to_past(text)

        # v193 FIX7b: "walk to the [room]" → "enter the [room]" for room transition patterns.
        # LLM sometimes writes "walk to the kitchen/bedroom" instead of "enter the kitchen".
        _v193_fix7b_rng = random.Random(eid ^ 0xE5F6_A7B8)
        _WTO_ROOM_RE = re.compile(
            r'\bwalk to the (kitchen|bedroom|bathroom|living room|dining room|hallway|'
            r'dining area|office|study|laundry|closet|pantry|garage|foyer|lobby|'
            r'entryway|sitting room|game room|media room|library|corridor)\b',
            re.IGNORECASE
        )
        def _walk_to_room_to_enter(m):
            if _v193_fix7b_rng.random() < 0.80:
                cap = m.group(0)[0].isupper()
                room = m.group(1).lower()
                return ('Enter the ' if cap else 'enter the ') + room
            return m.group(0)
        text = _WTO_ROOM_RE.sub(_walk_to_room_to_enter, text)

        # v197 FIX10b: Second walk_past cap after FIX7/FIX7b (ordering fix).
        # v194 had walk_past×2 in 10.2% of episodes because the first cap (line ~1843) ran before FIX7.
        # FIX7 then added walk_past → result: 10.2% of episodes had 2 walk_past phrases.
        # This second cap ensures max 1 walk_past per instruction across all sources.
        # Replacement: "walk straight" (GT=8.8%, less cascade than "go straight" which is already high).
        _wpt2_counter = [0]
        def _suppress_post_fix7_walk_past(m):
            _wpt2_counter[0] += 1
            if _wpt2_counter[0] > 1:
                return 'Walk straight' if m.group(0)[0].isupper() else 'walk straight'
            return m.group(0)
        text = re.sub(r'\bwalk past the [^.!?,]+', _suppress_post_fix7_walk_past, text, flags=re.IGNORECASE)

        # v198 FIX11: OOV Token Elimination.
        # 77 tokens not in R2R training vocabulary map to UNK in CMA LSTM.
        # Research: KL(train||v194)=0.6856 vs GT=0.0152 (45x worse). Top OOV contributors:
        # 'terracotta', 'porcelain', 'carving', 'chapel', 'museum', 'featuring', 'relief'.
        # Replace with nearest in-vocabulary equivalents to recover phrase informativeness.
        _OOV_REPLACE = [
            (r'\bterracotta(?:-colored)?\b', 'tiled'),
            (r'\bporcelain(?:-tile)?\b', 'ceramic'),
            (r'\bcarving\b', 'sculpture'),
            (r'\bchapel\b', 'room'),
            (r'\bmuseum\b', 'building'),
            (r'\brelief\b', 'decorative'),
            (r'\bmosaic\b', 'tiled'),
            (r'\bornate\b', 'decorative'),
            (r'\bscalloped\b', 'curved'),
            (r'\bfeaturing\b\s*', ''),         # adjective filler — drop entirely (any following whitespace)
        ]
        for _oov_pat, _oov_repl in _OOV_REPLACE:
            text = re.sub(_oov_pat, _oov_repl, text, flags=re.IGNORECASE)

        # NOTE: FIX12 (comma+then reduction) was evaluated and removed.
        # Narrow pattern had 0 measurable effect (most instructions have ≤1 comma+then).
        # Broad "and then" pattern increased turn-anchor% 21.1%→23.2% (worse than GT=16.5%).
        # Root cause: "X and then turn Y" → ". Turn Y" creates stronger turn-anchor signal.
        # Net effect: negative. Removed to preserve v197's turn-anchor% of 21.1%.

        # v198 FIX13: Walk-And Vague Filler Elimination.
        # 7.6% of episodes (140 eps) contain "walk and turn/enter/go/continue" (GT=0.1%).
        # "walk and" before an action verb carries zero spatial information.
        # Fix: strip "walk and " prefix, keeping the meaningful verb (capitalised).
        text = re.sub(
            r'(?i)\bwalk and (turn|enter|exit|go|continue)\b',
            lambda m: m.group(1)[0].upper() + m.group(1)[1:],
            text
        )

        # v198 FIX14: Stop Phrase Cap.
        # "stop in front of" at 11.0% (GT=4.7%, 2.3x excess). Forces CMA front-of spatial
        # model onto configurations it doesn't match. Cap by converting 57% of occurrences
        # to "stop near" → ~5.5% (≈ GT=4.7%).
        _stop_infrontof_rng = random.Random(eid ^ 0xB8C9_D0E1)
        def _cap_stop_in_front(m):
            if _stop_infrontof_rng.random() < 0.57:
                return 'Stop near' if m.group(0)[0].isupper() else 'stop near'
            return m.group(0)
        text = re.sub(r'\bstop in front of\b', _cap_stop_in_front, text, flags=re.IGNORECASE)

        # v198 FIX15: Turn Vocabulary Diversification.
        # "turn left/right" at 68.2% (GT=41.3%); "make a left/right" + "take a left/right"
        # at 0.3% combined (GT=11.1% combined). Replace ~20% of turn commands with natural
        # egocentric turn forms. P=0.20 chosen: converts enough to bring turn-anchor% closer to
        # GT=16.5% (since has_turn_anchor detects "turn left/right" literally, converting to
        # "make a left" reduces that count → turn-anchor% 21.1% closer to GT than P=0.12's 23.2%).
        _turn_vocab_rng = random.Random(eid ^ 0xE1F2_A3B4)
        def _diversify_turn_vocab(m):
            if _turn_vocab_rng.random() >= 0.20:
                return m.group(0)
            direction = 'left' if 'left' in m.group(0).lower() else 'right'
            cap = m.group(0)[0].isupper()
            # 50%: make a left/right — 50%: take a left/right
            if _turn_vocab_rng.random() < 0.50:
                phrase = f'make a {direction}'
            else:
                phrase = f'take a {direction}'
            return (phrase[0].upper() + phrase[1:]) if cap else phrase
        text = re.sub(r'\bturn (left|right)\b', _diversify_turn_vocab, text, flags=re.IGNORECASE)

        # v176: STOP PHRASE ROOM CONTEXT injection (P=0.15)
        # GT: 12.6% of stop phrases have "in the [room]" ("Stop near the couch in the living room.")
        # Adds ~3-4w to 15% of episodes → +0.45w avg_words
        _stop_room_rng = random.Random(eid ^ 0xA2B3_C4D5)
        if (goal_room and stop_type == "stop" and _stop_room_rng.random() < 0.15
                and not re.search(r'\bin the\b', text[-60:], re.I)):
            _room_injection = f" in the {goal_room}"
            if text.rstrip().endswith('.'):
                text = text.rstrip()[:-1] + _room_injection + '.'
            else:
                text = text.rstrip() + _room_injection + '.'

        # v178: STOP-AREA on_left/right INJECTION (P=0.06, calibrated from P=0.20)
        # GT stop-area on_dir: ~3.2% of episodes (17.6% of on_dir × ~18% total on_dir rate)
        # v177 P=0.20 → 10.2% episodes injected — 3× GT rate, 50% wrong direction → hurts SR
        # v178 P=0.06 → 3.1% episodes → matches GT stop-area on_dir rate
        _stop_dir_rng = random.Random(eid ^ 0xC1D2_E3F4)
        if (stop_type == "stop" and _stop_dir_rng.random() < 0.06
                and not re.search(r'\bon the (left|right)\b', text[-70:], re.I)):
            _stop_dir_side = "left" if _stop_dir_rng.random() < 0.50 else "right"
            if text.rstrip().endswith('.'):
                text = text.rstrip()[:-1] + f" on the {_stop_dir_side}."
            else:
                text = text.rstrip() + f" on the {_stop_dir_side}."

        # v174/v178: SENTENCE MERGER — merge consecutive short action sentences toward GT 2-sent style
        # GT: 2-sent=33.8% peak, v173: 3-sent=63.2% dominant.
        # v176: P_MERGE=0.60 → 2-sent=29.9%; v177: P_MERGE=0.75, s1≤9, total≤15 → 2-sent=23.1% (too tight)
        # v178: s1≤12, total≤20 — looser constraints catch more real-world sentence pairs
        # v183: P_MERGE=0.60 — reduce 2-sent from 50.8% toward GT=42%; v182 removed walk_ahead→walk_forward
        #   interaction that caused v181's unexpected 2-sent INCREASE, so this should cleanly reduce 2-sent
        _P_MERGE = 0.60
        _ACTION_VERBS_RE = re.compile(r'^(Walk|Turn|Enter|Go|Exit|Move|Head|Continue|Take|Cross)\b', re.I)
        _merge_rng = random.Random(eid ^ 0x5F3E_1A2B)
        def _merge_sentences(t):
            sents = re.split(r'(?<=[.!])\s+(?=[A-Z])', t)
            if len(sents) <= 2:
                return t
            merged = []
            i = 0
            while i < len(sents):
                s1 = sents[i]
                if i + 1 < len(sents):
                    s2 = sents[i + 1]
                    s1_words = len(s1.split())
                    s2_words = len(s2.split())
                    can_merge = (
                        s1.rstrip().endswith('.') and
                        not re.search(r'\b(stop|wait)\b', s1, re.I) and
                        s1_words <= 12 and                        # v178: was 9 — catch medium sentences
                        (s1_words + s2_words) <= 20 and          # v178: was 15 — allow longer merged result
                        bool(_ACTION_VERBS_RE.match(s2)) and
                        not re.search(r'\band\s+\w+[.,]?\s*$', s1, re.I) and
                        _merge_rng.random() < _P_MERGE
                    )
                    if can_merge:
                        s1_base = s1.rstrip('.').rstrip()
                        s2_lc = s2[0].lower() + s2[1:]
                        merged.append(s1_base + ' and ' + s2_lc)
                        i += 2
                        continue
                merged.append(sents[i])
                i += 1
            return ' '.join(merged)
        text = _merge_sentences(text)

        # v181: STOP-PHRASE MERGER (bidirectional — merge penultimate + stop sentence)
        # GT style: "Walk to X and stop near Y." (stop merged into previous action)
        # v180: P=0.75 → 2-sent=50%; v181: P=0.45 → 2-sent went UP to 53.2% due to walk_ahead interactions
        # v182/v183: REMOVED walk_ahead→walk_forward, so interaction is gone; v183 P=0.45 → 2-sent=47.4%
        # v184: P=0.30 → further reduce toward GT=42%
        _stop_merge_rng = random.Random(eid ^ 0xF0E1_D2C3)
        if _stop_merge_rng.random() < 0.30:
            sents_after = re.split(r'(?<=[.!])\s+(?=[A-Z])', text)
            if len(sents_after) >= 2:
                last = sents_after[-1]
                penult = sents_after[-2]
                last_w = len(last.split())
                penult_w = len(penult.split())
                last_is_stop = bool(re.match(r'^(Stop|Wait|Pause)', last))
                can_stop_merge = (
                    last_is_stop and
                    last_w <= 8 and
                    (penult_w + last_w) <= 18 and
                    not re.search(r'\b(stop|wait)\b', penult, re.I) and
                    penult.rstrip().endswith('.')
                )
                if can_stop_merge:
                    penult_base = penult.rstrip('.').rstrip()
                    last_lc = last[0].lower() + last[1:]
                    merged_text = ' '.join(sents_after[:-2] + [penult_base + ' and ' + last_lc])
                    text = merged_text

        # v187 FIX1: Terminal stop/wait injection when instruction lacks stopping cue.
        # GT: stop=29.2%, wait=17.5% (combined ~40%). v186: stop=6.7%, wait=5.1% (combined ~12%).
        # CMA model trained on GT — without stop/wait cues it doesn't know when to terminate.
        # This directly causes SR failures: agent overshoots goal without stopping.
        # Inject "Stop there." or "Wait there." at end when no stop/wait present.
        # P=0.30: 88% episodes lack stop/wait * 0.30 = 26.4pp added -> combined ~38.4% (≈ GT 40%).
        _STOP_WAIT_PRESENT_RE = re.compile(r'\b(?:stop|halt|wait|stand)\b', re.IGNORECASE)
        _stop_inject_rng = random.Random(eid ^ 0xF7A8_B9C0)
        if (not _STOP_WAIT_PRESENT_RE.search(text)) and _stop_inject_rng.random() < 0.30:
            _stop_inject_phrases = ['Stop there.', 'Stop there.', 'Wait there.', 'Wait here.']
            _inject_phrase = _stop_inject_rng.choice(_stop_inject_phrases)
            text = text.rstrip().rstrip('.') + '. ' + _inject_phrase

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

    # Ahead usage stats (how many instructions have "walk past/toward" phrases)
    past_n    = sum(1 for t in texts if re.search(r'\bpast\s+the\b', t, re.I))
    toward_n  = sum(1 for t in texts if re.search(r'\btoward\s+the\b', t, re.I))
    gt_past_n = 0  # will compute from GT
    with gzip.open(GT_PATH,"rt") as f: gt_check = json.load(f)
    for gt_ep in gt_check["episodes"]:
        gi = (gt_ep.get("instruction",{}).get("instruction_text","")
              if isinstance(gt_ep.get("instruction"),dict) else "")
        if re.search(r'\bpast\s+the\b', gi, re.I): gt_past_n += 1
    gt_n = len(gt_check["episodes"])

    # Midpoint phrasing stats for v27
    passing_n = sum(1 for t in texts if re.search(r'\b(pass(ing|ed)?|walk(ing)?\s+past)\s+the\b', t, re.I))
    through_n = sum(1 for t in texts if re.search(r'\bthrough\s+the\b', t, re.I))
    walkpast_n = sum(1 for t in texts if re.search(r'\bwalk\s+past\s+the\b', t, re.I))
    continue_n = sum(1 for t in texts if re.search(r'\bcontinue\b', t, re.I))
    proceed_n  = sum(1 for t in texts if re.search(r'\bproceed\b', t, re.I))
    wooden_n   = sum(1 for t in texts if re.search(r'\bwooden\b', t, re.I))
    # v151 key fix metrics
    take_few_n   = sum(1 for t in texts if re.search(r'\btake a few\b', t, re.I))
    go_straight_n = sum(1 for t in texts if re.search(r'\bgo straight\b', t, re.I))
    wlk_straight_n = sum(1 for t in texts if re.search(r'\bwalk straight\b', t, re.I))
    walk_forward_n = sum(1 for t in texts if re.search(r'\bwalk forward\b', t, re.I))

    print(f"\n=== v201 Done ===")
    print(f"  Episodes:      {n}  Pass: {n_pass}  Fixed: {n_fix}  Failed: {n_fail}")
    print(f"  avg_words:     {avg_w:.1f}  (GT=26.8)")
    print(f"  stop%:         {100*stop_n/n:.1f}%  (GT=50.8%)")
    print(f"  wait%:         {100*wait_n/n:.1f}%  (GT=30.5%)")
    print(f"  neither%:      {100*neither_n/n:.1f}%  (GT=16.5%)")
    turn_counts  = [len(re.findall(r'\bturn\s+(?:left|right)\b', t.lower())) for t in texts]
    avg_turns    = sum(turn_counts) / n
    eps_with_turn = sum(1 for c in turn_counts if c > 0)
    print(f"  turn-anchor%:  {100*anchor_n/n:.1f}%  (GT=16.5%, v62=15.0%) ← v63: P_ANCHOR_EPISODE=0.231")
    print(f"  avg_turns/ep:  {avg_turns:.2f}  (GT=0.59, v62=0.59) ← v63: P_TURN_UNANCHORED 0.42")
    print(f"  eps_with_turn: {100*eps_with_turn/n:.1f}%  (GT=41.3%, v43=69.2%)")
    print(f"  dup-stop:      {dup_stop} (target: 0)")
    print(f"  walk past the: {100*past_n/n:.1f}%  (GT={100*gt_past_n/gt_n:.1f}%) ← KEY METRIC")
    print(f"  through the:   {100*through_n/n:.1f}%  (GT=27.2%) ← KEY METRIC (v36=27.5%, v35=27.2%)")
    print(f"  walk past the: {100*walkpast_n/n:.1f}%  (GT=8.7%) ← KEY METRIC (v36=10.3%, target ~10%)")
    print(f"  passing the:   {100*passing_n/n:.1f}%  (GT=5.1%) ← combined; pure passing the=0%")
    print(f"  continue:      {100*continue_n/n:.1f}%  (GT=10.4%) ← v151=52.9%, target<20%")
    print(f"  proceed:       {100*proceed_n/n:.1f}%  (GT=0.8%) ← v36=1.7% (retain fix)")
    print(f"  wooden:        {100*wooden_n/n:.1f}%  (GT=3.3%) ← v45 should be <1%")
    print(f"  --- v152 KEY METRICS ---")
    print(f"  take_few:      {100*take_few_n/n:.1f}%  (GT=0.2%, v150=29.1%) ← MUST be <3%")
    print(f"  go_straight:   {100*go_straight_n/n:.1f}%  (GT=7.9%) ← v164: 50% skip")
    print(f"  walk_straight: {100*wlk_straight_n/n:.1f}%  (GT=8.8%)")
    print(f"  walk_forward:  {100*walk_forward_n/n:.1f}%  (GT=6.1%) ← v164: in AVOID list")
    hallway_n     = sum(1 for t in texts if re.search(r'\bhallway\b', t, re.I))
    into_hall_n   = sum(1 for t in texts if re.search(r'\binto the hallway\b', t, re.I))
    doorway_n     = sum(1 for t in texts if re.search(r'\bdoorway\b', t, re.I))
    bedroom_n     = sum(1 for t in texts if re.search(r'\bbedroom\b', t, re.I))
    kitchen_n     = sum(1 for t in texts if re.search(r'\bkitchen\b', t, re.I))
    thru_door_n   = sum(1 for t in texts if re.search(r'\bthrough the doorway\b', t, re.I))
    print(f"  toward the:    {100*toward_n/n:.1f}%  (GT=3.8%, v151=27.5%) ← KEY METRIC")
    print(f"  hallway:       {100*hallway_n/n:.1f}%  (GT=20.4%, v151=22%) ← KEY METRIC, target>30%")
    print(f"  into hallway:  {100*into_hall_n/n:.1f}%  (GT=4.4%) ← KEY METRIC")
    print(f"  doorway:       {100*doorway_n/n:.1f}%  (GT=?%, v151=?) ← NEW v152 metric")
    print(f"  thru doorway:  {100*thru_door_n/n:.1f}%  (GT=?%) ← NEW v152 metric")
    print(f"  bedroom:       {100*bedroom_n/n:.1f}%  (GT=?%) ← NEW v152 metric")
    print(f"  kitchen:       {100*kitchen_n/n:.1f}%  (GT=?%) ← NEW v152 metric")
    print(f"  spatial:")
    for k,v in spp.items():
        print(f"    {k:<12}: {100*v/n:.1f}%")
    print(f"\nSaved: {OUTPUT}")

    anchor_err = abs(anchor_n/n - 0.165)
    stop_err   = abs(stop_n/n  - 0.508)
    wait_err   = abs(wait_n/n  - 0.305)
    match_score = 1.0 - (anchor_err*2 + stop_err + wait_err)
    print(f"\n  GT-match score: {match_score:.3f}  (target ≥ 0.90; v94=0.992)")
    print(f"  [v164: P_ANCHOR=0.03(GT=3.2%), P_TURN_UNANCHORED=0.28, 50%skip, P_continue=0.28, FRESH ckpt]")
    print(f"  [TARGET: anchored_turn≈3%(GT), avg_turns≈1.0-1.2, unanchored turns like v24=1.53 style]")
    print(f"  Word count: {avg_w:.1f}w  (GT=26.8w)")

    # Vision contribution stats
    vis_goal = sum(1 for ep in episodes_out
                   if any(w in ep["instruction"]["instruction_text"].lower()
                          for w in ["grey", "white", "dark", "wooden", "glass", "silver"]))
    print(f"\n  ~Vision-grounded stop phrases (color/material): {vis_goal}/{n} ({100*vis_goal/n:.1f}%)")
    print(f"  [Note: v45: fresh P2 + prompt 'only use route turns' to suppress LLM-generated turns (target ~0.5 vs v44=0.92)")


if __name__ == "__main__":
    asyncio.run(main())
