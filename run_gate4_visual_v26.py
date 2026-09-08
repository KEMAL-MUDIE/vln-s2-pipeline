#!/usr/bin/env python3
"""
Gate 4 Visual v26 — Turn Pruning + Door Landmarks + Adjusted Word Budget

IMPROVEMENTS OVER v25:
  1. TURN PRUNING: Skip mild un-anchored turns (angle < 55 deg, no landmark, no room)
     from route description. GT avg explicit turn commands = 0.43/ep vs our 1.52.
  2. DOOR LANDMARKS ENABLED: "door"/"doors" removed from is_turn_generic() reject set.
     GT uses door references 47.9% vs our 28.4%.
  3. ADJUSTED WORD BUDGET: path_len=5 raised to 20-28w (was 18-26w).
  4. All v25 innovations retained: anti-loop, remove_loops, turn-side, dual-segment.
     Checkpoint: gate4_v24_turn_sides_p1_checkpoint.json

  NOTE: Dual-segment midpoints from v23.
     Checkpoint: gate4_v22_midpoint_p1_checkpoint.json (reused)

  Phase 2 — Instruction generation with turn-side + dual-midpoint landmarks:
    Turn landmarks: prefer turn-side desc, fall back to approach-direction Phase 1.
    Midpoints: first AND last segment (up to 2 [pass: X] per instruction).
    Checkpoint: outputs/gate4_v26_phase2_checkpoint.json
    Output: outputs/datasets/val_unseen_generated_gemma_visual_v26.json.gz
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
OUTPUT        = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v26.json.gz"
P1_CKPT       = ROOT / "outputs" / "gate4_v19_phase1_checkpoint.json"  # reuse v19 Phase 1!
P1_MID_CKPT   = ROOT / "outputs" / "gate4_v22_midpoint_p1_checkpoint.json"  # reuse v22 midpoints
P1_TS_CKPT    = ROOT / "outputs" / "gate4_v24_turn_sides_p1_checkpoint.json"  # reuse v24 turn-sides
P2_CKPT       = ROOT / "outputs" / "gate4_v26_phase2_checkpoint.json"

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


def load_turn_sides_checkpoint() -> Dict:
    """Load Phase 1 turn-sides checkpoint: {eid_str: {turn_N: desc}}"""
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
    """
    if not desc:
        return ""
    sent = re.split(r'(?<=[.!?])\s+', desc.strip())[0].strip()
    # Remove "Turn at/past/through..." prefix if Gemma added it
    sent = re.sub(r'^Turn\s+(at|past|through|around|into)\s+(the\s+)?', '', sent,
                  flags=re.IGNORECASE).strip()
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
    goal_vis = vis.get("goal")
    if goal_vis:
        info["goal_landmark"] = extract_noun_phrase(goal_vis, max_words=5)
        info["goal_full"] = extract_goal_full(goal_vis)  # for "Goal:" line in prompt
    elif info["goal_landmark"] == "the destination":
        info["goal_full"] = "the destination"
    else:
        info["goal_full"] = info["goal_landmark"]

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

def build_route_desc(primitives, sel_turns, seg_midpoint: Optional[Dict[str, str]] = None):
    """Build route description string.
    seg_midpoint: {endpoint_label: desc} — up to 2 entries (v23: first + last segment).
    endpoint_label is "turn_N" (1-indexed) or "goal".
    """
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
            lm    = td.get("landmark","")
            rm    = td.get("room","")
            rt    = td.get("room_trans","")
            ahead = td.get("ahead","")
            angle = p.get("angle_deg", 90)
            # v23: insert [pass: X] before this turn if this is a midpoint endpoint
            turn_label = f"turn_{turn_idx + 1}"
            if seg_midpoint and turn_label in seg_midpoint:
                parts.append(f"[pass: {seg_midpoint[turn_label]}]")
            # v26: Skip mild un-anchored turns (angle < 55°, no landmark, no room).
            # GT avg=0.43 explicit turns/ep vs our 1.52. These micro-turns confuse the model.
            if not lm and not rm and angle < 55:
                turn_idx += 1
                continue
            # Build turn phrase
            if lm and rt and "doorway" in rt.lower():
                base = f"turn {direction} past [{lm}] into [{rm}]"
                parts.append(f"{base} → [{ahead} ahead]" if ahead else base)
            elif lm:
                base = f"turn {direction} at [{lm}]"
                parts.append(f"{base} → [{ahead} ahead]" if ahead else base)
            elif rm:
                parts.append(f"turn {direction} → {rm}")
            else:
                parts.append(f"turn {direction}")
            turn_idx += 1
        elif t == "elevation":
            parts.append(f"{'up' if p.get('direction','up')=='up' else 'down'} stairs")
    # v23: [pass: X] for last segment (before goal), if not already in a turn
    if seg_midpoint and "goal" in seg_midpoint:
        parts.append(f"[pass: {seg_midpoint['goal']}]")
    return " → ".join(parts) or "walk forward"


def gt_start_verb(gt_instr):
    tok = gt_instr.strip().split()
    if not tok: return "Walk"
    w = tok[0].rstrip(".,!?;:")
    return (w[0].upper() + w[1:].lower()) if w else "Walk"


# ── System prompt (same as v17, updated for v18 vision context) ──────────────

SYSTEM_INTRO = """You write concise R2R navigation instructions for an indoor robot.

Style: Natural and brief, like a human R2R annotator.

RULES:
- When a turn has [object] in brackets: use that object as a turn reference — naturally: "turn left at the grey pillar", "turn right past the dining table"
- When a turn has → [object ahead] after the arrow: add a BRIEF walking phrase (3-5 words) like "walk past the grey sofa" or "continue toward the television" — keep it very short
- When route has [pass: object] before a turn or the goal: the robot walks past that object — write ONLY "passing the X" or "continuing past the X" (3-5 words). FORBIDDEN: "walk past", "go past", "head past". Use ONLY gerund form "passing the X". Use ONCE per [pass: X] marker (up to 2 per instruction)
- When a turn has (→ room) after the arrow: mention the room naturally — "walk into the living room" — NOT as a turn anchor
- Turns with no brackets/arrows: write ONLY "turn left" or "turn right" — no objects
- [object] and [object ahead] are SEPARATE: [object] = where to turn, [object ahead] = what to pass AFTER the turn. Do NOT confuse them.
- Do NOT invent or add objects not shown in the route
- CRITICAL: Do NOT repeat any direction, location, or landmark. Each segment is described EXACTLY ONCE. Never write "go down the stairs" or "turn right" more than once.
- Endings vary: "Stop near X." / "Wait near X." / no explicit ending
- Start verb is given — use it exactly"""


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
            # Only first sentence (about 15-25 words)
            first_sent = re.split(r'(?<=[.!?])\s+', desc)[0]
            lines.append(f"  Near turn {i+1} ({direction}): {first_sent}")

    if not lines:
        return ""
    header = ("Visual context (background only — do NOT use these to add landmarks to "
              "un-bracketed turns):\n")
    return header + "\n".join(lines)


def build_prompt(ep, gt_instr, examples, lm_info, primitives, sel_turns,
                 stop_phrase, stop_type, vision_map: Dict, seg_midpoint: Optional[Dict] = None):
    sv    = gt_start_verb(gt_instr)
    route = build_route_desc(primitives, sel_turns, seg_midpoint)
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
    n_pass = len(seg_midpoint) if seg_midpoint else 0
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
    if n_ahead > 0:
        anchor_note += (f" Route also has {n_ahead} [X ahead] marker(s) — after turning at"
                        f" [landmark], add a brief 'walk toward/past X' phrase (3-5 words max).")
    if n_pass == 1:
        anchor_note += (f" Route has 1 [pass: X] marker — write 'passing the X' ONCE (gerund form only;"
                        f" FORBIDDEN: 'walk past', 'go past') naturally in 3-4 words.")
    elif n_pass > 1:
        anchor_note += (f" Route has {n_pass} [pass: X] markers — write 'passing the X' ONCE PER MARKER"
                        f" (gerund form only; FORBIDDEN: 'walk past', 'go past'), 3-4 words each.")

    # v25/v26: path-length-aware word budget
    path_len = len(ep.get("reference_path", []))
    if path_len <= 4:
        anchor_note += " LENGTH: Write 13-21 words total. Very short path — be brief."
    elif path_len == 5:
        anchor_note += " LENGTH: Write 20-28 words total."  # v26: +2w vs v25 (was 18-26)
    elif path_len == 6:
        anchor_note += " LENGTH: Write 23-31 words total."
    else:
        anchor_note += " LENGTH: Write 27-36 words total."

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


def remove_loops(text: str, stop_phrase: str = "") -> str:
    """v25: Detect repeated 5-gram loops and truncate, appending stop_phrase if needed."""
    words = text.split()
    if len(words) < 12:
        return text
    seen: dict = {}
    for i in range(len(words) - 4):
        ng = ' '.join(words[i:i+5]).lower()
        if ng in seen:
            truncated = ' '.join(words[:seen[ng]]).rstrip('.,;:')
            if stop_phrase:
                truncated = truncated + '. ' + stop_phrase
            elif not truncated.rstrip().endswith(('.', '!', '?')):
                truncated += '.'
            return truncated
        seen[ng] = i
    return text


def quality_ok(text):
    words = text.split()
    if len(words) < 5:  return False, "too_short"
    if len(words) > 80: return False, "too_long"
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

    print("=== Gate 4 v26: Turn Pruning + Door Landmarks + Adjusted Word Budget ===")
    print(f"  P_ANCHOR_EPISODE: {p_anchor} → expected ~{p_anchor*0.44*100:.1f}% (GT=16.5%)")
    print(f"  stop/wait: {P_STOP*100:.0f}%/{P_WAIT*100:.0f}%/{(1-P_STOP-P_WAIT)*100:.0f}%")
    print(f"  Phase 1: REUSED from v19 checkpoint (no new API calls)")
    print(f"  Phase 1b: midpoints checkpoint (reused v22)")
    print(f"  Phase 1c: turn-sides checkpoint (reused v24, 89.4% good rate)")
    print(f"  Phase 2 concurrency: {args.concurrency_p2}")
    print(f"  v26 key changes: turn pruning (skip angle<55 un-anchored), door landmarks enabled, word budget +2w")
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

    # ── Phase 1b: Load midpoints checkpoint ──────────────────────────────────
    midpoints_ckpt = load_midpoints_checkpoint()
    print(f"Phase 1b midpoints checkpoint: {len(midpoints_ckpt)} episodes")
    has_mid = sum(1 for ep in episodes
                  if any(d for d in midpoints_ckpt.get(str(ep["episode_id"]),{}).values() if d))
    print(f"  Episodes with at least 1 midpoint: {has_mid}/{len(episodes)} "
          f"({100*has_mid/len(episodes):.1f}%)")

    # ── Phase 1c: Load turn-sides checkpoint (v24 new) ───────────────────────
    ts_vision_map = load_turn_sides_checkpoint()
    print(f"\nPhase 1c turn-sides checkpoint: {len(ts_vision_map)} episodes")
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
        stop_phrase, stop_type = choose_stop(lm_info["goal_landmark"], lm_info["goal_room"], eid)
        # v23: first AND last segment midpoints (up to 2 entries)
        seg_midpoint = get_all_segment_midpoints(ep, midpoints_ckpt)

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
                                   sel_turns, stop_phrase, stop_type, vision_map,
                                   seg_midpoint=seg_midpoint)
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
        # v25: remove loops BEFORE quality check — truncation may fix loop issues
        text = remove_loops(text, stop_phrase)
        ok, reason = quality_ok(text)
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

    # Midpoint "passing" usage stats
    passing_n = sum(1 for t in texts if re.search(r'\b(pass(ing|ed)?|walk(ing)?\s+past)\s+the\b', t, re.I))

    print(f"\n=== v26 Done ===")
    print(f"  Episodes:      {n}  Pass: {n_pass}  Fixed: {n_fix}  Failed: {n_fail}")
    print(f"  avg_words:     {avg_w:.1f}  (GT=26.8)")
    print(f"  stop%:         {100*stop_n/n:.1f}%  (GT=50.8%)")
    print(f"  wait%:         {100*wait_n/n:.1f}%  (GT=30.5%)")
    print(f"  neither%:      {100*neither_n/n:.1f}%  (GT=16.5%)")
    print(f"  turn-anchor%:  {100*anchor_n/n:.1f}%  (GT=16.5%) ← MUST STAY NEAR 16.5%")
    print(f"  dup-stop:      {dup_stop} (target: 0)")
    print(f"  walk past the: {100*past_n/n:.1f}%  (GT={100*gt_past_n/gt_n:.1f}%) ← KEY METRIC")
    print(f"  passing+walk:  {100*passing_n/n:.1f}% ← v23 dual-segment midpoint landmark usage")
    print(f"  walk toward:   {100*toward_n/n:.1f}%")
    print(f"  spatial:")
    for k,v in spp.items():
        print(f"    {k:<12}: {100*v/n:.1f}%")
    print(f"\nSaved: {OUTPUT}")

    anchor_err = abs(anchor_n/n - 0.165)
    stop_err   = abs(stop_n/n  - 0.508)
    wait_err   = abs(wait_n/n  - 0.305)
    match_score = 1.0 - (anchor_err*2 + stop_err + wait_err)
    print(f"\n  GT-match score: {match_score:.3f}  (target ≥ 0.90; v22=0.938, v19=0.965, v18=0.958)")
    print(f"  Word count: {avg_w:.1f}w  (GT=26.8w)")

    # Vision contribution stats
    vis_goal = sum(1 for ep in episodes_out
                   if any(w in ep["instruction"]["instruction_text"].lower()
                          for w in ["grey", "white", "dark", "wooden", "glass", "silver"]))
    print(f"\n  ~Vision-grounded stop phrases (color/material): {vis_goal}/{n} ({100*vis_goal/n:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
