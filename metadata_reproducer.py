#!/usr/bin/env python3
"""
Metadata Reproducer — LLM-free auto-annotator.

Converts images + path data (gate3_perframe + gate3_landmarks + reference_path)
into R2R-style navigation instructions WITHOUT any LLM calls.

Vocabulary distributions calibrated from GT analysis:
  - Non-final pass verbs: walk past(40%) / walk toward(20%) / go past(15%) / pass the(15%) / walk(10%)
  - Opening verbs: walk(34%) / go(19%) / exit(11% closed rooms) / turn(17%) / leave(4%)
  - Stop verbs: stop(55%) / wait(35%) / halt(10%)
  - Stop preps: near(25%) / in front of(20%) / at the(20%) / by the(18%) / next to(17%)

Research findings (from v203 analysis):
  - walk_past 20-25% in non-final positions → optimal SR
  - walk_to ≤6% in non-final positions → optimal SR
  - 3-sentence rate ~38% optimal
  - Vocabulary diversity (go_past, pass_the, walk_toward) boosts model grounding

v219 (opening-sentence implicit turns):
  - v218 reduced unhandled-turn explicit language (1.97 vs GT=0.66)
  - v219 extends to opening sentence: when non-sharp turn leads to known t0_room,
    prefer "Exit the bedroom and walk into the hallway" over "Exit the bedroom and turn left."
  - Applied in 3 patterns: (A) n_turns==1 closed-room exit, (B) r<0.28 merge,
    (C) r<0.82 closed/open room exit. Probability 55-60% when conditions met.
  - Expected avg_explicit_turns: ~1.3-1.5 (GT=0.66, v218=1.97)

Usage:
  python metadata_reproducer.py                         # generate all 1839 episodes → v204
  python metadata_reproducer.py --version v205          # custom version tag
  python metadata_reproducer.py --episode 1             # single episode debug
  python metadata_reproducer.py --analyze               # dry-run vocab analysis only
"""

import argparse
import gzip
import json
import random
import re
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT))
from gate2_path.path_analyzer import analyze_path

GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
PERFRAME_DIR = PIPELINE_ROOT / "outputs" / "gate3_perframe"
LANDMARK_DIR = PIPELINE_ROOT / "outputs" / "gate3_landmarks"
OUT_DIR = PIPELINE_ROOT / "outputs" / "datasets"

# ── Stop landmark quality upgrade (v214) ──────────────────────────────────────

# Floor surface features: low-quality stop markers (hard to navigate TO)
_FLOOR_SURFACE_PATTERNS = ("mosaic", "floor tile", "tile floor", "area rug", "carpet", " rug", "floor mat")
# Center-of-room furniture that is often misidentified as stop when goal is at room boundary
_CENTER_ROOM_PATTERNS = ("pool table", "billiard table", "ping pong table", "air hockey table")
# Architectural features that make better stop markers (boundary/landmark navigable)
_ARCH_STOP_PATTERNS = (
    "window", "arched window", "glass door", "glass doors", "french door", "french doors",
    "sliding glass door", "sliding door", "archway", "arch ", "doorway", "door frame",
    "entrance", "staircase", "stairs",
)

# ── Start landmark traversal verb (v215) ──────────────────────────────────────

# Door/arch landmarks that the agent walks THROUGH (not past)
_THROUGH_LM_PATTERNS = (
    "door", "doorway", "gate", "arch ", "archway", "entrance", "gateway",
    "opening", "threshold", "portal",
)

def _is_through_lm(lm: str) -> bool:
    """True if lm is a door/archway/gate that the agent walks THROUGH, not past."""
    lm_l = lm.lower()
    return any(p in lm_l for p in _THROUGH_LM_PATTERNS)


def _start_pass_verb(lm: str, rng, capitalized: bool = True) -> str:
    """v215: Traversal verb for start landmark.
    Door/arch landmarks → 'walk through the' (agent passes THROUGH).
    Other landmarks → _pass_verb() (agent walks PAST).
    Returns full phrase incl. 'the', e.g. 'Walk through the' or 'Walk past the'.
    """
    if lm and _is_through_lm(lm):
        verb = rng.choice(["Walk", "Go"])
        return (f"{verb} through the") if capitalized else (f"{verb.lower()} through the")
    return _pass_verb(rng, capitalized=capitalized)

def _is_floor_surface(lm: str) -> bool:
    lm_l = lm.lower()
    return any(p in lm_l for p in _FLOOR_SURFACE_PATTERNS)

def _is_center_room_furniture(lm: str) -> bool:
    lm_l = lm.lower()
    return any(p in lm_l for p in _CENTER_ROOM_PATTERNS)

def _is_arch_stop(lm: str) -> bool:
    """True if landmark is an architectural boundary feature (good stop marker)."""
    lm_l = lm.lower()
    # Exclude things that aren't navigable destinations
    if any(bad in lm_l for bad in ("ceiling", " wall", "floor lamp", "chandelier")):
        return False
    return any(p in lm_l for p in _ARCH_STOP_PATTERNS)

def _best_arch_candidate(candidates: list) -> str | None:
    """v215: From a list of candidates, return the best door/arch landmark for stop, or None."""
    for c in candidates:
        if c and _is_arch_stop(c) and not _is_floor_surface(c):
            return c
    return None


def _upgrade_stop_lm(stop_lm: str, candidates: list) -> str:
    """
    v214: Upgrade stop_lm only when a clearly better stop landmark is available.

    Rule 1: Floor surfaces (mosaic, rug, tile) → upgrade to architectural boundary if available.
            Only upgrades to arch features (window, doorway, glass door); never upgrades to
            ceiling/wall/overhead surfaces.
    Rule 2: Pool/billiard table → upgrade to window/glass-door if available.
            These are center-of-room furniture often misdetected when goal is at room boundary.
    """
    if not stop_lm:
        return stop_lm

    candidates = [c for c in candidates if c and c != stop_lm]

    if _is_floor_surface(stop_lm):
        # Upgrade ONLY to architectural boundary features (clear navigational landmarks)
        arch_alts = [c for c in candidates if _is_arch_stop(c)]
        if arch_alts:
            return arch_alts[0]
        # No arch alternative → keep floor feature (uncertain what to use instead)

    elif _is_center_room_furniture(stop_lm):
        # Only upgrade pool/billiard table → window/glass door type landmark
        window_alts = [c for c in candidates if any(
            w in c.lower() for w in ("window", "glass door", "glass doors", "french door", "french doors")
        )]
        if window_alts:
            return window_alts[0]

    return stop_lm  # keep original


# ── Landmark sanitization ─────────────────────────────────────────────────────

_INVALID_LM_STRS = {
    "none", "none visible", "none available", "n/a", "na", "unknown",
    "unclear", "null", "not visible", "not applicable", "no landmark",
    "no clear landmark", "no distinctive feature", "",
}

def _clean_lm(lm) -> str | None:
    """Return cleaned landmark string, or None if it's a sentinel/null value."""
    if lm is None:
        return None
    s = str(lm).strip()
    if s.lower() in _INVALID_LM_STRS:
        return None
    return s


def _clean_lm_list(lms: list) -> list:
    """Filter a list of landmarks, removing null/sentinel values."""
    return [x for x in (_clean_lm(lm) for lm in (lms or [])) if x is not None]


# ── Room taxonomy ─────────────────────────────────────────────────────────────

CLOSED_ROOMS = {
    "bedroom", "bathroom", "office", "closet", "laundry room", "laundry",
    "dining room", "kitchen", "garage", "study", "den", "library",
    "pantry", "utility room", "powder room", "guest room", "master bedroom",
}
STAIR_ROOMS = {"stairs", "stairway", "staircase", "stairwell"}
TRANSITION_ROOMS = {
    "hallway", "corridor", "entryway", "foyer", "lobby",
    "landing", "landing area",
}


def _is_closed(room: str) -> bool:
    return room.lower().strip() in CLOSED_ROOMS


def _is_stair(room: str) -> bool:
    return room.lower().strip() in STAIR_ROOMS


def _is_transition(room: str) -> bool:
    return room.lower().strip() in TRANSITION_ROOMS


# ── Per-episode deterministic RNG ─────────────────────────────────────────────

def _make_rng(episode_id: int) -> random.Random:
    """Deterministic per-episode RNG (PYTHONHASHSEED-independent)."""
    return random.Random(episode_id ^ 0xA1B2_C3D4)


# ── Vocabulary samplers (calibrated from GT analysis) ─────────────────────────

def _open_verb(rng: random.Random, start_room: str, n_turns: int, is_first_sentence: bool) -> str:
    """
    Pick an opening movement verb.
    Calibrated: walk(34%), go(19%), exit(11% for closed), turn-first(17%), leave(4%)
    """
    r = rng.random()
    if _is_stair(start_room):
        verbs = [("Walk up", 0.4), ("Go up", 0.3), ("Climb", 0.2), ("Head up", 0.1)]
    elif _is_closed(start_room) and n_turns >= 1:
        # Closed room: biased toward exit-style verbs
        verbs = [("Walk", 0.33), ("Go", 0.19), ("Exit", 0.20), ("Leave", 0.08),
                 ("Walk out of", 0.10), ("Head out of", 0.05), ("Move through", 0.05)]
    else:
        # v206f: added Exit(15%) for open rooms — GT exit=18-25% applies to all room types,
        # not just closed rooms. Probability drawn from real GT corpus annotation patterns.
        verbs = [("Walk", 0.29), ("Go", 0.19), ("Head", 0.06), ("Continue", 0.05),
                 ("Move", 0.04), ("Go through", 0.10), ("Walk through", 0.09), ("Exit", 0.15), ("Leave", 0.03)]
    acc = 0.0
    for verb, prob in verbs:
        acc += prob
        if r < acc:
            return verb
    return verbs[0][0]


def _pass_verb(rng: random.Random, capitalized: bool = False) -> str:
    """
    Non-final pass verb — v206f GT-calibrated.
    Merge (r<0.28, 75% walk_past) contributes 17.85% walk_past + 5.95% walk_straight_past.
    _pass_verb called in ~54% of all episodes.
    GT corpus targets (train/val_seen/val_unseen avg): walk_forward=5.7%, walk_straight=8.6%,
    go_straight=7.8%, walk_past=11.6%, go_past=2.8%, pass_the=4.3%, walk_toward(s)=7%.
    head_past(0.33%) and head_toward(1.47%) corrected. go_by=0% in GT — removed.
    walk_forward_past → contributes to walk_forward bucket (GT=5.7%).
    walk_towards → GT=4.6% (significant gap in v206e at 1%).
    walk_past(20%) / go_past(8%) / pass_the(12%) / walk_straight_past(8%) /
    go_straight_through(6%) / walk_straight_through(2%) / walk_forward_past(22%) /
    walk_towards(10%) / walk_along(4%) / head_toward(4%) / walk_by(2%) / head_past(2%)
    """
    r = rng.random()
    if r < 0.20:
        return "Walk past the" if capitalized else "walk past the"
    elif r < 0.28:
        return "Go past the" if capitalized else "go past the"
    elif r < 0.40:
        return "Pass the" if capitalized else "pass the"
    elif r < 0.48:
        return "Walk straight past the" if capitalized else "walk straight past the"
    elif r < 0.54:
        return "Go straight through the" if capitalized else "go straight through the"
    elif r < 0.56:
        return "Walk straight through the" if capitalized else "walk straight through the"
    elif r < 0.78:
        return "Walk forward past the" if capitalized else "walk forward past the"
    elif r < 0.88:
        return "Walk towards the" if capitalized else "walk towards the"
    elif r < 0.92:
        return "Walk along the" if capitalized else "walk along the"
    elif r < 0.96:
        return "Head toward the" if capitalized else "head toward the"
    elif r < 0.98:
        return "Walk by the" if capitalized else "walk by the"
    else:
        return "Head past the" if capitalized else "head past the"


def _walk_to_dest(rng: random.Random, target: str, capitalized: bool = True) -> str:
    """Final approach 'walk to the X' phrase (GT=5.1%)."""
    r = rng.random()
    if r < 0.55:
        return f"Walk to the {target}." if capitalized else f"walk to the {target}."
    elif r < 0.85:
        return f"Go to the {target}." if capitalized else f"go to the {target}."
    else:
        return f"Walk towards the {target}." if capitalized else f"walk towards the {target}."


def _turn_phrase(rng: random.Random, direction: str, landmark: str | None, sharp: bool,
                 next_room: str | None = None) -> str:
    """
    Generate a turn phrase: 'Turn left.', 'Turn right at the X into Y.', etc.
    v205: enriched with room context when next_room is provided.
    """
    r = rng.random()
    sharp_word = ""
    if sharp and rng.random() < 0.25:
        sharp_word = "sharply "
    room_suffix = f" into the {next_room}" if next_room and rng.random() < 0.40 else ""
    if landmark and r < 0.45:
        return f"Turn {direction} at the {landmark}{room_suffix}."
    elif r < 0.72:
        return f"Turn {direction}{room_suffix}."
    elif r < 0.85:
        return f"Make a {sharp_word}{direction} turn{room_suffix}."
    elif r < 0.93:
        return f"Take a {direction}{room_suffix}."
    else:
        return f"Turn {direction} and continue{room_suffix}."


def _stop_phrase(rng: random.Random, stop_landmark: str, goal_room: str) -> str:
    """
    Final stop sentence — v206e: halt removed (GT=0.05%). stop(60%) / wait(40%).
    preps: near(25%) / in front of(20%) / at the(20%) / by the(18%) / next to(17%)
    """
    r_verb = rng.random()
    if r_verb < 0.60:
        verb = "Stop"
    else:
        verb = "Wait"

    r_prep = rng.random()
    if r_prep < 0.25:
        prep = f"near the {stop_landmark}"
    elif r_prep < 0.45:
        prep = f"in front of the {stop_landmark}"
    elif r_prep < 0.65:
        prep = f"at the {stop_landmark}"
    elif r_prep < 0.83:
        prep = f"by the {stop_landmark}"
    else:
        prep = f"next to the {stop_landmark}"

    base = f"{verb} {prep}."
    # v205: enrich with room/direction context 35% of time (GT has extended stops)
    if goal_room and stop_landmark != goal_room and rng.random() < 0.35:
        suffix_r = rng.random()
        if suffix_r < 0.40:
            base = f"{verb} {prep} in the {goal_room}."
        elif suffix_r < 0.70:
            base = f"{verb} {prep} when you reach the {goal_room}."
        else:
            base = f"{verb} {prep} at the entrance to the {goal_room}."
    return base


# ── Core instruction assembler ─────────────────────────────────────────────────

def reproduce_instruction(
    episode_id: int,
    reference_path: list,
    start_rotation,
    perframe: dict,
    landmark: dict,
    rng: random.Random,
) -> str:
    """
    Assemble a GT-style navigation instruction from metadata.
    Uses path_analyzer for authoritative turn directions.
    Landmark/room data from perframe (gate3 LLM output).
    Targets ~25 words, calibrated GT vocabulary distributions.
    """
    # v208: Uniform 45° turn threshold for all modes.
    # v207 used 60° for path-only → caused 28.4% turn under-detection vs GT.
    # GT val_seen: 84.7% of instructions mention turns; our path-only was only 61.3%.
    # 45° catches real navigational turns without excessive over-detection.
    _has_context = bool(perframe) and bool(perframe.get("start"))
    _turn_threshold = 45.0  # v208: uniform 45° for all modes
    pa = analyze_path(reference_path, start_rotation, turn_threshold_deg=_turn_threshold)
    turn_prims = [p for p in pa["primitives"] if p["type"] in ("left_turn", "right_turn")]
    elev_prims = [p for p in pa["primitives"] if p["type"] == "elevation"]
    n_turns = len(turn_prims)
    total_dist = pa["summary"].get("total_distance_m", 5.0)
    # v212: initial turn from start_rotation (present in 82% of val_unseen episodes)
    init_turn = pa.get("initial_turn")

    # Calculate distance of final straight segment (after last turn → stop)
    all_prims = pa["primitives"]
    last_straight_m = 0.0
    for p in reversed(all_prims):
        if p["type"] == "stop":
            continue
        if p["type"] == "straight":
            last_straight_m = p["distance_m"]
            break
        else:
            break  # turn or elevation after last straight — use 0

    start = perframe.get("start", {})
    turns_pf = perframe.get("turns", [])
    goal = perframe.get("goal", {})
    lm_ctx = landmark.get("scene_context", {})
    lm_goal = landmark.get("goal_landmark", {})

    # v208: Path-only room/landmark defaults calibrated from GT analysis.
    # GT val_seen: hallway/corridor/hall appear in 61.2% of instructions.
    # Target: ~25% hallway-type per room pick → P(≥1 hallway in ~4 refs) ≈ 68%, close to GT 61.2%.
    # Bug fix: no "the " prefix on start_lm (pass_verb already includes "the").
    _PATH_ONLY_ROOMS = ["room", "room", "room", "hallway"]           # 25% hallway
    _PATH_ONLY_LMS = ["room", "room", "hallway", "corridor", "room"] # 40% hallway-type (slightly higher for landmarks)
    _PATH_ONLY_STOP = ["room", "room", "hallway", "corridor", "room"] # for stop_lm

    if not _has_context:
        start_room = rng.choice(_PATH_ONLY_ROOMS)
        # No "the " prefix: _pass_verb already includes "the" (e.g., "Walk past the")
        start_lm = rng.choice(_PATH_ONLY_LMS)
        start_extra = []
        goal_room = rng.choice(_PATH_ONLY_ROOMS)
        stop_lm = rng.choice(_PATH_ONLY_STOP)
        stop_desc = ""
        lm_dir_hint = ""
    else:
        start_room = (start.get("room") or "room").lower()
        start_lm = (
            _clean_lm(start.get("main_landmark"))
            or (_clean_lm_list(start.get("landmarks") or []) + [start_room])[0]
        )
        start_extra = _clean_lm_list(start.get("landmarks") or [])[:2]
        goal_room = (goal.get("room") or "room").lower()
        _raw_stop_lm = (
            _clean_lm(goal.get("stop_landmark"))
            or _clean_lm(lm_goal.get("stop_landmark"))
            or (_clean_lm_list(goal.get("landmarks") or []) + [goal_room])[0]
        )
        # v214: upgrade floor features and pool/billiard tables to better stop markers
        # when architectural alternatives (window, doorway, glass door) are available.
        _stop_candidates = (
            _clean_lm_list(goal.get("landmarks") or [])
            + _clean_lm_list(lm_goal.get("landmarks") or [])
        )
        stop_lm = _upgrade_stop_lm(_raw_stop_lm, _stop_candidates)
        stop_desc = goal.get("stop_description", "")
        lm_dir_hint = lm_ctx.get("direction_hint", "")

        # v214: Same start/goal room on long path → gate3 VLM detected wrong goal room.
        # When path_dist > 4m, an R2R episode always crosses room boundaries.
        # If the VLM reports start_room == goal_room for a long path, the goal room is wrong:
        # the model would stop in the START room instead of the actual destination.
        # Fix: override goal/stop to path-only defaults (generic room vocabulary).
        # Keep start data (correctly detected) but use generic stop.
        # v217: Exclude bedroom variants from Rule 3. Bedroom→bedroom paths are ambiguous:
        # VLM can misidentify closet/bathroom as bedroom, but "bed" is still a useful visual
        # anchor (on-path or legitimate bedroom suite). Bathroom/kitchen same-room is unambiguous.
        _RULE3_ROOMS = CLOSED_ROOMS - {"bedroom", "guest room", "master bedroom"}
        _SPECIFIC_ROOMS = _RULE3_ROOMS | STAIR_ROOMS | {"game room", "billiard room", "drawing room"}
        _same_room_long_path = (
            start_room.lower() in _SPECIFIC_ROOMS
            and start_room.lower() == goal_room.lower()
            and total_dist > 4.0
        )
        if _same_room_long_path:
            goal_room = rng.choice(_PATH_ONLY_ROOMS)
            # v215: prefer a door/arch candidate from goal frame instead of pure generic
            _arch_override = _best_arch_candidate(_stop_candidates)
            stop_lm = _arch_override if _arch_override else rng.choice(_PATH_ONLY_STOP)
            stop_desc = ""
            lm_dir_hint = ""

    # Merge perframe turn landmarks with path-analyzer turn directions (path is authoritative)
    turns = []
    for i, prim in enumerate(turn_prims):
        direction = "left" if prim["type"] == "left_turn" else "right"
        pf_t = turns_pf[i] if i < len(turns_pf) else {}
        trans = pf_t.get("room_transition", "")
        if _clean_lm(trans) is None:
            trans = ""  # filter "none visible" room transitions
        turns.append({
            "direction": direction,
            "sharp": prim.get("sharp", False),
            "angle": prim["angle_deg"],
            "landmark": _clean_lm(pf_t.get("main_landmark")),
            "extra": _clean_lm_list(pf_t.get("landmarks") or [])[:2],
            "room": (pf_t.get("room") or "").lower(),
            "room_transition": trans,
        })

    sentences = []

    # v216: Track previous room across turn handling (defined early — used in both opening and turn sections)
    _prev_room = [start_room]

    # ── v212: Initial turn phrase from start_rotation ─────────────────────────
    # 82% of episodes have misalignment >20°; without this the agent walks away from path immediately.
    # Merge into first sentence opening rather than standalone so sentence count stays calibrated.
    _init_turn_prefix = ""  # prepended to first sentence verb if set
    if init_turn:
        td = init_turn["direction"]
        angle = init_turn["angle_deg"]
        if init_turn["is_around"]:  # ≥150°
            _init_turn_prefix = "Turn around and "
        elif angle >= 90:
            _init_turn_prefix = f"Turn {td} and "
        elif angle >= 45:
            _init_turn_prefix = f"Turn {td} and "
        else:  # 20-45°: gentle re-orientation, use "face" language
            _init_turn_prefix = f"Turn slightly {td} and "

    # ── Helper: pick a room transition phrase ────────────────────────────────
    def _room_trans(room_trans_str: str, next_room: str) -> str:
        """Convert room_transition text to a brief continuation sentence (v205: diverse verbs)."""
        if room_trans_str and "entering" in room_trans_str.lower():
            m = re.search(r"entering\s+(.+?)(?:\s+area)?$", room_trans_str, re.I)
            if m:
                verb = rng.choice(["Walk", "Head", "Go"])
                return f"{verb} into the {m.group(1).strip()}."
        if next_room:
            prep = rng.choice(["into", "through", "toward"])
            verb = rng.choice(["Walk", "Head", "Go"])
            return f"{verb} {prep} the {next_room}."
        return ""

    # ── Sentence 1: Opening movement ──────────────────────────────────────────

    if _is_stair(start_room):
        stair_verb = rng.choice(["Walk up", "Go up", "Climb up", "Head up"])
        if n_turns == 0:
            sentences.append(f"{stair_verb} the stairs to the top.")
        else:
            sentences.append(f"{stair_verb} the {start_lm}.")

    elif n_turns == 0:
        # v208: 0-turn paths — reduce 1-sentence merge probability for better word count.
        # GT val_seen: 0-turn paths average 23.8 words and 2.5 sentences, NOT 1 sentence.
        # path-only: reduce merge from 55%→20% to generate longer, richer instructions.
        # gate3: keep original 55% merge (already has landmarks for word count).
        _merge_prob = 0.20 if not _has_context else 0.55
        if total_dist < 8.0 and rng.random() < _merge_prob:
            # Generate single merged sentence: "Walk through the X and stop near the Y."
            verb = rng.choice(["Walk", "Go", "Head"])
            if _is_closed(start_room):
                move_part = f"Exit the {start_room}"
            elif not _has_context:
                # Path-only: always use room preposition (start_lm has "the" prefix)
                move_part = f"{verb} through the {start_room}"
            elif stop_lm and stop_lm != start_room and rng.random() < 0.50:
                move_part = f"{verb} past the {start_lm}"
            else:
                move_part = f"{verb} through the {start_room}"
            stop_part = _stop_phrase(rng, stop_lm, goal_room)
            merged = f"{move_part} and {stop_part[0].lower()}{stop_part[1:]}"
            sentences.append(merged)
            result = " ".join(sentences)
            if _init_turn_prefix and result:
                result = _init_turn_prefix + result[0].lower() + result[1:]
            return result  # single sentence — skip rest of builder
        # Straight path: aim for 2-3 sentences with explicit forward language
        r = rng.random()
        if not _has_context:
            # v211: Path-only 0-turn — enrich opening with destination context instead of "straight".
            # "straight" drives forward% excess; destination phrasing adds words without forward-vocab.
            # e.g. "Walk through the room and into the hallway." vs just "Walk through the room."
            verb = rng.choice(["Walk", "Go"])
            path_verb = rng.choice(["through", "along", "down"])
            # v211: Enrich 0-turn opening with destination or continuation (+2-4 words)
            if goal_room and goal_room != start_room and rng.random() < 0.55:
                # Different rooms: add destination phrase
                dest_conn = rng.choice(["and into", "and toward", "toward", "into"])
                sentences.append(f"{verb} {path_verb} the {start_room} {dest_conn} the {goal_room}.")
            elif rng.random() < 0.35:
                # Same rooms: add continuation phrase (no "forward" to avoid vocab excess)
                cont = rng.choice(["and continue", "and keep going", "and keep walking"])
                sentences.append(f"{verb} {path_verb} the {start_room} {cont}.")
            else:
                sentences.append(f"{verb} {path_verb} the {start_room}.")
            # v210/v211: intermediate for long 0-turn paths — mixed forward/room vocabulary
            if total_dist > 5.0 and rng.random() < 0.40:
                int_rm = rng.choice(_PATH_ONLY_LMS)
                fwd_phrase = rng.choice([
                    f"Continue straight ahead through the {goal_room}.",
                    f"Walk straight forward into the {goal_room}.",
                    f"Keep walking forward toward the {goal_room}.",
                    f"Continue through the {goal_room}.",
                    f"Walk through the {int_rm} and continue.",
                    f"Keep going through the {int_rm}.",
                ])
                sentences.append(fwd_phrase)
            # v211: Add arrival sentence for very long 0-turn paths (+5-8 words, 30% prob)
            if total_dist > 8.0 and rng.random() < 0.30:
                arrival = rng.choice([
                    f"You will reach the {goal_room} at the end.",
                    f"Continue until you reach the {goal_room}.",
                    f"The {goal_room} is at the end of the path.",
                    f"Keep going until you arrive at the {goal_room}.",
                ])
                sentences.append(arrival)
        elif r < 0.35 and start_lm != start_room:
            pass_v = _start_pass_verb(start_lm, rng, capitalized=True)
            prep = rng.choice(["and continue through", "and walk into", "toward"])
            sentences.append(f"{pass_v} {start_lm} {prep} the {goal_room}.")
        elif r < 0.65:
            verb = rng.choice(["Walk", "Go"])
            sentences.append(f"{verb} through the {start_room} toward the {goal_room}.")
        elif r < 0.82:
            sentences.append(f"Go straight through the {start_room} and into the {goal_room}.")
        else:
            pass_v = _start_pass_verb(start_lm, rng, capitalized=True)
            sentences.append(f"{pass_v} {start_lm}.")
        # intermediate only for long paths with start_extra available (gate3 mode)
        if _has_context and total_dist > 5.0 and start_extra and rng.random() < 0.45:
            cont_lm = rng.choice(start_extra)
            pass_v2 = _pass_verb(rng, capitalized=True)
            sentences.append(f"{pass_v2} {cont_lm}.")

    elif n_turns == 1 and _is_closed(start_room) and rng.random() < 0.50:
        # Classic exit-and-turn: "Exit the bedroom and turn left [into the hallway]."
        # v219: non-sharp turns with known destination room → prefer implicit movement language.
        exit_verb = rng.choice(["Exit", "Leave", "Walk out of"])
        td = turns[0]["direction"]
        t0_room = (turns[0].get("room") or "").lower()
        is_sharp_t0 = turns[0].get("sharp", False)
        room_entry = f" into the {t0_room}" if (_has_context and t0_room and t0_room != start_room.lower()) else ""
        _use_implicit_exit = _has_context and t0_room and t0_room != start_room.lower() and not is_sharp_t0
        # If post-turn straight is long, merge continuation into s1 or add separately
        if last_straight_m > 4.0 and goal_room and rng.random() < 0.5:
            if _use_implicit_exit and rng.random() < 0.60:
                move_v = rng.choice(["walk", "head", "go"])
                sentences.append(f"{exit_verb} the {start_room} and {move_v} into the {t0_room}.")
            else:
                sentences.append(f"{exit_verb} the {start_room} and turn {td}{room_entry}.")
            cont_prep = rng.choice(["into", "through", "toward"])
            sentences.append(f"Walk {cont_prep} the {goal_room}.")
        elif last_straight_m > 4.0:
            if _use_implicit_exit and rng.random() < 0.60:
                move_v = rng.choice(["walk", "head", "go"])
                cont_prep = rng.choice(["and continue into", "and walk into", "and head into"])
                sentences.append(f"{exit_verb} the {start_room} and {move_v} into the {t0_room}.")
            else:
                cont_prep = rng.choice(["and continue into", "and walk into", "and head into"])
                sentences.append(f"{exit_verb} the {start_room} and turn {td} {cont_prep} the {goal_room}.")
        else:
            if _use_implicit_exit and rng.random() < 0.60:
                move_v = rng.choice(["walk", "head", "go"])
                sentences.append(f"{exit_verb} the {start_room} and {move_v} into the {t0_room}.")
            else:
                sentences.append(f"{exit_verb} the {start_room} and turn {td}{room_entry}.")
        turns[0]["_handled"] = True
        if _has_context and t0_room:
            _prev_room[0] = t0_room

    elif n_turns == 1 and rng.random() < 0.42:
        # v206f: 1-sentence merge for n_turns==1 — contributes to GT 19% 1-sent target.
        # Generates: "Walk past the X, turn Y[at Z], and stop near the W."
        td = turns[0]["direction"]
        lm_t = turns[0].get("landmark")
        turn_frag = f"turn {td} at the {lm_t}" if lm_t and rng.random() < 0.50 else f"turn {td}"
        stop_raw = _stop_phrase(rng, stop_lm, goal_room)
        stop_frag = stop_raw[0].lower() + stop_raw[1:-1]  # lowercase + strip trailing period
        if start_lm and start_lm != start_room:
            pv = _start_pass_verb(start_lm, rng, capitalized=False)
            opening = f"{pv} {start_lm}"
        elif _is_closed(start_room):
            ev = rng.choice(["Exit", "Leave"])
            opening = f"{ev} the {start_room}"
        else:
            vb = rng.choice(["Walk through", "Go through", "Walk past"])
            opening = f"{vb} the {start_room}"
        sentences.append(f"{opening}, {turn_frag}, and {stop_frag}.")
        turns[0]["_handled"] = True
        result = " ".join(sentences)
        if _init_turn_prefix and result:
            result = _init_turn_prefix + result[0].lower() + result[1:]
        return result  # single sentence — done

    elif n_turns >= 1:
        r = rng.random()
        if r < 0.28 and start_lm != start_room:
            # v206e: merge back at 28% (lower merge→more sents). Only walk_past family.
            # v215: door/arch landmarks use "walk through" instead of "walk past".
            # v216: add room entry context when the merged first turn leads to a new room.
            # v219: implicit movement when non-sharp turn leads to known destination room.
            td = turns[0]["direction"]
            t0_room = (turns[0].get("room") or "").lower()
            is_sharp_t0 = turns[0].get("sharp", False)
            room_entry = f" into the {t0_room}" if (_has_context and t0_room and t0_room != start_room.lower()) else ""
            _use_implicit_r28 = _has_context and t0_room and t0_room != start_room.lower() and not is_sharp_t0
            if _use_implicit_r28 and rng.random() < 0.55:
                # Implicit: "Walk through the door into the kitchen" / "Walk past the sofa into the living room"
                if _is_through_lm(start_lm):
                    verb = rng.choice(["Walk through", "Go through"])
                    prep = rng.choice(["into", "through"])
                    sentences.append(f"{verb} the {start_lm} {prep} the {t0_room}.")
                else:
                    verb = rng.choice(["Walk past", "Walk straight past", "Walk past"])
                    lm_part = start_lm if rng.random() < 0.7 else start_room
                    prep = rng.choice(["into", "toward"])
                    sentences.append(f"{verb} the {lm_part} {prep} the {t0_room}.")
            else:
                if _has_context and _is_through_lm(start_lm):
                    verb = rng.choice(["Walk through", "Go through"])
                    sentences.append(f"{verb} the {start_lm} and turn {td}{room_entry}.")
                else:
                    verb = rng.choice(["Walk past", "Walk straight past", "Walk past", "Walk past"])
                    lm_part = start_lm if rng.random() < 0.7 else start_room
                    sentences.append(f"{verb} the {lm_part} and turn {td}{room_entry}.")
            turns[0]["_handled"] = True
            # Update _prev_room so subsequent turns see the correct previous room
            if t0_room:
                _prev_room[0] = t0_room
        elif r < 0.55:
            pass_v = _start_pass_verb(start_lm, rng, capitalized=True)
            # v205: enriched with destination context ~65% of time (+3 words)
            if goal_room and goal_room != start_room and rng.random() < 0.65:
                prep = rng.choice(["toward", "into", "and into"])
                sentences.append(f"{pass_v} {start_lm} {prep} the {goal_room}.")
            elif start_extra and rng.random() < 0.35:
                extra_lm = rng.choice(start_extra)
                sentences.append(f"{pass_v} {start_lm} and {_pass_verb(rng)} {extra_lm}.")
            else:
                sentences.append(f"{pass_v} {start_lm}.")
        elif r < 0.68:
            verb = rng.choice(["Walk", "Go"])
            if _is_closed(start_room):
                prep = rng.choice(["through", "out of", "out of"])
            else:
                prep = rng.choice(["through", "down", "along"])
            # v205: add destination room to complete the sentence (+3 words, ~75%)
            if goal_room and goal_room != start_room and rng.random() < 0.75:
                sentences.append(f"{verb} {prep} the {start_room} toward the {goal_room}.")
            else:
                sentences.append(f"{verb} {prep} the {start_room}.")
        elif r < 0.82:
            if _is_closed(start_room):
                exit_verb = rng.choice(["Exit", "Leave", "Walk out of"])
                td = turns[0]["direction"]
                t0_room = (turns[0].get("room") or "").lower()
                is_sharp_t0 = turns[0].get("sharp", False)
                room_entry = f" into the {t0_room}" if (_has_context and t0_room and t0_room != start_room.lower()) else ""
                # v219: implicit exit for non-sharp turns with known destination room
                if _has_context and t0_room and t0_room != start_room.lower() and not is_sharp_t0 and rng.random() < 0.60:
                    move_v = rng.choice(["walk", "head", "go"])
                    sentences.append(f"{exit_verb} the {start_room} and {move_v} into the {t0_room}.")
                else:
                    sentences.append(f"{exit_verb} the {start_room} and turn {td}{room_entry}.")
                turns[0]["_handled"] = True
                if t0_room:
                    _prev_room[0] = t0_room
            elif rng.random() < 0.55:
                # v206f: exit for open rooms — GT exit=18-25% applies to ALL room types
                # ("Exit the living room and turn left" is common in GT R2R)
                exit_verb = rng.choice(["Exit", "Leave", "Walk out of"])
                td = turns[0]["direction"]
                t0_room = (turns[0].get("room") or "").lower()
                is_sharp_t0 = turns[0].get("sharp", False)
                # v219: implicit movement when non-sharp + known t0_room
                if _has_context and t0_room and t0_room != start_room.lower() and not is_sharp_t0 and rng.random() < 0.55:
                    move_v = rng.choice(["walk", "head", "go"])
                    sentences.append(f"{exit_verb} the {start_room} and {move_v} into the {t0_room}.")
                elif goal_room and goal_room != start_room and rng.random() < 0.5:
                    cont_prep = rng.choice(["into", "toward", "through"])
                    sentences.append(f"{exit_verb} the {start_room} and turn {td} {cont_prep} the {goal_room}.")
                elif _has_context and t0_room and t0_room != start_room.lower():
                    sentences.append(f"{exit_verb} the {start_room} and turn {td} into the {t0_room}.")
                else:
                    sentences.append(f"{exit_verb} the {start_room} and turn {td}.")
                turns[0]["_handled"] = True
                if t0_room:
                    _prev_room[0] = t0_room
            else:
                pass_v = _start_pass_verb(start_lm, rng, capitalized=True)
                fwd_cont = rng.choice(["and keep going", "and proceed forward", "and continue"])
                sentences.append(f"{pass_v} {start_lm} {fwd_cont}.")
        else:
            pass_v = _start_pass_verb(start_lm, rng, capitalized=True)
            if start_extra:
                extra_lm = rng.choice(start_extra)
                sentences.append(f"{pass_v} {start_lm} and {_pass_verb(rng)} {extra_lm}.")
            else:
                sentences.append(f"{pass_v} {start_lm}.")

    # ── Turn sentences — v205: aggressive compression into fewest sentences ──────
    #
    # Strategy: pack multiple turns into single sentences using ", then turn" connectors.
    # Target: 1-turn→1 sent, 2-turn→1-2 sents, 3+-turn→2 sents max.

    # Build list of unhandled turns
    unhandled = [t for t in turns if not t.get("_handled")]

    if unhandled:
        # Helper: build a "turn X [at Y]" fragment (lowercase, no period)
        def _turn_frag(t: dict, capital: bool = False) -> str:
            td = t["direction"]
            lm = t.get("landmark")
            if lm and rng.random() < 0.45:
                part = f"turn {td} at the {lm}"
            else:
                part = f"turn {td}"
            return part[0].upper() + part[1:] if capital else part

        # v209: Path-only enriched turn fragment — adds continuation verb + room context.
        # GT turn sentences average 8-10 words; path-only was generating only 2-3 words per turn.
        # "turn left and continue through the hallway" = +5 words vs "turn left" → +4 avg words total.
        # GT vocabulary: "continue" (10.4%), "walk" (34%), "go" (19%) — "proceed" is NOT GT vocab.
        # Forward% calibration: 80% room-based, 20% "forward" → combined with intermediate sentences ≈ GT 37.8%.
        def _turn_frag_rich(t: dict, capital: bool = False) -> str:
            """Enriched turn fragment for path-only mode — includes continuation verb."""
            td = t["direction"]
            lm = t.get("landmark")
            # GT-calibrated verbs: walk/go/continue only (no "proceed" — rare in GT)
            cont_verb = rng.choice(["continue", "walk", "go", "walk", "continue"])  # weight toward GT vocab
            cont_prep = rng.choice(["through", "into", "along", "down", "toward"])
            next_rm = rng.choice(_PATH_ONLY_ROOMS)
            if lm and rng.random() < 0.45:
                part = f"turn {td} at the {lm} and {cont_verb} {cont_prep} the {next_rm}"
            elif rng.random() < 0.80:  # 80% room-based continuation (GT-calibrated: forward% ≈ 37.8%)
                part = f"turn {td} and {cont_verb} {cont_prep} the {next_rm}"
            else:
                # 20% "forward" variant — contributes ~15% to forward%, balanced with intermediate sents
                part = f"turn {td} and {cont_verb} forward"
            return part[0].upper() + part[1:] if capital else part

        # Helper: build continuation clause for last turn (v206c: 10% continue to hit GT=10.4%)
        def _last_cont_clause() -> str:
            # v206f: removed "head" from verb choices (head_toward was 10% vs GT 1.5%)
            last_t = unhandled[-1]
            next_room = last_t.get("room", "")
            if goal_room and goal_room != next_room:
                prep = rng.choice(["into", "through", "toward"])
                verb = "continue" if rng.random() < 0.10 else rng.choice(["walk", "go"])
                return f"{verb} {prep} the {goal_room}"
            elif last_straight_m > 2.0:
                dest = stop_lm if stop_lm != goal_room and rng.random() < 0.4 else goal_room
                verb = rng.choice(["walk", "go"])
                return f"{verb} toward the {dest}"
            return ""

        # v218: Build turn fragment with room entry context — GT-aligned implicit language.
        # GT instructions use "walk into the kitchen" (implicit) far more than "turn left into the kitchen" (explicit).
        # GT avg explicit turns per instruction: 0.66; v217 had 2.04 (3× over-specified).
        # Fix: for non-sharp room-change turns, prefer implicit movement verb ("walk into") over
        # explicit "turn {direction}". Sharp turns (>75°) keep explicit because direction cue is critical.
        # This reduces explicit turns while matching GT vocabulary distribution.
        def _turn_frag_with_room(t: dict, prev_room: str, capital: bool = False) -> str:
            """Gate3-mode turn fragment. v218: implicit movement for non-sharp room-change turns."""
            td = t["direction"]
            lm = t.get("landmark")
            turn_room = (t.get("room") or "").lower().strip()
            room_changed = turn_room and turn_room != prev_room.lower().strip()
            is_sharp = t.get("sharp", False)  # angle > 75° — needs explicit direction
            r = rng.random()
            if room_changed and r < 0.70:
                # v218: Non-sharp turns → 65% implicit ("walk into the room") for GT-style alignment.
                # Sharp turns always use explicit direction — agent can't infer a 90°+ turn from room name alone.
                use_implicit = (not is_sharp) and (rng.random() < 0.65)
                if use_implicit:
                    verb = rng.choice(["walk", "go", "head", "walk", "go"])  # weight walk/go
                    if lm and _is_through_lm(lm) and rng.random() < 0.45:
                        # Door/arch landmark → "walk through the doorway into the kitchen"
                        part = f"{verb} through the {lm} into the {turn_room}"
                    else:
                        prep = rng.choice(["into", "through"])
                        part = f"{verb} {prep} the {turn_room}"
                else:
                    # Explicit: "turn left [at the X] into the room"
                    if lm and rng.random() < 0.40:
                        part = f"turn {td} at the {lm} into the {turn_room}"
                    else:
                        prep = rng.choice(["into", "through"])
                        part = f"turn {td} {prep} the {turn_room}"
            elif lm and r < 0.45:
                part = f"turn {td} at the {lm}"
            else:
                part = f"turn {td}"
            return part[0].upper() + part[1:] if capital else part

        n_un = len(unhandled)

        # v216: Use room-aware turn fragments in gate3 mode; path-only stays with rich frags.
        # _prev_room already defined at top of sentences section — reuse it (tracks across opening+turns).
        def _frag_fn_gate3(t: dict, capital: bool = False) -> str:
            frag = _turn_frag_with_room(t, _prev_room[0], capital)
            _prev_room[0] = (t.get("room") or _prev_room[0]).lower()
            return frag

        # v209: For path-only mode, use richer turn fragments with continuation verbs.
        # Gate3 mode uses room-aware fragments (v216).
        _frag_fn = _turn_frag_rich if not _has_context else _frag_fn_gate3

        if n_un == 1:
            # Single turn: merge with continuation 75% of time
            t = unhandled[0]
            cont_clause = _last_cont_clause()
            if cont_clause and rng.random() < 0.75:
                frag = _frag_fn(t, capital=True)
                sentences.append(f"{frag} and {cont_clause}.")
            else:
                sentences.append(_turn_phrase(rng, t["direction"], t["landmark"], t["sharp"]))
                if cont_clause:
                    cont_verb = rng.choice(["Walk", "Head", "Go"])
                    cont_prep = rng.choice(["into", "through", "toward"])
                    # Only add if goal_room differs
                    if goal_room and goal_room != t.get("room", ""):
                        sentences.append(f"{cont_verb} {cont_prep} the {goal_room}.")

        elif n_un == 2:
            # Two turns: 65% merge both into one sentence (gate3) | 80% merge for path-only
            # v210 fix: path-only rich frags already include room continuation — skip redundant cont_clause
            # to avoid "walk toward the room and walk toward the room" repetition bug.
            _merge_prob = 0.80 if not _has_context else 0.65
            if rng.random() < _merge_prob:
                frag1 = _frag_fn(unhandled[0], capital=True)
                frag2 = _frag_fn(unhandled[1])
                if not _has_context:
                    # Path-only: rich frags already have continuation — no cont_clause needed
                    sentences.append(f"{frag1}, then {frag2}.")
                else:
                    cont_clause = _last_cont_clause()
                    if cont_clause:
                        sentences.append(f"{frag1}, then {frag2} and {cont_clause}.")
                    else:
                        sentences.append(f"{frag1}, then {frag2}.")
            else:
                # First turn standalone, second merged with continuation
                t1, t2 = unhandled
                t1_next_room = t2.get("room") or None
                sentences.append(_turn_phrase(rng, t1["direction"], t1["landmark"], t1["sharp"],
                                              next_room=t1_next_room))
                if not _has_context:
                    # Path-only: rich frag2 already has continuation
                    frag2 = _frag_fn(t2, capital=True)
                    sentences.append(f"{frag2}.")
                else:
                    cont_clause = _last_cont_clause()
                    if cont_clause and rng.random() < 0.80:
                        frag2 = _frag_fn(t2, capital=True)
                        sentences.append(f"{frag2} and {cont_clause}.")
                    else:
                        sentences.append(_turn_phrase(rng, t2["direction"], t2["landmark"], t2["sharp"]))

        else:
            # 3+ turns: pack into max 2 sentences
            mid_turns = unhandled[:-1]
            frags = [_frag_fn(mid_turns[0], capital=True)]
            for t in mid_turns[1:]:
                frags.append(f"then {_frag_fn(t)}")
            sentences.append(", ".join(frags) + ".")
            # Last sentence: final turn + continuation
            # v210 fix: path-only rich frags already include continuation — skip redundant cont_clause
            last_t = unhandled[-1]
            if not _has_context:
                frag_last = _frag_fn(last_t, capital=True)
                sentences.append(f"{frag_last}.")
            else:
                cont_clause = _last_cont_clause()
                if cont_clause and rng.random() < 0.80:
                    frag_last = _frag_fn(last_t, capital=True)
                    sentences.append(f"{frag_last} and {cont_clause}.")
                else:
                    sentences.append(_turn_phrase(rng, last_t["direction"], last_t["landmark"], last_t["sharp"]))

    # ── v210: Intermediate sentences for path-only multi-turn paths ─────────────────
    # v210: Mixed forward/room vocabulary to calibrate forward% closer to GT 37.8%.
    # Previous v208/v209 used 100% forward phrases → forward%=45.9% (GT=37.8%).
    # v210: 50% forward vocabulary / 50% room-based vocabulary in continuation sentences.
    # Also added room-based n_turns==0 variant and stronger continuation coverage.
    if not _has_context and n_turns >= 1:
        if n_turns >= 2 and rng.random() < 0.42:  # slight increase to maintain word count
            # v210: 4 forward + 3 room-based = ~57% forward (was 100%)
            cont_rm = rng.choice(_PATH_ONLY_ROOMS)
            fwd = rng.choice([
                "Continue walking forward.",
                "Walk straight ahead.",
                "Keep going forward.",
                "Continue forward.",
                f"Continue through the {cont_rm}.",
                f"Walk through the {cont_rm}.",
                f"Keep going down the {cont_rm}.",
            ])
            sentences.append(fwd)
        elif n_turns == 1 and last_straight_m > 4.0 and rng.random() < 0.32:  # slightly raised
            next_room = goal_room if goal_room != start_room else rng.choice(_PATH_ONLY_LMS)
            # v210: 3 forward + 3 room-based = 50% forward each
            fwd = rng.choice([
                f"Walk straight ahead toward the {next_room}.",
                f"Continue forward into the {next_room}.",
                f"Walk forward through the {next_room}.",
                f"Continue through the {next_room}.",
                f"Walk through the {next_room}.",
                f"Keep going into the {next_room}.",
            ])
            sentences.append(fwd)

    # ── Elevation handling ─────────────────────────────────────────────────────
    # v211: Increase stair probability 30%→70% for detected elevation.
    # GT val_seen uses stairs in 32.3% of episodes; our elevation detection covers 15.4%.
    # At 70%, we get 15.4% * 70% ≈ 10.8% stair mentions (was 3.5%).
    # Added richer stair phrases (+2-4 words vs "Go down the stairs.").

    if elev_prims and not _is_stair(start_room):
        if rng.random() < 0.70:  # v211: 30%→70%
            direction = "up" if elev_prims[0]["direction"] == "up" else "down"
            stair_phrases = [
                f"Go {direction} the stairs.",
                f"Walk {direction} the stairs.",
                f"Take the stairs {direction}.",
                f"Head {direction} the stairs.",
            ]
            if direction == "up":
                stair_phrases += [
                    f"Go up the stairs to the next floor.",
                    f"Walk up the staircase.",
                    f"Take the stairs up to the upper level.",
                ]
            else:
                stair_phrases += [
                    f"Go down the stairs to the lower level.",
                    f"Walk down the staircase.",
                    f"Take the stairs down.",
                ]
            sentences.append(rng.choice(stair_phrases))

    # ── Optional walk-to-destination before stop (v205: ~5% of episodes, GT=5.1%) ──

    if rng.random() < 0.05 and len(sentences) <= 3:
        walk_to_sent = _walk_to_dest(rng, stop_lm if stop_lm != goal_room else goal_room)
        sentences.append(walk_to_sent)

    # ── Final stop sentence ────────────────────────────────────────────────────
    # v210: Enrich stop sentences for path-only (GT stop sentences avg ~9 words; ours was ~5.5).
    # Adding arrival context ("at the end of the hallway", "when you get there") to 70% of stops.
    # Separate paths for path-only vs gate3 to avoid disrupting gate3 landmark-driven sentences.
    if not _has_context:
        r_verb = rng.random()
        verb = "Stop" if r_verb < 0.60 else "Wait"
        r_prep = rng.random()
        if r_prep < 0.25:
            prep = f"near the {stop_lm}"
        elif r_prep < 0.45:
            prep = f"in front of the {stop_lm}"
        elif r_prep < 0.65:
            prep = f"at the {stop_lm}"
        elif r_prep < 0.83:
            prep = f"by the {stop_lm}"
        else:
            prep = f"next to the {stop_lm}"
        # v210: 70% of path-only stops get arrival/context suffix (+3-5 words)
        if rng.random() < 0.70:
            r_ctx = rng.random()
            if stop_lm != goal_room and r_ctx < 0.35:
                arrival = rng.choice([
                    f"at the end of the {goal_room}",
                    f"when you reach the {goal_room}",
                    f"once you enter the {goal_room}",
                    f"after entering the {goal_room}",
                ])
                stop_sent = f"{verb} {prep} {arrival}."
            elif r_ctx < 0.60:
                # v210: removed "hallway" from generic arrivals (was driving hallway% too high)
                arrival = rng.choice([
                    "when you reach the end",
                    "at your destination",
                    "when you get there",
                    "once you have arrived",
                    "at the end of the path",
                ])
                stop_sent = f"{verb} {prep} {arrival}."
            else:
                # Directional hint
                side = rng.choice(["on your right", "on your left", "in front of you", "ahead"])
                stop_sent = f"{verb} {prep} {side}."
        else:
            stop_sent = f"{verb} {prep}."
        sentences.append(stop_sent)
    else:
        stop_sent = _stop_phrase(rng, stop_lm, goal_room)
        sentences.append(stop_sent)

    result = " ".join(sentences)

    # v212: Prepend initial turn to first sentence (agent start orientation fix).
    # Capitalizes prefix, lowercases first char of original first sentence.
    if _init_turn_prefix and result:
        first_sent_end = result.find(".")
        if first_sent_end == -1:
            first_sent_end = len(result)
        first_sent = result[:first_sent_end + 1]
        rest = result[first_sent_end + 1:]
        # Merge: "Turn left and " + lowercase first letter of sentence
        merged_first = _init_turn_prefix + first_sent[0].lower() + first_sent[1:]
        result = (merged_first + rest).strip()

    # Trim if excessively long (>50 words)
    if len(result.split()) > 50:
        keep = [sentences[0]]
        for s in sentences[1:]:
            if any(w in s for w in ["Turn", "Make a", "Take a", "left", "right"]):
                keep.append(s)
        keep.append(sentences[-1])
        result = " ".join(keep)
        # Re-apply initial turn prefix after trim
        if _init_turn_prefix and result:
            first_sent_end = result.find(".")
            if first_sent_end == -1:
                first_sent_end = len(result)
            first_sent = result[:first_sent_end + 1]
            rest = result[first_sent_end + 1:]
            result = (_init_turn_prefix + first_sent[0].lower() + first_sent[1:] + rest).strip()

    return result


# ── Batch runner ───────────────────────────────────────────────────────────────

def run_batch(version: str = "v204_meta", episode_filter: int | None = None):
    print(f"[MetadataReproducer] Loading GT dataset...")
    with gzip.open(GT_PATH, "rt") as f:
        gt = json.load(f)

    episodes = {e["episode_id"]: e for e in gt["episodes"]}
    print(f"[MetadataReproducer] {len(episodes)} GT episodes loaded.")

    results = []
    missing_perframe = 0
    vocab_counts: dict[str, int] = {}

    eids = [episode_filter] if episode_filter else sorted(episodes.keys())

    for eid in eids:
        ep = episodes.get(eid)
        if not ep:
            continue

        pf_path = PERFRAME_DIR / f"episode_{eid:06d}.json"
        lm_path = LANDMARK_DIR / f"episode_{eid:06d}.json"

        if not pf_path.exists():
            missing_perframe += 1
            # Fallback: use landmark data only
            perframe = {}
        else:
            perframe = json.loads(pf_path.read_text())

        landmark = json.loads(lm_path.read_text()) if lm_path.exists() else {}

        rng = _make_rng(eid)
        instr = reproduce_instruction(
            episode_id=eid,
            reference_path=ep["reference_path"],
            start_rotation=ep.get("start_rotation"),
            perframe=perframe,
            landmark=landmark,
            rng=rng,
        )

        # Track vocab
        for word in re.findall(r"\b\w+\b", instr.lower()):
            vocab_counts[word] = vocab_counts.get(word, 0) + 1

        # Build output episode (same format as GT)
        out_ep = dict(ep)
        out_ep["instruction"] = {
            "instruction_text": instr,
            "_source": f"metadata_reproducer:{version}",
            "_gt_instruction": ep["instruction"]["instruction_text"],
        }
        results.append(out_ep)

        if episode_filter:
            gt_instr = ep["instruction"]["instruction_text"].strip()
            print(f"\nEpisode {eid}:")
            print(f"  GT:   \"{gt_instr}\"")
            print(f"  Auto: \"{instr}\"")
            print(f"  Words: {len(instr.split())}")

    if episode_filter:
        return

    if missing_perframe:
        print(f"[MetadataReproducer] WARNING: {missing_perframe} episodes missing perframe data.")

    # Vocabulary analysis
    n_total = len(results)

    def _pct(pat, flags=re.I):
        return sum(1 for e in results if re.search(pat, e["instruction"]["instruction_text"], flags)) / n_total * 100

    avg_words = sum(len(e["instruction"]["instruction_text"].split()) for e in results) / n_total

    # Sentence count distribution
    sent_counts = {}
    for e in results:
        sents = re.split(r'(?<=[.!?])\s+', e["instruction"]["instruction_text"].strip())
        n = len(sents)
        sent_counts[n] = sent_counts.get(n, 0) + 1
    avg_sents = sum(k * v for k, v in sent_counts.items()) / n_total

    print(f"\n[MetadataReproducer] Vocabulary analysis ({n_total} episodes):")
    print(f"  avg words: {avg_words:.1f} (GT=26.8)")
    print(f"  avg sents: {avg_sents:.2f} (GT=2.50)")
    print(f"  walk_past:     {_pct(r'walk past'):.1f}% (GT=10.5%, target 20-25%)")
    print(f"  walk_toward:   {_pct(r'walk toward'):.1f}% (GT=1.3%)")
    print(f"  walk_towards:  {_pct(r'walk towards'):.1f}% (GT=4.6%)")
    print(f"  walk_to_dest:  {_pct(r'walk to the|go to the'):.1f}% (GT=5.1%+4.6%)")
    print(f"  go_past:       {_pct(r'go past'):.1f}% (GT=2.6%)")
    print(f"  pass_the:      {_pct(r'pass the'):.1f}% (GT=4.1%)")
    print(f"  continue:      {_pct('continue'):.1f}% (GT=10.4%)")
    print(f"  stop:          {_pct('\\bstop\\b'):.1f}% (GT=53.0%)")
    print(f"  wait:          {_pct('\\bwait\\b'):.1f}% (GT=32.6%)")
    print(f"  sentence distribution: {dict(sorted(sent_counts.items()))}")

    # Save output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"val_unseen_generated_meta_{version}.json.gz"
    out_data = {"episodes": results}
    with gzip.open(out_path, "wt") as f:
        json.dump(out_data, f)
    print(f"\n[MetadataReproducer] Saved: {out_path} ({n_total} episodes)")

    return out_path


# ── Quality analysis ───────────────────────────────────────────────────────────

def analyze_quality(version: str = "v204_meta"):
    """Compare generated vs GT for word-overlap metrics."""
    out_path = OUT_DIR / f"val_unseen_generated_meta_{version}.json.gz"
    if not out_path.exists():
        print(f"[analyze] Not found: {out_path}")
        return

    with gzip.open(out_path, "rt") as f:
        data = json.load(f)

    f1s = []
    for ep in data["episodes"]:
        gen = ep["instruction"]["instruction_text"].lower()
        gt = ep["instruction"].get("_gt_instruction", "").lower()
        gen_words = set(re.findall(r"\b\w+\b", gen))
        gt_words = set(re.findall(r"\b\w+\b", gt))
        if not gt_words:
            continue
        precision = len(gen_words & gt_words) / len(gen_words) if gen_words else 0
        recall = len(gen_words & gt_words) / len(gt_words)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)

    print(f"[analyze] {version}: word F1={sum(f1s)/len(f1s):.3f} over {len(f1s)} episodes")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v204_meta", help="Output version tag")
    parser.add_argument("--episode", type=int, default=None, help="Debug single episode")
    parser.add_argument("--analyze", action="store_true", help="Analyze quality of existing output")
    args = parser.parse_args()

    if args.analyze:
        analyze_quality(args.version)
    else:
        run_batch(version=args.version, episode_filter=args.episode)
        if not args.episode:
            analyze_quality(args.version)
