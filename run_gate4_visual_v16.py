#!/usr/bin/env python3
"""
Gate 4 Visual v16 — Calibrated Episode-Level Landmark Control

ROOT CAUSE ANALYSIS:
  v14: P_ANCHOR_TURN=0.22 per-turn → 34.1% episode anchor rate
  WHY: P(at least one turn gets landmark) = 1 - (1-0.22)^5.8 ~ 78%
       Of those episodes, ~44% produce anchor phrases → 78% × 44% ~ 34%

THE FIX (v15): EPISODE-LEVEL CONTROL
  - For each episode, with P_ANCHOR_EPISODE = 0.21 probability:
    provide landmark for exactly ONE turn (most distinctive)
  - All other episodes (60%): NO turn gets landmark context
  - Expected anchor rate: 40% × 44% ~ 17.6% (matching GT's 16.5%)

ADDITIONALLY:
  - Stronger system prompt: explicit "no-anchor" instruction for turns without brackets
  - Same enforce_stop_phrase() post-processing (near-perfect in v14)

Output: outputs/datasets/val_unseen_generated_gemma_visual_v16.json.gz
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
from typing import Dict, List, Tuple

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

GT_PATH    = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
G3PF_DIR   = ROOT / "outputs" / "gate3_perframe"
G3_OLD_DIR = ROOT / "outputs" / "gate3_landmarks"
OUTPUT     = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v16.json.gz"
CKPT       = ROOT / "outputs" / "gate4_visual_v16_checkpoint.json"

from gate2_path.path_analyzer import analyze_path
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode

# ── Constants ─────────────────────────────────────────────────────────────────

# Episode-level probability: 40% of episodes get ONE landmark.
# Expected anchor rate: 0.40 * 0.44 ~ 17.6% (vs GT 16.5%).
P_ANCHOR_EPISODE = 0.21
# Calibrated from v15 data:
# v15: P=0.40, anchor_rate=31.9% → P(anchor|shown) ≈ 80%
# Target: 16.5%. P_needed = 0.165 / 0.80 = 0.206 → use 0.21

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


# ── Landmark loading ──────────────────────────────────────────────────────────

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


def get_landmark_info(ep, pf_map, old_map, primitives):
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


# ── EPISODE-LEVEL landmark control (KEY v15 innovation) ───────────────────────

def episode_selective_turns(lm_info: Dict, eid: int, p_anchor: float = P_ANCHOR_EPISODE) -> List[Dict]:
    """
    Episode-level control: only p_anchor fraction of episodes get ANY landmark.
    When selected, exactly ONE turn (most distinctive) receives landmark context.
    Expected anchor rate ≈ p_anchor × 0.44 ≈ 17.6% for p_anchor=0.40.
    """
    rng = random.Random(eid ^ 0xE3B6_D14F)
    turns = lm_info.get("turns", [])

    # Default: no landmarks
    result = [
        {"direction": t["direction"], "landmark": "", "room": "", "room_trans": ""}
        for t in turns
    ]

    if rng.random() < p_anchor:
        turns_with_lm = [(i, t) for i, t in enumerate(turns) if t.get("landmark")]
        if turns_with_lm:
            # Most distinctive = longest name (e.g., "wooden coffee table" > "door")
            best_idx, best_turn = max(turns_with_lm,
                                      key=lambda x: len(x[1].get("landmark", "")))
            result[best_idx] = {
                "direction":  best_turn["direction"],
                "landmark":   best_turn["landmark"],
                "room":       best_turn["room"],
                "room_trans": best_turn["room_trans"],
            }

    return result


# ── Stop/wait — GT distribution ───────────────────────────────────────────────

def choose_stop(goal_lm: str, goal_room: str, eid: int) -> Tuple[str, str]:
    rng = random.Random(eid ^ 0xA3B7)
    r   = rng.random()
    lm  = f"the {goal_lm}" if not goal_lm.startswith("the ") else goal_lm
    if goal_lm == "the destination": lm = "your destination"

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
    # Sentence-split and remove all stop/wait directives before appending canonical
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    cleaned = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        # Drop full-sentence stop/wait directives (e.g., "Stop at X.", "Wait there.")
        if re.match(r'^(Stop|Wait|Halt|Stand)\b', s, re.IGNORECASE):
            continue
        # Strip trailing stop/wait clause. Two alternatives:
        # 1) comma-led: ", [connectors] stop [any content]"
        # 2) connector-led: " and/then/to/continue/until/once/when stop [any content]"
        #    includes optional subject word, e.g. "until you stop"
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


# ── System prompt (v15: explicit no-anchor rule) ──────────────────────────────

SYSTEM_INTRO = """You write concise R2R navigation instructions for an indoor robot.

Style: Natural and brief (18-30 words), like a human R2R annotator.

RULES:
- Use landmarks ONLY when shown in square brackets in the route
- For any turn WITHOUT brackets: write ONLY "turn left" or "turn right" — no objects, no furniture
- Do NOT invent or add objects for turns that have no brackets
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
    anchor_note = (
        f"Route has {n_bracketed} bracketed landmark(s) — use them naturally."
        if n_bracketed > 0
        else "Route has NO bracketed landmarks — write ONLY direction words for all turns."
    )

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


# ── Async generation ──────────────────────────────────────────────────────────

async def generate_one(client, task, sem, done, total, t0):
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
        print(f"  [{done[0]}/{total}] {r:.1f}/s  ETA={eta/60:.1f}m", flush=True)
    return eid, result


async def generate(tasks, concurrency=12):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
    sem    = asyncio.Semaphore(concurrency)
    done   = [0]; t0 = time.time(); total = len(tasks)
    results = {}
    for eid, result in await asyncio.gather(
        *[generate_one(client, t, sem, done, total, t0) for t in tasks]
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
    ap.add_argument("--n-episodes",       type=int,   default=None)
    ap.add_argument("--concurrency",      type=int,   default=12)
    ap.add_argument("--k-similar",        type=int,   default=8)
    ap.add_argument("--p-anchor-episode", type=float, default=P_ANCHOR_EPISODE,
                    help=f"Fraction of episodes that get ONE landmark (default {P_ANCHOR_EPISODE})")
    args = ap.parse_args()
    p_anchor = args.p_anchor_episode

    print("=== Gate 4 v16: Episode-Level Landmark Control (Calibrated) ===")
    print(f"  P_ANCHOR_EPISODE: {p_anchor} → expected anchor rate ~{p_anchor*0.44*100:.1f}% (GT=16.5%)")
    print(f"  stop/wait: {P_STOP*100:.0f}%/{P_WAIT*100:.0f}%/{(1-P_STOP-P_WAIT)*100:.0f}%")
    print(f"  concurrency: {args.concurrency}")
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

    print("Loading landmarks...")
    pf_map  = load_perframe_landmarks(G3PF_DIR)
    old_map = load_old_landmarks(G3_OLD_DIR)
    n_pf = sum(1 for ep in episodes if ep["episode_id"] in pf_map)
    print(f"  Gate 3 v2: {n_pf}/{len(episodes)} episodes")

    ckpt = {}
    if CKPT.exists():
        ckpt = json.load(open(CKPT))
        print(f"Checkpoint: {len(ckpt)} episodes done")

    tokenizer = VLNTokenizer(GT_PATH)
    tasks     = []
    task_meta = {}
    skipped   = 0
    ep_feats  = {ep["episode_id"]: extract_path_features(ep) for ep in episodes}

    for ep in episodes:
        eid      = ep["episode_id"]
        gt_instr = gt_map.get(eid,"")
        ep_feat  = ep_feats[eid]
        prims    = get_primitives(ep)
        lm_info  = get_landmark_info(ep, pf_map, old_map, prims)

        sel_turns            = episode_selective_turns(lm_info, eid, p_anchor)
        stop_phrase, stop_type = choose_stop(lm_info["goal_landmark"], lm_info["goal_room"], eid)

        if str(eid) in ckpt:
            skipped += 1
            task_meta[eid] = {
                "sv":          gt_start_verb(gt_instr),
                "goal_lm":     lm_info["goal_landmark"],
                "stop_phrase": stop_phrase,
                "stop_type":   stop_type,
            }
            continue

        examples  = sc_idx.top_k(ep, ep_feat, k=args.k_similar)
        prompt, sv = build_prompt(ep, gt_instr, examples, lm_info, prims,
                                  sel_turns, stop_phrase, stop_type)
        task_meta[eid] = {
            "sv":          sv,
            "goal_lm":     lm_info["goal_landmark"],
            "stop_phrase": stop_phrase,
            "stop_type":   stop_type,
        }
        tasks.append({"episode_id": eid, "prompt": prompt, "sv": sv})

    print(f"Tasks: {len(tasks)}  Skipped: {skipped}")

    # Pre-generation stats
    n_with_lm = 0
    for ep in episodes:
        eid   = ep["episode_id"]
        prims = get_primitives(ep)
        lm_info = get_landmark_info(ep, pf_map, old_map, prims)
        sel = episode_selective_turns(lm_info, eid, p_anchor)
        if any(t.get("landmark") for t in sel):
            n_with_lm += 1
    print(f"  Episodes with landmark shown: {n_with_lm}/{len(episodes)} = {100*n_with_lm/len(episodes):.1f}%")

    from collections import Counter
    stop_types = [task_meta[ep["episode_id"]]["stop_type"]
                  for ep in episodes if ep["episode_id"] in task_meta]
    sc = Counter(stop_types)
    n  = len(stop_types)
    print(f"  Stop/Wait/None: {sc.get('stop',0)/n*100:.1f}%/{sc.get('wait',0)/n*100:.1f}%/{sc.get('none',0)/n*100:.1f}%")
    print()

    if tasks:
        sample = tasks[0]
        print(f"--- Sample prompt (ep {sample['episode_id']}) ---")
        print(sample["prompt"][:1400])
        print("-"*60)
        print()

    if tasks:
        print("Generating...")
        results = await generate(tasks, args.concurrency)
        ckpt.update({str(k):v for k,v in results.items()})
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        with open(CKPT,"w") as f: json.dump(ckpt,f)

    # ── Assemble ──────────────────────────────────────────────────────────────
    print("\nAssembling dataset...")
    episodes_out = []
    n_pass = n_fix = n_fail = 0

    for ep in episodes:
        eid  = ep["episode_id"]
        raw  = ckpt.get(str(eid),"")
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
            text = f"{sv} to {goal_lm}."
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
    spp = {k:0 for k in ["past","through","into","at the"]}
    for t in texts:
        tl = t.lower()
        for k in spp: spp[k] += k in tl

    print(f"\n=== v16 Done ===")
    print(f"  Episodes:     {n}  Pass: {n_pass}  Fixed: {n_fix}  Failed: {n_fail}")
    print(f"  avg_words:    {avg_w:.1f}  (GT=26.8)")
    print(f"  stop%:        {100*stop_n/n:.1f}%  (GT=50.8%)")
    print(f"  wait%:        {100*wait_n/n:.1f}%  (GT=30.5%)")
    print(f"  neither%:     {100*neither_n/n:.1f}%  (GT=16.5%)")
    print(f"  turn-anchor%: {100*anchor_n/n:.1f}%  (GT=16.5%) ← TARGET")
    print(f"  spatial:")
    for k,v in spp.items():
        print(f"    {k:<12}: {100*v/n:.1f}%")
    print(f"\nSaved: {OUTPUT}")

    anchor_err = abs(anchor_n/n - 0.165)
    stop_err   = abs(stop_n/n  - 0.508)
    wait_err   = abs(wait_n/n  - 0.305)
    match_score = 1.0 - (anchor_err*2 + stop_err + wait_err)
    print(f"\n  GT-match score: {match_score:.3f}  (target ≥ 0.90; v14≈0.77)")


if __name__ == "__main__":
    asyncio.run(main())
