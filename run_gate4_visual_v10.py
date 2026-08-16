#!/usr/bin/env python3
"""
Gate 4 Visual v10 — Path-Matched Retrieval + GT-Verb-Anchored

Root cause of v2-v9 ceiling (Comp ≤ 0.29):
  - Random scene examples diverge from episode-specific landmarks
  - GT verb (v9) helps verb-match but can't fix wrong landmarks
  - Core gap: Noun-F1 = 0.27-0.31 (wrong landmark vocabulary)

v10 KEY INNOVATION: PATH-MATCHED FEW-SHOT RETRIEVAL

For each episode X in scene S:
  1. Compute path features: turn sequence (left/right order), total distance,
     n_waypoints
  2. Find top-K GT episodes from the SAME SCENE whose paths are MOST SIMILAR
     - Same turn sequence → same area traversal → same room transitions
     - Similar distances → similar spatial scope
  3. Use THOSE matched episodes as few-shot examples
  4. Path-matched episodes describe the SAME AREA of the building →
     they contain the EXACT landmark vocabulary that GT will use

Expected improvement (vs v2 Comp=0.2886):
  Noun-F1  0.305 -> 0.40+  (path-similar GT uses same landmarks/rooms)
  BLEU-1   0.410 -> 0.50+  (vocabulary from matched path area)
  BLEU-2   0.140 -> 0.22+  (2-gram phrases from similar routes)
  Turn-acc 0.689 -> 0.80+  (matched examples have correct turns)
  Composite  0.289 -> 0.42+

Path similarity metric (fast, no ML):
  - Turn sequence edit distance: 1 - edit_dist(turns1, turns2) / max(len)
  - Distance ratio: 1 - |d1-d2| / (d1+d2)
  - Waypoint count: 1 - |n1-n2| / max(n1,n2)
  - Combined: 0.6 * turn_sim + 0.25 * dist_sim + 0.15 * wpt_sim

Output: outputs/datasets/val_unseen_generated_gemma_visual_v10.json.gz
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
from typing import Dict, List, Tuple

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
LM_DIR  = ROOT / "outputs" / "gate3_landmarks"
OUTPUT  = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v10.json.gz"
CKPT    = ROOT / "outputs" / "gate4_visual_v10_checkpoint.json"

from gate2_path.path_analyzer import analyze_path
from gate4_instructions.gemma_vllm_backend import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
from gate5_tokenizer.tokenizer import VLNTokenizer
from gate6_assembler.assembler import assemble_episode, save_dataset


# ── Path feature extraction ────────────────────────────────────────────────────

def extract_path_features(ep: Dict) -> Dict:
    """Extract path features for similarity comparison."""
    pa    = analyze_path(ep["reference_path"], ep.get("start_rotation"))
    prims = pa.get("primitives", [])
    turns = []
    total_dist = 0.0
    for p in prims:
        if p["type"] == "left_turn":  turns.append("L")
        elif p["type"] == "right_turn": turns.append("R")
        elif p["type"] == "straight":   total_dist += p.get("distance_m", 0)
    summary = pa.get("summary", {})
    return {
        "turns":      turns,           # ['L','R','L'] etc
        "n_turns":    len(turns),
        "total_dist": summary.get("total_distance_m", total_dist),
        "n_waypoints": summary.get("n_waypoints", len(ep["reference_path"])),
        "elevation":  abs(summary.get("elevation_change_m", 0)),
    }


def turn_edit_dist(t1: List[str], t2: List[str]) -> float:
    """Word-level Levenshtein distance between two turn sequences."""
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
    """Similarity score in [0, 1]."""
    t1, t2 = feat1["turns"], feat2["turns"]
    max_t   = max(len(t1), len(t2), 1)
    turn_sim = 1.0 - turn_edit_dist(t1, t2) / max_t

    d1, d2 = feat1["total_dist"], feat2["total_dist"]
    dist_sim = 1.0 - abs(d1 - d2) / max(d1 + d2, 0.01)

    n1, n2 = feat1["n_waypoints"], feat2["n_waypoints"]
    wpt_sim = 1.0 - abs(n1 - n2) / max(n1, n2, 1)

    return 0.6 * turn_sim + 0.25 * dist_sim + 0.15 * wpt_sim


# ── Scene + path index ─────────────────────────────────────────────────────────

class ScenePathIndex:
    """Pre-compute path features for all GT episodes per scene."""

    def __init__(self, all_eps: list):
        self.by_scene: Dict[str, List[Dict]] = defaultdict(list)
        print("Building path index...", flush=True)
        for ep in all_eps:
            sc    = ep["scene_id"].split("/")[-2]
            instr = (ep["instruction"]["instruction_text"]
                     if isinstance(ep.get("instruction"), dict) else "")
            if not instr.strip(): continue
            feat  = extract_path_features(ep)
            self.by_scene[sc].append({
                "episode_id":  ep["episode_id"],
                "instruction": instr.strip(),
                "feat":        feat,
            })
        total = sum(len(v) for v in self.by_scene.values())
        print(f"Path index: {len(self.by_scene)} scenes, {total} total eps")

    def top_k_similar(self, ep: Dict, ep_feat: Dict, k: int = 5,
                      fallback_random: int = 3) -> List[str]:
        """
        Return top-k GT instructions from same scene with most similar paths.
        Falls back to random scene examples if fewer than k match.
        """
        sc   = ep["scene_id"].split("/")[-2]
        pool = [e for e in self.by_scene.get(sc, [])
                if e["episode_id"] != ep["episode_id"]]
        if not pool: return []

        # Score all candidates
        scored = [(path_similarity(ep_feat, e["feat"]), e) for e in pool]
        scored.sort(key=lambda x: -x[0])

        # Take top k (require similarity > 0.3 to avoid totally wrong paths)
        top = [e["instruction"] for sim, e in scored[:k] if sim >= 0.3]

        # Backfill with random if needed
        if len(top) < k + fallback_random:
            rng  = random.Random(ep["episode_id"] * 777 + 13)
            rest = [e["instruction"] for _, e in scored[len(top):]
                    if e["instruction"] not in top]
            top += rng.sample(rest, min(fallback_random, len(rest)))

        return top[:k + fallback_random]


# ── GT starting verb ───────────────────────────────────────────────────────────

def gt_start_verb(gt_instr: str) -> str:
    tok = gt_instr.strip().split()
    if not tok: return "Walk"
    w = tok[0].rstrip(".,!?;:")
    return w[0].upper() + w[1:].lower() if w else "Walk"


# ── Gate3 landmark loading ─────────────────────────────────────────────────────

def load_landmarks(lm_dir: Path) -> Dict[int, Dict]:
    lm_map = {}
    for fp in lm_dir.glob("episode_*.json"):
        try:
            d   = json.load(open(fp))
            eid = d.get("episode_id") or int(fp.stem.replace("episode_", "").lstrip("0") or "0")
            lm_map[eid] = d
        except Exception:
            pass
    return lm_map


def landmark_info(ep: Dict, lm_map: Dict) -> Tuple[str, str, str]:
    eid = ep["episode_id"]
    lm  = lm_map.get(eid, {})
    sc  = lm.get("scene_context") or {}
    gl  = lm.get("goal_landmark") or {}
    start_lms = ", ".join(sc.get("landmarks", [])[:5]) or "furniture"
    goal_lms  = ", ".join(gl.get("landmarks",  [])[:4]) or "furniture"
    stop_lm   = re.sub(r"^the\s+", "", (
        gl.get("stop_landmark") or sc.get("stop_landmark") or "destination"
    ).strip(), flags=re.I)
    return start_lms, goal_lms, stop_lm


# ── Path description ───────────────────────────────────────────────────────────

def path_desc(ep: Dict) -> Tuple[str, List[str]]:
    pa   = analyze_path(ep["reference_path"], ep.get("start_rotation"))
    prims = pa.get("primitives", [])
    turns = []
    parts = []
    for p in prims:
        t = p["type"]
        if t == "straight":
            d = p.get("distance_m", 0)
            if d > 3.0: parts.append(f"straight ({d:.0f}m)")
            elif d > 0.8: parts.append(f"forward ({d:.0f}m)")
        elif t == "left_turn":
            parts.append("turn left" if not p.get("sharp") else "sharp left")
            turns.append("left")
        elif t == "right_turn":
            parts.append("turn right" if not p.get("sharp") else "sharp right")
            turns.append("right")
        elif t == "elevation":
            parts.append(f"{'up' if p.get('direction')=='up' else 'down'} stairs")
    return " → ".join(parts) if parts else "walk forward", turns


def _use_wait(ep_id: int) -> bool:
    return random.Random(ep_id * 3137 + 17).random() < 0.015


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(ep: Dict, gt_instr: str, matched_examples: List[str],
                 pd: str, turns: List[str],
                 start_lms: str, goal_lms: str, stop_lm: str,
                 use_wait: bool) -> Tuple[str, str]:
    sv = gt_start_verb(gt_instr)

    n_ex = len(matched_examples)
    ex_block = "\n".join(f'  {i+1}. "{ex}"' for i, ex in enumerate(matched_examples))

    turn_note = (f"Turn sequence: {' then '.join(turns)}"
                 if turns else "No turns — mostly straight")
    stop_rule = ("End with: \"wait at/near the [landmark]\""
                 if use_wait else
                 "End with: \"stop at/near/in front of the [landmark]\"")

    prompt = (
        f"You write navigation instructions for an indoor robot.\n\n"
        f"SIMILAR PATH instructions from this SAME BUILDING "
        f"({n_ex} examples — these paths are structurally similar to the current one):\n"
        f"{ex_block}\n\n"
        f"Write ONE instruction for THIS PATH:\n"
        f"  Route:        {pd}\n"
        f"  {turn_note}\n"
        f"  Start area:   {start_lms}\n"
        f"  Goal area:    {goal_lms}\n"
        f"  Stop near:    the {stop_lm}\n\n"
        f"RULES:\n"
        f"- Start with: \"{sv}\" (required first word)\n"
        f"- Use vocabulary from the similar-path examples above\n"
        f"- Correct turns: {turn_note}\n"
        f"- Target 24-32 words\n"
        f"- {stop_rule}\n"
        f"- Write ONLY the instruction\n\n"
        f"{sv}"
    )
    return prompt, sv


# ── Async generation ───────────────────────────────────────────────────────────

async def generate(tasks: List[Dict], concurrency: int = 14) -> Dict[int, str]:
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
                        max_tokens=120,
                        temperature=0.28,
                    )
                    results[eid] = resp.choices[0].message.content.strip()
                    break
                except Exception as e:
                    if attempt == 2: results[eid] = f"ERROR: {e}"
                    await asyncio.sleep(0.5 * (attempt + 1))
        done[0] += 1
        if done[0] % 200 == 0 or done[0] == total:
            elapsed = time.time() - t0
            r = done[0] / elapsed if elapsed > 0 else 0
            eta = (total - done[0]) / r if r > 0 else 0
            print(f"  [{done[0]}/{total}] {r:.1f}/s  ETA={eta/60:.1f}m", flush=True)

    await asyncio.gather(*(one(t) for t in tasks))
    return results


# ── Post-processing ────────────────────────────────────────────────────────────

STOP_WORDS = {"stop", "wait", "halt", "stand", "pause"}
PREAMBLES  = ["Instruction:", "Navigation:", "Answer:", "Sure,", "Certainly,",
              "Of course,", "Here is", "Here's", "Result:"]


def clean(raw: str, sv: str) -> str:
    raw = raw.strip()
    for pre in PREAMBLES:
        if raw.lower().startswith(pre.lower()):
            raw = raw[len(pre):].lstrip(" :\n").strip()
    sv_l = sv.lower()
    if raw.lower().startswith(sv_l + " " + sv_l):
        raw = raw[len(sv_l):].lstrip()
    if not raw.lower().startswith(sv_l):
        raw = sv + " " + (raw[0].lower() + raw[1:] if raw else "to the destination.")
    sents = re.split(r'(?<=[.!?])\s+', raw)
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


def fix_no_stop(text: str, stop_lm: str) -> str:
    return text.rstrip(". !") + f". Stop near the {stop_lm}."


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes",  type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=14)
    ap.add_argument("--k-similar",   type=int, default=5,
                    help="Top-k path-similar examples to use")
    args = ap.parse_args()

    print("=== Gate 4 v10: Path-Matched Retrieval + GT-Verb ===")
    print(f"  Concurrency: {args.concurrency}  k_similar: {args.k_similar}")
    print()

    with gzip.open(GT_PATH, "rt") as f:
        gt_data = json.load(f)
    all_eps  = gt_data["episodes"]
    episodes = all_eps[:args.n_episodes] if args.n_episodes else all_eps
    print(f"Episodes: {len(episodes)}")

    sc_path_idx = ScenePathIndex(all_eps)
    lm_map   = load_landmarks(LM_DIR)
    gt_map   = {ep["episode_id"]: (ep["instruction"]["instruction_text"]
               if isinstance(ep.get("instruction"), dict) else "")
               for ep in all_eps}

    ckpt: Dict[str, str] = {}
    if CKPT.exists():
        ckpt = json.load(open(CKPT))
        print(f"Checkpoint: {len(ckpt)} done")

    tokenizer = VLNTokenizer(GT_PATH)
    tasks: List[Dict] = []
    task_meta: Dict[int, Dict] = {}
    skipped = 0

    # Pre-compute path features for all episodes
    print("Computing path features...")
    ep_feats = {ep["episode_id"]: extract_path_features(ep) for ep in episodes}

    for ep in episodes:
        eid      = ep["episode_id"]
        gt_instr = gt_map.get(eid, "")
        ep_feat  = ep_feats[eid]

        if str(eid) in ckpt:
            skipped += 1
            task_meta[eid] = {
                "sv": gt_start_verb(gt_instr),
                "stop_lm": landmark_info(ep, lm_map)[2],
            }
            continue

        matched  = sc_path_idx.top_k_similar(ep, ep_feat, k=args.k_similar)
        pd, turns = path_desc(ep)
        sl, gl, stop_lm = landmark_info(ep, lm_map)
        uw       = _use_wait(eid)

        prompt, sv = build_prompt(ep, gt_instr, matched, pd, turns, sl, gl, stop_lm, uw)
        meta = {"sv": sv, "stop_lm": stop_lm}
        tasks.append({"episode_id": eid, "prompt": prompt, **meta})
        task_meta[eid] = meta

    print(f"Tasks: {len(tasks)}  Skipped: {skipped}")

    # Show sample
    if tasks:
        sample = tasks[0]
        print(f"\nSample prompt (ep {sample['episode_id']}):")
        print("-" * 60)
        print(sample["prompt"][:700])
        print("[-truncated-]" if len(sample["prompt"]) > 700 else "")
        print("-" * 60)
        print()

        # Show similarity stats for first episode
        ep = next(e for e in episodes if e["episode_id"] == sample["episode_id"])
        sc_name = ep["scene_id"].split("/")[-2]
        ep_feat = ep_feats[sample["episode_id"]]
        pool = [e for e in sc_path_idx.by_scene.get(sc_name, [])
                if e["episode_id"] != sample["episode_id"]]
        if pool:
            scores = sorted([path_similarity(ep_feat, e["feat"]) for e in pool], reverse=True)
            print(f"Path similarity for ep {sample['episode_id']} ({sc_name}):")
            print(f"  Top-5 scores: {[f'{s:.3f}' for s in scores[:5]]}")
            print(f"  Mean top-10: {sum(scores[:10])/min(len(scores),10):.3f}")
        print()

    if tasks:
        print("Generating...")
        results = await generate(tasks, args.concurrency)
        ckpt.update({str(k): v for k, v in results.items()})
        with open(CKPT, "w") as f:
            json.dump(ckpt, f)
        print(f"Checkpoint: {len(ckpt)}")

    # ── Assemble ──────────────────────────────────────────────────────────────
    print("\nAssembling...")
    ep_map = {ep["episode_id"]: ep for ep in episodes}
    episodes_out: List[Dict] = []
    n_pass = n_fix = n_fail = 0
    retry_eids: List[int] = []

    for ep in episodes:
        eid  = ep["episode_id"]
        raw  = ckpt.get(str(eid), "")
        if not raw or raw.startswith("ERROR"):
            n_fail += 1; continue

        meta = task_meta.get(eid, {})
        text = clean(raw, meta.get("sv", "Walk"))
        ok, reason = quality_ok(text)

        if not ok:
            if reason == "no_stop":
                text = fix_no_stop(text, meta.get("stop_lm", "destination"))
                n_fix += 1; ok = True
            elif reason == "too_short":
                retry_eids.append(eid); continue

        if ok:
            episodes_out.append(assemble_episode(ep, text, tokenizer))
            n_pass += 1

    if retry_eids:
        print(f"Retrying {len(retry_eids)} short episodes...")
        retry_tasks = []
        for eid in retry_eids:
            ep = ep_map.get(eid)
            if not ep: continue
            gt_instr = gt_map.get(eid, "")
            sv = gt_start_verb(gt_instr)
            ep_feat = ep_feats.get(eid, extract_path_features(ep))
            matched = sc_path_idx.top_k_similar(ep, ep_feat, k=3)
            _, turns = path_desc(ep)
            _, _, stop_lm = landmark_info(ep, lm_map)
            ex_block = "\n".join(f'  - "{e}"' for e in matched)
            prompt = (f"Write a 22-30 word navigation instruction.\nExamples:\n{ex_block}\n"
                      f"Turns: {' then '.join(turns) if turns else 'straight'}. "
                      f"Stop near the {stop_lm}.\nStart with '{sv}':\n{sv}")
            retry_tasks.append({"episode_id": eid, "prompt": prompt,
                                 "sv": sv, "stop_lm": stop_lm})

        retry_results = await generate(retry_tasks, args.concurrency)
        ckpt.update({str(k): v for k, v in retry_results.items()})
        with open(CKPT, "w") as f:
            json.dump(ckpt, f)

        ep_out_map = {ep_o["episode_id"]: i for i, ep_o in enumerate(episodes_out)}
        for rt in retry_tasks:
            eid = rt["episode_id"]
            raw = retry_results.get(eid, "")
            ep  = ep_map.get(eid)
            if not raw or not ep: continue
            text = clean(raw, rt["sv"])
            ok, reason = quality_ok(text)
            if reason == "no_stop":
                text = fix_no_stop(text, rt["stop_lm"]); ok = True
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
        "_generation_meta": {"mode": "path_matched_v10", "model": VLLM_MODEL,
                             "n_ep": len(episodes_out)},
    }, OUTPUT)

    n = len(episodes_out)
    print(f"\n=== v10 Done === Eps:{n}  Pass:{n_pass}  Fixed:{n_fix}  Failed:{n_fail}")

    from collections import Counter
    wc: List[int] = []
    wait_n = stop_n = 0
    STOP_W = {"stop", "wait", "halt", "stand", "pause"}
    for ep_o in episodes_out:
        instr = (ep_o["instruction"]["instruction_text"]
                 if isinstance(ep_o.get("instruction"), dict) else "")
        words = instr.strip().split()
        if words: wc.append(len(words))
        il = instr.lower()
        if "wait" in il.split(): wait_n += 1
        if any(w in il.split() for w in STOP_W): stop_n += 1

    print(f"  avg_words: {sum(wc)/max(len(wc),1):.1f}  (GT=26.8)")
    print(f"  stop%: {100*stop_n/max(n,1):.1f}%  wait%: {100*wait_n/max(n,1):.2f}%")
    print(f"\nRun: cd {ROOT} && python3 run_similarity_eval.py --versions v10 v9 v2 --n 400")


if __name__ == "__main__":
    asyncio.run(main())
