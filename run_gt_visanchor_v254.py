"""
Gate 4 v254: GT body + visual stop anchor with VARIED vocabulary.

Improvement over v252/v253:
- Same smart sentence handling as v253 (REPLACE pure stops, APPEND nav-containing)
- NEW Gemma prompt forces R2R-style stop vocabulary variety:
  GT distribution: at (19%), in (18%), by (10%), near (9%), in-front-of (7%)
  v252/v253 generated: in-front-of (94%) — WRONG
- Expected: better stop condition alignment with model's training distribution

No new API calls to Gemma if v254 checkpoint already exists. If not, generates 613 calls (~10 min).
"""

import gzip
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import base64
from openai import OpenAI

ROOT        = Path(__file__).parent
RF_DIR      = ROOT / "outputs" / "rendered_frames"
GT_FILE     = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz")
V238_FILE   = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v238.json.gz")
OUT_FILE    = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v254.json.gz")
CKPT_FILE   = ROOT / "outputs" / "gate4_v254_checkpoint.json"

GEMMA_BASE  = "http://10.77.32.231:8000/v1"
GEMMA_MODEL = "cyankiwi/gemma-4-31B-it-AWQ-4bit"

client = OpenAI(base_url=GEMMA_BASE, api_key="dummy")

NAV_VERBS = re.compile(
    r'\b(walk|go|turn|head|proceed|continue|pass|enter|exit|cross|climb|descend|move|take)\b',
    re.IGNORECASE
)

# ── Prompt: varied vocabulary matching R2R GT distribution ─────────────────
# GT distribution: at (19%), in (18%), by (10%), near (9%), in-front-of (7%)
GOAL_PROMPT_V254 = """\
You are writing a navigation stop instruction. Look at this image.
The robot has just reached its destination.

Write a STOP instruction (5-10 words) using one of these patterns — choose whichever fits best:
- "Stop at the [specific object or door]."     ← use for doorways, furniture, specific objects
- "Stop in the [room or area]."                ← use for rooms, hallways, open areas
- "Stop by the [furniture or fixture]."        ← use for lamps, chairs, counters, windows
- "Stop near the [prominent landmark]."        ← use for distinctive objects
- "Stop when you reach the [destination]."     ← use when you can describe the arrival clearly

DO NOT say "Stop in front of" — that is rarely used.
Use short, concrete nouns. Match R2R navigation instruction style.

Examples:
- "Stop at the wooden door at the end of the hall."
- "Stop in the kitchen area near the sink."
- "Stop by the grey armchair."
- "Stop near the dining table."
- "Stop when you reach the staircase landing."

Write ONLY the stop instruction (5-10 words):"""


def load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def goal_frame_from_poses(eid: int) -> Optional[str]:
    ep_str = f"episode_{eid:06d}"
    poses_file = RF_DIR / ep_str / "poses.json"
    if not poses_file.exists():
        return None
    try:
        with open(poses_file) as f:
            poses = json.load(f)
        best_idx = -1
        best_frame = None
        for i, fr in enumerate(poses.get("frames", [])):
            wp = fr.get("waypoint_idx", -1)
            cand = RF_DIR / ep_str / f"frame_{i:04d}_rgb.jpg"
            if wp > best_idx and cand.exists():
                best_idx = wp
                best_frame = str(cand)
        return best_frame
    except Exception:
        return None


def split_sentences(text: str):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]


def is_pure_stop(sentence: str) -> bool:
    return not bool(NAV_VERBS.search(sentence))


def combine_gt_and_visual_stop(gt_text: str, visual_stop: str) -> str:
    sents = split_sentences(gt_text)
    if len(sents) == 0:
        return visual_stop
    if len(sents) == 1:
        return sents[0].rstrip(' .!?') + '. ' + visual_stop
    last = sents[-1]
    body = ' '.join(sents[:-1])
    if is_pure_stop(last):
        return body + ' ' + visual_stop
    else:
        return gt_text.rstrip(' .!?') + '. ' + visual_stop


def generate_visual_stop(goal_frame_path: str, fallback: str) -> str:
    try:
        img_b64 = load_image_b64(goal_frame_path)
        resp = client.chat.completions.create(
            model=GEMMA_MODEL,
            max_tokens=40,
            temperature=0.4,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": GOAL_PROMPT_V254},
                ]
            }]
        )
        raw = resp.choices[0].message.content.strip()
        if raw and not raw[0].isupper():
            raw = raw[0].upper() + raw[1:]
        if not raw.lower().startswith("stop") and not raw.lower().startswith("wait"):
            raw = "Stop " + raw
        sentences = re.split(r'(?<=[.!?])\s+', raw.strip())
        return sentences[0].strip()
    except Exception as e:
        print(f"    [WARN] Gemma call failed: {e}", flush=True)
        return fallback


def main():
    print("Loading GT dataset...", flush=True)
    with gzip.open(GT_FILE) as f:
        gt_data = json.load(f)
    gt_eps = gt_data["episodes"]

    if CKPT_FILE.exists():
        with open(CKPT_FILE) as f:
            checkpoint = json.load(f)
        print(f"Loaded checkpoint: {len(checkpoint)} trajectories", flush=True)
    else:
        checkpoint = {}

    traj_to_eps = {}
    for ep in gt_eps:
        tid = ep["trajectory_id"]
        if tid not in traj_to_eps:
            traj_to_eps[tid] = []
        traj_to_eps[tid].append(ep)

    unique_trajs = sorted(traj_to_eps.keys())
    missing = [tid for tid in unique_trajs if str(tid) not in checkpoint]
    print(f"Unique trajs: {len(unique_trajs)}, need Gemma: {len(missing)}", flush=True)

    errors = 0
    processed = 0
    for i, tid in enumerate(missing):
        eps = traj_to_eps[tid]
        rep_ep = eps[0]
        eid = rep_ep["episode_id"]

        goal_frame = goal_frame_from_poses(eid)
        if not goal_frame:
            for ep in eps[1:]:
                goal_frame = goal_frame_from_poses(ep["episode_id"])
                if goal_frame:
                    break

        sents = split_sentences(rep_ep["instruction"]["instruction_text"])
        fallback = sents[-1] if sents else "Stop here."

        if not goal_frame:
            checkpoint[str(tid)] = {"visual_stop": fallback, "fallback": True}
            errors += 1
        else:
            visual_stop = generate_visual_stop(goal_frame, fallback)
            checkpoint[str(tid)] = {"visual_stop": visual_stop, "fallback": False}

        processed += 1
        if processed % 20 == 0:
            with open(CKPT_FILE, "w") as f:
                json.dump(checkpoint, f)
            rate = processed / max(i + 1, 1)
            eta = (len(missing) - processed) / max(rate, 0.01) / 60
            print(f"[{len(checkpoint)}/{len(unique_trajs)}] errors={errors} ETA {eta:.0f}min", flush=True)

    with open(CKPT_FILE, "w") as f:
        json.dump(checkpoint, f)
    print(f"Done. errors={errors}", flush=True)

    # Build output
    print("Building output dataset...", flush=True)
    new_eps = []
    replace_count = 0
    append_count = 0

    for ep in gt_eps:
        tid_str = str(ep["trajectory_id"])
        gt_text = ep["instruction"]["instruction_text"]
        visual_stop = checkpoint.get(tid_str, {}).get("visual_stop", "Stop here.")
        new_text = combine_gt_and_visual_stop(gt_text, visual_stop)

        sents = split_sentences(gt_text)
        if len(sents) > 1 and is_pure_stop(sents[-1]):
            replace_count += 1
        else:
            append_count += 1

        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        new_ep["instruction"]["instruction_text"] = new_text
        new_ep["instruction"]["instruction_tokens"] = None
        new_eps.append(new_ep)

    with gzip.open(V238_FILE) as f:
        v238_data = json.load(f)

    out_data = {"instruction_vocab": v238_data["instruction_vocab"], "episodes": new_eps}

    with gzip.open(OUT_FILE, "wt") as f:
        json.dump(out_data, f)

    lengths = [len(e["instruction"]["instruction_text"].split()) for e in new_eps]
    print(f"Written: {OUT_FILE}", flush=True)
    print(f"Episodes: {len(new_eps)}, REPLACE={replace_count}, APPEND={append_count}", flush=True)
    print(f"Avg words: {sum(lengths)/len(lengths):.1f}", flush=True)
    print("Samples:", flush=True)
    for ep in new_eps[:4]:
        print(f"  ep{ep['episode_id']}: {ep['instruction']['instruction_text'][:180]}", flush=True)


if __name__ == "__main__":
    main()
