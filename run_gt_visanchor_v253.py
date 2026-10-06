"""
Gate 4 v253: GT body + visual stop anchor — smarter sentence handling.

Improvement over v252:
- SAME visual stop anchor generation (goal frame via poses.json → Gemma)
- SMARTER combination:
  * Pure stop sentences (43.9%): REPLACE with visual anchor (same as v252)
  * Navigation-containing last sentences (36.8%): KEEP + APPEND visual anchor
  * Single-sentence instructions (19.2%): KEEP + APPEND visual anchor
- Preserves 100% of GT navigation content; only replaces vague stop conditions

Expected: > v252 (> 63.57% GT baseline) by preserving navigation info in 36.8% of cases.
"""

import gzip
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import base64
from openai import OpenAI

ROOT        = Path(__file__).parent
RF_DIR      = ROOT / "outputs" / "rendered_frames"
GT_FILE     = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz")
V238_FILE   = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v238.json.gz")
OUT_FILE    = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v253.json.gz")
CKPT_FILE   = ROOT / "outputs" / "gate4_v253_checkpoint.json"

# Reuse v252 checkpoint since we use the same Gemma calls
V252_CKPT   = ROOT / "outputs" / "gate4_v252_checkpoint.json"

GEMMA_BASE  = "http://10.77.32.231:8000/v1"
GEMMA_MODEL = "cyankiwi/gemma-4-31B-it-AWQ-4bit"

client = OpenAI(base_url=GEMMA_BASE, api_key="dummy")

NAV_VERBS = re.compile(
    r'\b(walk|go|turn|head|proceed|continue|pass|enter|exit|cross|climb|descend|move|take)\b',
    re.IGNORECASE
)


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
    """Returns True if sentence contains NO navigation verbs (it's a pure stop condition)."""
    return not bool(NAV_VERBS.search(sentence))


def combine_gt_and_visual_stop(gt_text: str, visual_stop: str) -> str:
    """
    Combine GT instruction with visual stop anchor using smart strategy:
    - Single sentence or navigation-containing last sentence: APPEND visual stop
    - Pure stop last sentence: REPLACE with visual stop
    """
    sents = split_sentences(gt_text)

    if len(sents) == 0:
        return visual_stop

    if len(sents) == 1:
        # Single sentence — always APPEND (preserve navigation)
        return sents[0].rstrip(' .!?') + '. ' + visual_stop

    last = sents[-1]
    body = ' '.join(sents[:-1])

    if is_pure_stop(last):
        # Last sentence is pure stop (e.g., "Stop here.", "Wait near the rug.")
        # REPLACE with visual anchor
        return body + ' ' + visual_stop
    else:
        # Last sentence contains navigation (e.g., "Walk past the couch and stop near the rug.")
        # KEEP full GT + APPEND visual anchor
        return gt_text.rstrip(' .!?') + '. ' + visual_stop


GOAL_PROMPT = """\
You are a navigation assistant. Look at this image.
The robot has just reached its destination.
Write a STOP instruction in 5-10 words using this template:
"Stop [when/near/at/in front of] [specific visual landmark]."

Examples of good stop instructions:
- "Stop near the wooden dresser by the window."
- "Stop in front of the white door at the end."
- "Stop at the foot of the staircase."
- "Stop when you reach the red pool table."
- "Stop here in the hallway by the elevator."

Write ONLY the stop instruction (one sentence, 5-10 words):"""


def generate_visual_stop(goal_frame_path: str, fallback: str) -> str:
    try:
        img_b64 = load_image_b64(goal_frame_path)
        resp = client.chat.completions.create(
            model=GEMMA_MODEL,
            max_tokens=40,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": GOAL_PROMPT},
                ]
            }]
        )
        raw = resp.choices[0].message.content.strip()
        if raw and not raw[0].isupper():
            raw = raw[0].upper() + raw[1:]
        if not raw.lower().startswith("stop") and not raw.lower().startswith("wait"):
            raw = "Stop " + raw
        stop_match = re.split(r'(?<=[.!?])\s+', raw.strip())
        return stop_match[0].strip()
    except Exception as e:
        print(f"    [WARN] Gemma call failed: {e}", flush=True)
        return fallback


def main():
    print("Loading GT dataset...", flush=True)
    with gzip.open(GT_FILE) as f:
        gt_data = json.load(f)
    gt_eps = gt_data["episodes"]

    # Load v252 checkpoint (reuse same Gemma results — no extra API calls)
    if V252_CKPT.exists():
        with open(V252_CKPT) as f:
            checkpoint = json.load(f)
        print(f"Loaded v252 checkpoint: {len(checkpoint)} trajectories (reusing Gemma results)", flush=True)
    elif CKPT_FILE.exists():
        with open(CKPT_FILE) as f:
            checkpoint = json.load(f)
        print(f"Loaded v253 checkpoint: {len(checkpoint)} trajectories", flush=True)
    else:
        checkpoint = {}

    # Build trajectory → episodes mapping
    traj_to_eps = {}
    for ep in gt_eps:
        tid = ep["trajectory_id"]
        if tid not in traj_to_eps:
            traj_to_eps[tid] = []
        traj_to_eps[tid].append(ep)

    unique_trajs = sorted(traj_to_eps.keys())
    print(f"Unique trajectories: {len(unique_trajs)}", flush=True)

    # Check if we need to fill any missing (v252 checkpoint had 0 errors so should be complete)
    missing = [tid for tid in unique_trajs if str(tid) not in checkpoint]
    print(f"Missing trajectories needing Gemma: {len(missing)}", flush=True)

    done = len(checkpoint)
    errors = 0
    for i, tid in enumerate(missing):
        eps = traj_to_eps[tid]
        rep_ep = eps[0]
        eid = rep_ep["episode_id"]
        gt_text = rep_ep["instruction"]["instruction_text"]

        goal_frame = goal_frame_from_poses(eid)
        if not goal_frame:
            for ep in eps[1:]:
                goal_frame = goal_frame_from_poses(ep["episode_id"])
                if goal_frame:
                    break

        sents = split_sentences(gt_text)
        fallback_stop = sents[-1] if sents else "Stop here."

        if not goal_frame:
            checkpoint[str(tid)] = {"visual_stop": fallback_stop, "fallback": True}
            errors += 1
        else:
            visual_stop = generate_visual_stop(goal_frame, fallback_stop)
            checkpoint[str(tid)] = {"visual_stop": visual_stop, "fallback": False}

        done += 1
        if done % 20 == 0:
            with open(CKPT_FILE, "w") as f:
                json.dump(checkpoint, f)
            print(f"[{done}/{len(unique_trajs)}] errors={errors}", flush=True)

    with open(CKPT_FILE, "w") as f:
        json.dump(checkpoint, f)
    print(f"All trajectories done. errors={errors}", flush=True)

    # Analyze instruction types
    replace_count = 0
    append_count = 0

    # Build output dataset
    print("Building output dataset...", flush=True)
    new_eps = []
    for ep in gt_eps:
        tid = ep["trajectory_id"]
        tid_str = str(tid)
        gt_text = ep["instruction"]["instruction_text"]

        visual_stop = checkpoint.get(tid_str, {}).get("visual_stop", "Stop here.")
        new_text = combine_gt_and_visual_stop(gt_text, visual_stop)

        # Track strategy used
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

    out_data = {
        "instruction_vocab": v238_data["instruction_vocab"],
        "episodes": new_eps,
    }

    with gzip.open(OUT_FILE, "wt") as f:
        json.dump(out_data, f)

    lengths = [len(e["instruction"]["instruction_text"].split()) for e in new_eps]
    print(f"Written: {OUT_FILE}", flush=True)
    print(f"Episodes: {len(new_eps)}", flush=True)
    print(f"Strategy: REPLACE={replace_count} ({replace_count/len(new_eps)*100:.0f}%), APPEND={append_count} ({append_count/len(new_eps)*100:.0f}%)", flush=True)
    print(f"Avg words: {sum(lengths)/len(lengths):.1f}, min={min(lengths)}, max={max(lengths)}", flush=True)
    print("Sample instructions:", flush=True)
    for ep in new_eps[:4]:
        print(f"  ep{ep['episode_id']}: {ep['instruction']['instruction_text'][:200]}", flush=True)


if __name__ == "__main__":
    main()
