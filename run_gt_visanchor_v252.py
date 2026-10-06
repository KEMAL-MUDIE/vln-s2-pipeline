"""
Gate 4 v252: GT body + visual stop anchor from goal frame.

Strategy: Keep ALL sentences of GT instruction except the last,
replace the stop sentence with a 5-10 word visual stop generated
from the correct goal frame (identified via poses.json).

Only 613 unique Gemma API calls (one per trajectory, not per episode).
Expected: > 63.57% (GT baseline) since navigation body is GT-quality
and stop anchors are visually grounded to the actual goal frame.
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

# ─────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
RF_DIR      = ROOT / "outputs" / "rendered_frames"
GT_FILE     = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz")
V238_FILE   = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v238.json.gz")
OUT_FILE    = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v252.json.gz")
CKPT_FILE   = ROOT / "outputs" / "gate4_v252_checkpoint.json"

GEMMA_BASE  = "http://10.77.32.231:8000/v1"
GEMMA_MODEL = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
# ─────────────────────────────────────────────────────────

client = OpenAI(base_url=GEMMA_BASE, api_key="dummy")


def load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def goal_frame_from_poses(eid: int) -> Optional[str]:
    """Return path to the rendered frame AT the goal waypoint."""
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


def split_body_stop(text: str):
    """
    Split GT instruction into (body, stop_sentence).
    Body = all sentences except the last.
    Stop = last sentence.
    """
    # Split on sentence boundaries (.!?) followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return text.strip(), ""
    return " ".join(sentences[:-1]), sentences[-1]


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
    """Call Gemma to generate a visual stop description from the goal frame."""
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
        # Ensure it starts with "Stop" (capitalize if needed)
        if raw and not raw[0].isupper():
            raw = raw[0].upper() + raw[1:]
        if not raw.lower().startswith("stop") and not raw.lower().startswith("wait"):
            raw = "Stop " + raw
        # Remove any extra sentences
        stop_match = re.split(r'(?<=[.!?])\s+', raw.strip())
        return stop_match[0].strip()
    except Exception as e:
        print(f"    [WARN] Gemma call failed: {e}", flush=True)
        return fallback


def main():
    # Load GT dataset
    print("Loading GT dataset...", flush=True)
    with gzip.open(GT_FILE) as f:
        gt_data = json.load(f)
    gt_eps = gt_data["episodes"]

    # Load checkpoint
    if CKPT_FILE.exists():
        with open(CKPT_FILE) as f:
            checkpoint = json.load(f)
        print(f"Loaded checkpoint: {len(checkpoint)} trajectories done", flush=True)
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
    print(f"Total episodes: {len(gt_eps)}", flush=True)

    # For each unique trajectory: generate visual stop anchor
    done = 0
    errors = 0
    for i, tid in enumerate(unique_trajs):
        if str(tid) in checkpoint:
            done += 1
            continue

        eps = traj_to_eps[tid]
        # Use first episode's ID to find goal frame
        rep_ep = eps[0]
        eid = rep_ep["episode_id"]

        # Get GT instruction for this trajectory
        gt_text = rep_ep["instruction"]["instruction_text"]
        body, gt_stop = split_body_stop(gt_text)

        # Get goal frame
        goal_frame = goal_frame_from_poses(eid)
        # Try other episode IDs in trajectory if first doesn't have it
        if not goal_frame:
            for ep in eps[1:]:
                goal_frame = goal_frame_from_poses(ep["episode_id"])
                if goal_frame:
                    break

        if not goal_frame:
            # Fallback: keep v238-style stop or GT stop
            fallback_stop = gt_stop if gt_stop else "Stop here."
            checkpoint[str(tid)] = {"visual_stop": fallback_stop, "fallback": True}
            errors += 1
        else:
            # Fallback stop = GT stop or "Stop here."
            fallback_stop = gt_stop if gt_stop else "Stop here."
            visual_stop = generate_visual_stop(goal_frame, fallback_stop)
            checkpoint[str(tid)] = {"visual_stop": visual_stop, "fallback": False,
                                     "goal_frame": goal_frame}

        done += 1

        # Save checkpoint every 20 trajectories
        if done % 20 == 0:
            with open(CKPT_FILE, "w") as f:
                json.dump(checkpoint, f)
            pct = i / len(unique_trajs) * 100
            rate = done / (i + 1)
            eta_min = (len(unique_trajs) - i - 1) / max(rate, 0.1) / 60 if rate > 0 else 999
            print(f"[{i+1}/{len(unique_trajs)}] {pct:.1f}% | errors={errors} | ETA {eta_min:.0f}min",
                  flush=True)

    # Final checkpoint save
    with open(CKPT_FILE, "w") as f:
        json.dump(checkpoint, f)
    print(f"All trajectories processed. errors={errors}", flush=True)

    # Build output dataset
    print("Building output dataset...", flush=True)
    new_eps = []
    for ep in gt_eps:
        tid = ep["trajectory_id"]
        tid_str = str(tid)
        gt_text = ep["instruction"]["instruction_text"]
        body, _ = split_body_stop(gt_text)

        if tid_str in checkpoint:
            visual_stop = checkpoint[tid_str]["visual_stop"]
        else:
            # Shouldn't happen, but fallback to GT
            visual_stop = split_body_stop(gt_text)[1] or "Stop here."

        # Combine: body + visual stop
        if body:
            new_text = body + " " + visual_stop
        else:
            new_text = visual_stop

        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        new_ep["instruction"]["instruction_text"] = new_text
        new_ep["instruction"]["instruction_tokens"] = None
        new_eps.append(new_ep)

    # Build output with instruction_vocab from v238
    with gzip.open(V238_FILE) as f:
        v238_data = json.load(f)

    out_data = {
        "instruction_vocab": v238_data["instruction_vocab"],
        "episodes": new_eps,
    }

    with gzip.open(OUT_FILE, "wt") as f:
        json.dump(out_data, f)

    # Stats
    lengths = [len(e["instruction"]["instruction_text"].split()) for e in new_eps]
    print(f"Written: {OUT_FILE}", flush=True)
    print(f"Episodes: {len(new_eps)}", flush=True)
    print(f"Avg words: {sum(lengths)/len(lengths):.1f}, min={min(lengths)}, max={max(lengths)}", flush=True)
    print("Sample instructions:", flush=True)
    for ep in new_eps[:3]:
        print(f"  ep{ep['episode_id']}: {ep['instruction']['instruction_text'][:180]}", flush=True)


if __name__ == "__main__":
    main()
