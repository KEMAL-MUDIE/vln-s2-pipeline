#!/usr/bin/env python3
"""
Gate4-only re-annotation: reads existing meta (Gate3 scene descriptions already done)
and re-runs ONLY instruction generation with the current Gate4 prompt.

Use this to test improved prompts (v279) without redoing Gate1-3.

Usage:
  # Re-annotate mini train (400 eps) using v279 config
  python3 reannotate_gate4_only.py \
    --config configs/annotator_v279.yaml \
    --meta-dir outputs/annotated_datasets_top5/train_v278_meta \
    --ep-list outputs/mini_train_gt.json.gz \
    --out-dir outputs/mini_train_v279_meta \
    --workers 8

  # Then assemble the dataset:
  python3 reannotate_gate4_only.py --assemble \
    --meta-dir outputs/mini_train_v279_meta \
    --gt-data /mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/train/train.json.gz \
    --out outputs/mini_train_auto_v279.json.gz
"""
import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import yaml

PIPELINE_ROOT = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_ROOT))


def load_config(config_path: str) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _generate_text(client, prompt: str, cfg: Dict, temperature_override=None) -> str:
    gate4_cfg = cfg.get("gate4", {})
    temp = temperature_override if temperature_override is not None else gate4_cfg.get("temperature", 0.45)
    resp = client.chat.completions.create(
        model=cfg["vllm"]["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=gate4_cfg.get("max_tokens", 150),
        temperature=temp,
    )
    return resp.choices[0].message.content.strip()


def _build_instruction_prompt(meta: Dict) -> str:
    from run_annotator_configured import INSTRUCTION_GENERATION_PROMPT
    path_analysis = meta.get("path_analysis", {})
    motion_text = path_analysis.get("motion_text", "Navigate to the destination.")
    landmark_ann = meta.get("landmark_annotations", {})
    per_frame = landmark_ann.get("per_frame", {})
    goal_landmark = landmark_ann.get("goal_landmark", {})

    scene_context_text = "\n".join(
        f"  [Frame {k}] {v.get('room','?').title()} — {', '.join(v.get('landmarks',[]))}, dir: {v.get('direction','?')}"
        for k, v in sorted(per_frame.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0)
    ) or "  Indoor environment."
    goal_desc = goal_landmark.get("description", "the destination")

    return INSTRUCTION_GENERATION_PROMPT.format(
        motion_text=motion_text,
        scene_context=scene_context_text,
        goal_landmark=goal_desc,
    )


def _quality_check(text: str, cfg: Dict) -> List[str]:
    from run_annotator_configured import _quality_check as _qc
    return _qc(text, cfg)


def reannotate_episode(ep_id: int, meta_dir: Path, out_dir: Path, cfg: Dict) -> Dict:
    meta_f = meta_dir / f"ep_{ep_id:06d}.json"
    if not meta_f.exists():
        return {"ep_id": ep_id, "error": "meta_not_found"}

    with open(meta_f) as f:
        meta = json.load(f)

    from openai import OpenAI
    client = OpenAI(base_url=cfg["vllm"]["base_url"], api_key=cfg["vllm"]["api_key"])

    prompt = _build_instruction_prompt(meta)
    gate4_cfg = cfg.get("gate4", {})
    base_temp = gate4_cfg.get("temperature", 0.45)
    max_retries = gate4_cfg.get("max_retries", 5)

    generated_text = ""
    quality_ok = False
    issues = []
    for attempt in range(max_retries):
        try:
            temp = base_temp + attempt * 0.1
            generated_text = _generate_text(client, prompt, cfg, temperature_override=temp)
            issues = _quality_check(generated_text, cfg)
            quality_ok = len(issues) == 0
            if quality_ok:
                break
        except Exception as e:
            issues = [str(e)]

    # Update meta with new instruction
    updated_meta = {**meta}
    updated_meta["generated_instruction"] = {
        "text": generated_text,
        "generator": cfg["vllm"]["model"],
        "version": cfg.get("version", "v279"),
        "quality_ok": quality_ok,
        "quality_issues": issues,
    }
    updated_meta["_annotation_version"] = cfg.get("version", "v279")

    out_f = out_dir / f"ep_{ep_id:06d}.json"
    with open(out_f, "w") as f:
        json.dump(updated_meta, f)

    return {"ep_id": ep_id, "quality_ok": quality_ok, "issues": issues, "text": generated_text[:60]}


def assemble(meta_dir: Path, gt_data_path: str, out_path: Path):
    """Assemble final dataset from re-annotated meta files."""
    with gzip.open(gt_data_path, "rt") as f:
        gt_data = json.load(f)
    gt_eps = {ep["episode_id"]: ep for ep in gt_data["episodes"]}

    meta_files = sorted(meta_dir.glob("ep_*.json"))
    print(f"Assembling {len(meta_files)} episodes → {out_path}")

    episodes = []
    failed = 0
    for mf in meta_files:
        with open(mf) as f:
            meta = json.load(f)
        ep_id = meta["episode_id"]
        vlm_text = meta.get("generated_instruction", {}).get("text", "").strip()
        gt_ep = gt_eps.get(ep_id, {})
        gt_text = gt_ep.get("instruction", {}).get("instruction_text", "")

        if not vlm_text:
            vlm_text = gt_text
            failed += 1

        ep = {k: v for k, v in gt_ep.items()}
        ep["instruction"] = {"instruction_text": vlm_text, "instruction_tokens": vlm_text.split()}
        ep["_vlm_instruction"] = vlm_text
        ep["_gt_instruction"] = gt_text
        ep["_quality_ok"] = meta.get("generated_instruction", {}).get("quality_ok", False)
        ep["_annotation_version"] = meta.get("_annotation_version", "v279")
        episodes.append(ep)

    out = {k: v for k, v in gt_data.items() if k != "episodes"}
    out["episodes"] = episodes
    with gzip.open(out_path, "wt") as f:
        json.dump(out, f)

    print(f"Done: {len(episodes)} episodes, {failed} VLM-failed (used GT fallback)")
    print(f"Quality OK: {sum(1 for e in episodes if e['_quality_ok'])} / {len(episodes)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    parser.add_argument("--meta-dir", required=True)
    parser.add_argument("--out-dir", required=False)
    parser.add_argument("--ep-list", help="json.gz with episodes to re-annotate (subset)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--gt-data", help="GT json.gz for assemble mode")
    parser.add_argument("--out", help="Output path for assemble mode")
    args = parser.parse_args()

    meta_dir = Path(args.meta_dir)

    if args.assemble:
        if not args.gt_data or not args.out:
            print("--assemble requires --gt-data and --out")
            sys.exit(1)
        assemble(meta_dir, args.gt_data, Path(args.out))
        return

    if not args.config or not args.out_dir:
        print("Need --config and --out-dir for annotation mode")
        sys.exit(1)

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get episode list
    if args.ep_list:
        with gzip.open(args.ep_list, "rt") as f:
            ep_data = json.load(f)
        ep_ids = [ep["episode_id"] for ep in ep_data["episodes"]]
    else:
        ep_ids = [int(f.stem.replace("ep_", "")) for f in sorted(meta_dir.glob("ep_*.json"))]

    # Skip already done
    if args.resume:
        done = {int(f.stem.replace("ep_", "")) for f in out_dir.glob("ep_*.json")}
        ep_ids = [eid for eid in ep_ids if eid not in done]

    print(f"Re-annotating Gate4 for {len(ep_ids)} episodes (workers={args.workers})")
    print(f"  Config: {args.config}  version={cfg.get('version','?')}")
    print(f"  Source meta: {meta_dir}")
    print(f"  Output meta: {out_dir}")

    t0 = time.time()
    ok = err = bias = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(reannotate_episode, eid, meta_dir, out_dir, cfg): eid for eid in ep_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r.get("error"):
                err += 1
            elif r["quality_ok"]:
                ok += 1
            else:
                if "go_straight_bias" in r.get("issues", []):
                    bias += 1
                err += 1
            if i % 50 == 0 or i == len(ep_ids):
                elapsed = time.time() - t0
                print(f"  [{i}/{len(ep_ids)}] OK={ok} err={err} bias_fail={bias}  {elapsed:.0f}s")

    print(f"\nDone: {ok}/{len(ep_ids)} quality OK, {err} failed ({bias} bias-rejected)")
    if args.out_dir:
        print(f"\nNext: python3 reannotate_gate4_only.py --assemble --meta-dir {out_dir} "
              f"--gt-data <train.json.gz> --out outputs/mini_train_auto_v279.json.gz")


if __name__ == "__main__":
    main()
