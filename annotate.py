#!/usr/bin/env python3
"""
Label-config annotator: one YAML per label, no script copies.

  configs/labels/<label>.yaml     label config. `base: <label>` inherits and overrides (deep merge).
                                  Label naming: {base}_{axis}-{value}, e.g. v278fix_stop-gtdist.
  configs/labels/ablations.yaml   variant list -> make_labels.py writes the per-label yamls.
  configs/labels/registry.yaml    one entry per label: base, overrides, hash, status, quality metrics.

Same gate1-4 pipeline as run_annotator_configured.py (whose helpers/prompts are imported, not
copied), plus:
  - every knob in the yaml (frames dir, key-frame mapping, gate3/gate4 params, prompt options)
  - per-request vLLM seed derived from (seed, episode, stage, frame/attempt) -> reproducible, and
    ablations that share a gate3 config reuse one gate3 cache (identical scene descriptions)
  - prompt options: length (15-40 | free | gtdist), stop (must | gtdist | optional),
    fewshot N (GT train sentences of other trajectories as style examples), diverse_siblings
  - _generation_meta (resolved config, hash, git sha) inside every output json.gz
  - never overwrites: an existing output with a different config hash is an error

Usage:
  python3 annotate.py v278fix                         # all splits -> assemble -> deploy -> registry
  python3 annotate.py v278fix --splits val_unseen     # one split (deploy waits until all 3 exist)
  python3 annotate.py v278fix_len-free --subset 500   # screening: first 500 eps/split, screen/ dir, no deploy
  python3 annotate.py --selfcheck                     # baseline prompt == run_annotator_configured prompt
"""
import argparse
import copy
import gzip
import hashlib
import json
import random
import shutil
import statistics as st
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gate7_eval"))

import run_annotator_configured as rac  # noqa: E402  (helpers + Kemal's prompts, unmodified)
from convert_top5_generated import normalize  # noqa: E402
from local_paths import CACHE_ROOT, GT_PATHS, HABITAT_BASE, VOCAB_SOURCE  # noqa: E402

LABEL_DIR = ROOT / "configs" / "labels"
REGISTRY = LABEL_DIR / "registry.yaml"
OUT_ROOT = CACHE_ROOT / "labels"          # <label>/{split}_meta/, {split}.json.gz
SCREEN_ROOT = CACHE_ROOT / "screen"       # --subset runs, never deployed
GATE3_CACHE = CACHE_ROOT / "gate3_cache"  # <gate3 hash>/<split>/ep_XXXXXX.json, shared across labels
SHARE = Path("/home/irteam/data-vol2/relabel_half/labels")  # data-vol2 copy of deployed labels
SPLITS = ["train", "val_seen", "val_unseen"]

# GT train stop-expression distribution (2026-10-08, 10,819 sentences): stop 53.4 / wait 31.2 /
# stand 2.5 / no stop phrase 10.9 (rest <2%: "end" is mostly "the end of the hall").
STOP_GTDIST = {"stop": 0.55, "wait": 0.32, "stand": 0.03, "none": 0.10}
STOP_LINE = {
    "must": '- MUST include a clear stop condition: "stop at/near/by [landmark]" or "wait at [landmark]"',
    "stop": '- The final sentence must use the verb "stop" (e.g. "stop at/near/by [landmark]")',
    "wait": '- The final sentence must use the verb "wait", not "stop" (e.g. "wait at/near/by [landmark]")',
    "stand": '- The final sentence must use the verb "stand", not "stop" (e.g. "stand next to/in front of [landmark]")',
    "none": '- Do NOT use the words "stop", "wait" or "stand"; end by naming the final place to reach '
            '(e.g. "...and walk into the office area.")',
    "optional": '- You may end with "stop/wait/stand at [landmark]", or simply by naming the final place to reach',
}

# Kemal's INSTRUCTION_GENERATION_PROMPT with the two varied lines turned into slots.
GATE4_TEMPLATE = """\
You are a navigation instruction writer for vision-language robot navigation.

Write a concise, natural navigation instruction for this path.

PATH MOTION SEQUENCE:
{motion_text}

SCENE CONTEXT AT KEY POINTS:
{scene_context}

GOAL/DESTINATION:
{goal_landmark}

REQUIREMENTS:
- {length_rule}
- Reference specific visible landmarks (furniture, rooms, doorways, etc.)
- Use natural turn language: "turn left/right", "make a left", "go left at"
{stop_rule}
- No distances in metres or numbers
- Concise, direct — like giving directions to a person
{extra}
Write ONLY the instruction text:
"""


# --------------------------------------------------------------------------- config
def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_label(label):
    raw = yaml.safe_load(open(LABEL_DIR / f"{label}.yaml"))
    assert raw.get("label") == label, f"{label}.yaml has label={raw.get('label')!r}"
    base = raw.pop("base", None)
    cfg = deep_merge(load_label(base), raw) if base else raw
    cfg["label"] = label
    return cfg


def h(obj, n=12):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:n]


def gen_hash(cfg):  # everything that can change the text (incl. prompt text in this file); not run plumbing
    return h({k: v for k, v in cfg.items() if k not in ("label", "workers", "vllm", "notes")}
             | {"model": cfg["vllm"]["model"], "template": GATE4_TEMPLATE, "stop_lines": STOP_LINE,
                "gate3": gate3_hash(cfg)})


def gate3_hash(cfg):
    return h({"model": cfg["vllm"]["model"], "frames": cfg["frames"], "keyframes": cfg["keyframes"],
              "seed": cfg["seed"], "gate3": cfg["gate3"],
              "prompts": [rac.SCENE_DESCRIPTION_PROMPT, rac.GOAL_LANDMARK_PROMPT]})


def call_seed(*parts):
    return int(hashlib.sha1(":".join(map(str, parts)).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- GT side info
_GT_TRAIN = None


def gt_train():
    """(word-length list, [(trajectory_id, text)]) from GT train — gtdist length + fewshot pool."""
    global _GT_TRAIN
    if _GT_TRAIN is None:
        eps = json.load(gzip.open(GT_PATHS["train"]))["episodes"]
        pool = [(e["trajectory_id"], e["instruction"]["instruction_text"].strip()) for e in eps]
        _GT_TRAIN = ([len(t.split()) for _, t in pool], pool)
    return _GT_TRAIN


# --------------------------------------------------------------------------- gate3 (cached)
def _image_call(client, cfg, img, prompt, max_tokens, seed):
    import base64
    b64 = base64.b64encode(open(img, "rb").read()).decode()
    mime = "image/jpeg" if Path(img).suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    resp = client.chat.completions.create(
        model=cfg["vllm"]["model"],
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt}]}],
        max_tokens=max_tokens, temperature=cfg["gate3"]["temperature"], seed=seed)
    return resp.choices[0].message.content.strip()


def gate3(client, cfg, split, ep):
    ep_id = ep["episode_id"]
    cache = GATE3_CACHE / gate3_hash(cfg) / split / f"ep_{ep_id:06d}.json"
    if cache.exists():
        return json.load(open(cache))
    ep_dir = CACHE_ROOT / cfg["frames"] / split / f"episode_{ep_id:06d}"
    poses = json.load(open(ep_dir / "poses.json"))
    frame_paths = [str(ep_dir / f["path"]) for f in poses["frames"]]
    pa = rac._path_analysis(ep["reference_path"], ep.get("start_rotation", [0, 0, 0, 1]))
    kfi = pa.get("key_frame_indices", list(range(len(frame_paths))))
    black = {f["frame_idx"] for f in poses["frames"] if f.get("black")}
    if cfg["keyframes"] == "by_waypoint":  # gate2 returns waypoint indices; map them to rendered frames
        wp2frame = {f["waypoint_idx"]: f["frame_idx"] for f in poses["frames"]}
        frames = sorted({wp2frame[i] for i in kfi if i in wp2frame} - black)
    else:  # "by_index" = upstream behaviour (drops most turn frames)
        frames = [i for i in kfi if i < len(frame_paths)]
    g3 = cfg["gate3"]
    per_frame = {}
    for fi in frames:
        try:
            raw = _image_call(client, cfg, frame_paths[fi], rac.SCENE_DESCRIPTION_PROMPT, g3["max_tokens"],
                              call_seed(cfg["seed"], ep_id, "g3", fi))
            per_frame[str(fi)] = rac._parse_scene_description(raw)
        except Exception as e:
            per_frame[str(fi)] = {"room": "unknown", "landmarks": [], "direction": "unclear", "error": str(e)}
    goal = {"description": "the destination", "raw": ""}
    if poses["frames"][-1]["frame_idx"] not in black:
        try:
            raw = _image_call(client, cfg, frame_paths[-1], rac.GOAL_LANDMARK_PROMPT, g3["goal_max_tokens"],
                              call_seed(cfg["seed"], ep_id, "goal"))
            goal = {"description": raw, "raw": raw}
        except Exception as e:
            goal["error"] = str(e)
    out = {"per_frame": per_frame, "goal_landmark": goal, "key_frames": frames,
           "motion_text": pa.get("motion_text", "Navigate to the destination."), "n_frames": len(frame_paths)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(cache, "w"))
    return out


# --------------------------------------------------------------------------- gate4
def build_prompt(cfg, ep, g3, siblings):
    p = cfg["prompt"]
    rng = random.Random(f"{cfg['seed']}:{ep['episode_id']}:prompt")
    if p["length"] == "free":
        length_rule = "Use as many sentences and words as the path needs"
    elif p["length"] == "gtdist":
        length_rule = f"1-4 sentences, about {rng.choice(gt_train()[0])} words total"
    else:
        length_rule = f"1-4 short sentences, {p['length']} words total"
    stop = p["stop"]
    if stop == "gtdist":
        stop = rng.choices(list(STOP_GTDIST), weights=list(STOP_GTDIST.values()))[0]
    extra = ""
    if p["fewshot"]:
        pool = gt_train()[1]
        ex = [t for tid, t in rng.sample(pool, 4 * p["fewshot"]) if tid != ep.get("trajectory_id")][:p["fewshot"]]
        extra += ("\nSTYLE EXAMPLES (human-written instructions for OTHER paths — match their style, "
                  "not their content):\n" + "".join(f"- {t}\n" for t in ex))
    if siblings:
        extra += ("\nALREADY WRITTEN for this same path — yours must be clearly different in wording, "
                  "sentence structure and which landmarks you mention:\n" + "".join(f"- {t}\n" for t in siblings))
    scene = "\n".join(
        f"  [Frame {k}] {v.get('room', '?').title()} — {', '.join(v.get('landmarks', []))}, dir: {v.get('direction', '?')}"
        for k, v in sorted(g3["per_frame"].items(), key=lambda x: int(x[0]))) or "  Indoor environment."
    prompt = GATE4_TEMPLATE.format(motion_text=g3["motion_text"], scene_context=scene,
                                   goal_landmark=g3["goal_landmark"]["description"],
                                   length_rule=length_rule, stop_rule=STOP_LINE[stop], extra=extra)
    return prompt, stop


def gate4(client, cfg, ep, g3, siblings):
    prompt, stop = build_prompt(cfg, ep, g3, siblings)
    qcfg = cfg if stop not in ("none", "optional") else deep_merge(cfg, {"quality": {"require_stop_condition": False}})
    g4 = cfg["gate4"]
    text, issues = "", []
    for attempt in range(g4["max_retries"]):
        try:
            resp = client.chat.completions.create(
                model=cfg["vllm"]["model"], messages=[{"role": "user", "content": prompt}],
                max_tokens=g4["max_tokens"], temperature=g4["temperature"] + 0.1 * attempt,
                seed=call_seed(cfg["seed"], ep["episode_id"], "g4", attempt))
            text = resp.choices[0].message.content.strip()
            issues = rac._quality_check(text, qcfg)
            if not issues:
                break
        except Exception as e:
            issues = [str(e)]
    return {"text": text, "quality_ok": not issues, "quality_issues": issues, "stop_style": stop,
            "attempts": attempt + 1}


# --------------------------------------------------------------------------- batch
def run_split(cfg, split, workers, subset):
    from openai import OpenAI
    client = OpenAI(base_url=cfg["vllm"]["base_url"], api_key=cfg["vllm"]["api_key"], max_retries=5)
    root = (SCREEN_ROOT if subset else OUT_ROOT) / cfg["label"]
    meta_dir = root / f"{split}_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    ghash = gen_hash(cfg)
    gt = json.load(gzip.open(GT_PATHS[split]))
    eps = gt["episodes"][:subset] if subset else gt["episodes"]
    groups = defaultdict(list)  # same trajectory -> one unit (siblings are generated in order)
    for e in eps:
        groups[e["trajectory_id"]].append(e)

    def do_group(g):
        siblings = []
        for ep in sorted(g, key=lambda e: e["episode_id"]):
            f = meta_dir / f"ep_{ep['episode_id']:06d}.json"
            if f.exists():
                m = json.load(open(f))
                assert m["gen_hash"] == ghash, f"{f} was made with another config ({m['gen_hash']} != {ghash})"
            else:
                g3 = gate3(client, cfg, split, ep)
                g4 = gate4(client, cfg, ep, g3, siblings if cfg["prompt"]["diverse_siblings"] else [])
                m = {"episode_id": ep["episode_id"], "trajectory_id": ep["trajectory_id"], "gen_hash": ghash,
                     "generated_instruction": g4, "landmark_annotations": g3}
                json.dump(m, open(f, "w"))
            siblings.append(m["generated_instruction"]["text"])
        return len(g)

    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(do_group, g) for g in groups.values()]
        for fu in as_completed(futs):
            done += fu.result()
            if done % 500 < 3:
                print(f"  [{split}] {done}/{len(eps)}  {time.time() - t0:.0f}s", flush=True)

    out = {"episodes": [], "instruction_vocab": json.load(gzip.open(VOCAB_SOURCE))["instruction_vocab"],
           "_generation_meta": {"label": cfg["label"], "gen_hash": ghash, "gate3_hash": gate3_hash(cfg),
                                "config": cfg, "git": git_sha(), "split": split, "subset": subset,
                                "created": time.strftime("%Y-%m-%d %H:%M:%S")}}
    # episodes keep only GT keys: habitat builds VLNEpisode(**episode) and rejects unknown keys.
    # per-episode generation records stay in {split}_meta/ep_*.json; only a summary goes here.
    n_bad, styles = 0, Counter()
    for e in eps:
        m = json.load(open(meta_dir / f"ep_{e['episode_id']:06d}.json"))
        gi = m["generated_instruction"]
        n_bad += not gi["quality_ok"]
        styles[gi["stop_style"]] += 1
        ep = {k: e[k] for k in ("episode_id", "trajectory_id", "scene_id", "start_position", "start_rotation",
                                "goals", "reference_path", "info") if k in e}
        ep["instruction"] = {"instruction_text": normalize(gi["text"]), "instruction_tokens": []}
        out["episodes"].append(ep)
    out["_generation_meta"].update(quality_fail=n_bad, stop_style=dict(styles))
    dst = root / f"{split}.json.gz"
    with gzip.open(dst, "wt") as f:
        json.dump(out, f)
    print(f"  [{split}] {len(eps)} eps -> {dst}  quality_fail={n_bad}  {time.time() - t0:.0f}s", flush=True)
    return dst


# --------------------------------------------------------------------------- metrics / deploy / registry
def metrics(path, split):
    import re
    from check_label import load_specs, measure
    q, p = load_specs()
    gt_path = GT_PATHS[split]
    s = measure(str(path), str(gt_path), split, q, p)
    eps = json.load(gzip.open(path))["episodes"]
    t = [e["instruction"]["instruction_text"].strip() for e in eps]
    gt = {e["episode_id"]: e["instruction"]["instruction_text"] for e in json.load(gzip.open(gt_path))["episodes"]}

    def f1(a, b):
        a, b = Counter(re.findall(r"[a-z]+", a.lower())), Counter(re.findall(r"[a-z]+", b.lower()))
        c = sum((a & b).values())
        return 2 * c / (sum(a.values()) + sum(b.values())) if c else 0.0

    by_traj = defaultdict(list)
    for e in eps:
        by_traj[e["trajectory_id"]].append(e["instruction"]["instruction_text"])
    sib = [f1(a, b) for g in by_traj.values() for i, a in enumerate(g) for b in g[i + 1:]]
    stopw = Counter(next((k for k in ("stop", "wait", "stand") if re.search(rf"\b{k}", x.lower())), "none") for x in t)
    w = [len(x.split()) for x in t]
    return {"n": len(t), "unique_pct": round(s["unique_pct"], 1), "content_vocab": s["content_vocab"],
            "direction": round(s["direction_consistency"], 3), "g6_pct": round(s["stop_grounded_pct"], 1),
            "words_mean": round(st.mean(w), 1), "words_std": round(st.pstdev(w), 1),
            "stop_pct": {k: round(100 * v / len(t), 1) for k, v in stopw.most_common()},
            "first_word_go_pct": round(100 * sum(x.lower().startswith("go ") for x in t) / len(t), 1),
            "sibling_f1": round(st.mean(sib), 3) if sib else None,
            "f1_vs_gt": round(st.mean(f1(e["instruction"]["instruction_text"], gt[e["episode_id"]]) for e in eps), 3)}


def check_episode_keys(path, split):
    """habitat's VLNEpisode(**episode) dies on any episode key the GT file doesn't have."""
    allowed = set(json.load(gzip.open(GT_PATHS[split]))["episodes"][0])
    extra = set().union(*(e.keys() for e in json.load(gzip.open(path))["episodes"])) - allowed
    assert not extra, f"{path}: episode keys {sorted(extra)} not in GT -> habitat eval would crash"


def deploy(cfg):
    label, ghash = cfg["label"], gen_hash(cfg)
    for split in SPLITS:
        src = OUT_ROOT / label / f"{split}.json.gz"
        dst = HABITAT_BASE / split / f"{split}_{label}.json.gz"
        check_episode_keys(src, split)
        if dst.exists():  # never overwrite; same hash = already deployed
            old = json.load(gzip.open(dst)).get("_generation_meta", {}).get("gen_hash")
            assert old == ghash, f"{dst} exists with gen_hash {old} != {ghash}; not overwriting"
            check_episode_keys(dst, split)
        else:
            shutil.copy(src, dst)
        (SHARE / split).mkdir(parents=True, exist_ok=True)
        if not (SHARE / split / dst.name).exists():
            shutil.copy(dst, SHARE / split / dst.name)
        print(f"  deployed {dst} (+ data-vol2 copy)")


def update_registry(label, entry):
    reg = yaml.safe_load(open(REGISTRY)) if REGISTRY.exists() else {}
    reg = reg or {}
    reg[label] = deep_merge(reg.get(label, {}), entry)
    yaml.safe_dump(reg, open(REGISTRY, "w"), sort_keys=False, allow_unicode=True, width=200)


# --------------------------------------------------------------------------- main
def selfcheck():
    cfg = load_label("v278fix")
    g3 = {"per_frame": {"0": {"room": "hall", "landmarks": ["a", "b"], "direction": "left"}},
          "goal_landmark": {"description": "the sofa"}, "motion_text": "Turn left."}
    ours, stop = build_prompt(cfg, {"episode_id": 1, "trajectory_id": 1}, g3, [])
    scene = "  [Frame 0] Hall — a, b, dir: left"
    kemal = rac.INSTRUCTION_GENERATION_PROMPT.format(motion_text="Turn left.", scene_context=scene, goal_landmark="the sofa")
    assert ours == kemal and stop == "must", "baseline prompt drifted from run_annotator_configured.py"
    print("selfcheck ok: v278fix gate4 prompt == run_annotator_configured.INSTRUCTION_GENERATION_PROMPT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?")
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--subset", type=int, help="first N episodes per split -> screen/ dir, no deploy")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    cfg = load_label(a.label)
    assert cfg.get("generator") == "annotator", f"{a.label}: generator={cfg.get('generator')} is not run by annotate.py"
    workers = a.workers or cfg.get("workers", 64)
    print(f"label={a.label} gen_hash={gen_hash(cfg)} gate3_hash={gate3_hash(cfg)} splits={a.splits} subset={a.subset}")
    outs = {s: run_split(cfg, s, workers, a.subset) for s in a.splits}
    m = {s: metrics(p, s) for s, p in outs.items()}
    for s, v in m.items():
        print(f"  [{s}] {v}")
    if a.subset:
        update_registry(a.label, {"screen": {"subset": a.subset, "gen_hash": gen_hash(cfg), "metrics": m}})
        return
    entry = {"base": yaml.safe_load(open(LABEL_DIR / f"{a.label}.yaml")).get("base"), "gen_hash": gen_hash(cfg),
             "status": "generated", "metrics": m}
    if all((OUT_ROOT / a.label / f"{s}.json.gz").exists() for s in SPLITS):
        deploy(cfg)
        entry["status"] = "deployed"
    update_registry(a.label, entry)


if __name__ == "__main__":
    main()
