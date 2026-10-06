"""
Gate 4 v256: Same as v255 but with expanded NAV_VERBS regex.

Fix: v255 missed 'travel', 'face', 'stand', 'leave', inflected forms like 'walking'.
Result: 0 empty-body cases (was 10 in v255). Uses v254 visual stop checkpoint (0 API calls).
"""
import gzip, json, re, sys
from pathlib import Path

ROOT    = Path(__file__).parent
GT_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz")
V238_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v238.json.gz")
OUT_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v256.json.gz")
V254_CKPT = ROOT / "outputs" / "gate4_v254_checkpoint.json"

NAV_VERBS = re.compile(
    r'\b(?:walk|walks|walking|walked|go|goes|going|went|turn|turns|turning|turned|'
    r'head|heads|heading|headed|proceed|proceeds|proceeding|continued?|continu(?:e|es|ing)|'
    r'pass(?:es|ing|ed)?|enter(?:s|ing|ed)?|exit(?:s|ing|ed)?|'
    r'cross(?:es|ing|ed)?|climb(?:s|ing|ed)?|descend(?:s|ing|ed)?|'
    r'mov(?:e|es|ing|ed)|tak(?:e|es|ing|en|took)|travel(?:s|ing|ed|led)?|'
    r'face(?:s|d|ing)?|stand(?:s|ing|stood)?|leav(?:e|es|ing)|left|'
    r'keep(?:s|ing|kept)?|step(?:s|ping|ped)?|follow(?:s|ing|ed)?|'
    r'approach(?:es|ing|ed)?|reach(?:es|ing|ed)?|veer(?:s|ing|ed)?|'
    r'navigat(?:e|es|ing|ed)?)\b',
    re.IGNORECASE
)

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

def strip_stop_clause(sentence):
    cleaned = re.sub(r'[,\s]+(?:and\s+)?(?:stop(?:ped)?|wait)\b.*$', '', sentence, flags=re.I)
    return cleaned.strip(' ,;')

def build_nav_body(gt_text):
    parts = []
    for sent in split_sentences(gt_text):
        if not NAV_VERBS.search(sent):
            continue
        nav = strip_stop_clause(sent)
        if nav:
            parts.append(nav)
        else:
            m = re.match(r'^(\S+(?:\s+\S+){0,8})', sent, re.I)
            if m:
                parts.append(m.group(1).strip())
    return ' '.join(parts)

def combine(gt_text, visual_stop):
    nav = build_nav_body(gt_text)
    if nav:
        return nav.rstrip(' .!?') + '. ' + visual_stop
    return visual_stop

def main():
    print("Loading GT...", flush=True)
    with gzip.open(GT_FILE) as f:
        gt_eps = json.load(f)["episodes"]

    if not V254_CKPT.exists():
        print("ERROR: v254 checkpoint not found", flush=True); sys.exit(1)
    with open(V254_CKPT) as f:
        ck = json.load(f)
    print(f"Loaded v254 checkpoint: {len(ck)} trajs", flush=True)

    new_eps = []
    empty_body = 0
    for ep in gt_eps:
        tid_str = str(ep["trajectory_id"])
        gt_text = ep["instruction"]["instruction_text"]
        visual_stop = ck.get(tid_str, {}).get("visual_stop", "Stop here.")
        new_text = combine(gt_text, visual_stop)
        if not build_nav_body(gt_text):
            empty_body += 1
        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        new_ep["instruction"]["instruction_text"] = new_text
        new_ep["instruction"]["instruction_tokens"] = None
        new_eps.append(new_ep)

    with gzip.open(V238_FILE) as f:
        vocab = json.load(f)["instruction_vocab"]
    out = {"instruction_vocab": vocab, "episodes": new_eps}
    with gzip.open(OUT_FILE, "wt") as f:
        json.dump(out, f)

    lens = [len(e["instruction"]["instruction_text"].split()) for e in new_eps]
    print(f"Written: {OUT_FILE}", flush=True)
    print(f"eps={len(new_eps)}, empty_body={empty_body}, avg_words={sum(lens)/len(lens):.1f}", flush=True)
    for ep in new_eps[:4]:
        print(f"  ep{ep['episode_id']}: {ep['instruction']['instruction_text'][:180]}", flush=True)

if __name__ == "__main__":
    main()
