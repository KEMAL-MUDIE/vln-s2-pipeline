"""
Gate 4 v257: v256 nav body + GT-distribution-aligned stop vocabulary.

Problem: v254 checkpoint has at(41%), by(28%), in(24%), near(5%) — biased.
GT distribution (pure-stop last sentences): in(20%), at(16%), near(8%), by(7%), on(7%).

Fix: Rule-based rephrase of v254 visual anchors:
  - "Stop by the X"  → "Stop near the X"  (by=7% in GT vs 28% in v254)
  - "Stop at the X"  → "Stop near the X" when X is furniture/object, keep for doors/stairs
  - "Stop in the X"  → keep (matches GT)
  - Others           → keep

Result: ~near(33%), at(20%), in(24%), by(5%), other — much closer to GT.
0 extra Gemma API calls. Reuses v254 checkpoint + v256 nav body logic.
"""

import gzip, json, re, sys
from pathlib import Path

ROOT    = Path(__file__).parent
GT_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz")
V238_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v238.json.gz")
OUT_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v257.json.gz")
V254_CKPT = ROOT / "outputs" / "gate4_v254_checkpoint.json"

# Same expanded NAV_VERBS as v256 (0 empty bodies)
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

# Objects where "at" → "near" makes sense (furniture, appliances, generic objects)
FURNITURE_WORDS = re.compile(
    r'\b(chair|table|sofa|couch|bed|dresser|desk|shelf|cabinet|counter|sink|'
    r'fireplace|tv|television|lamp|rug|armchair|bookcase|wardrobe|closet|'
    r'refrigerator|bathtub|toilet|mirror|window|painting|artwork|piano|bar)\b',
    re.IGNORECASE
)

# Objects where "at" should stay (architectural features)
ARCHITECTURAL_WORDS = re.compile(
    r'\b(door|stairs|staircase|stairway|elevator|wall|hallway|entrance|exit|'
    r'landing|top|bottom|end|corner|intersection|arch|threshold)\b',
    re.IGNORECASE
)


def rephrase_stop(stop: str) -> str:
    """Rephrase stop sentence to match GT vocabulary distribution."""
    m = re.match(r'^(Stop)\s+(by|at)\s+the\s+(.+)', stop, re.IGNORECASE)
    if not m:
        return stop

    prep = m.group(2).lower()
    rest = m.group(3)

    if prep == 'by':
        # "Stop by the X" → "Stop near the X" (GT: by=7%, near=8%)
        return f"Stop near the {rest}"
    elif prep == 'at':
        # "Stop at the X" → keep if architectural feature, else "Stop near the X"
        if ARCHITECTURAL_WORDS.search(rest) and not FURNITURE_WORDS.search(rest):
            return stop  # keep "Stop at the door/stairs/..."
        elif FURNITURE_WORDS.search(rest):
            return f"Stop near the {rest}"  # "Stop near the chair" better than "Stop at the chair"
        else:
            return stop  # keep other "at" cases unchanged

    return stop


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
    rephrased = rephrase_stop(visual_stop)
    if nav:
        return nav.rstrip(' .!?') + '. ' + rephrased
    return rephrased


def main():
    print("Loading GT...", flush=True)
    with gzip.open(GT_FILE) as f:
        gt_eps = json.load(f)["episodes"]

    if not V254_CKPT.exists():
        print("ERROR: v254 checkpoint not found", flush=True); sys.exit(1)
    with open(V254_CKPT) as f:
        ck = json.load(f)
    print(f"Loaded v254 checkpoint: {len(ck)} trajs", flush=True)

    # Show rephrase statistics
    rephrased_count = {"near_from_by": 0, "near_from_at_furn": 0, "kept": 0}
    new_eps = []
    empty_body = 0

    for ep in gt_eps:
        tid_str = str(ep["trajectory_id"])
        gt_text = ep["instruction"]["instruction_text"]
        visual_stop = ck.get(tid_str, {}).get("visual_stop", "Stop here.")

        # Count rephrase type
        m = re.match(r'^Stop\s+(by|at)\s+the\s+(.+)', visual_stop, re.IGNORECASE)
        if m:
            prep, rest = m.group(1).lower(), m.group(2)
            if prep == 'by':
                rephrased_count["near_from_by"] += 1
            elif prep == 'at' and FURNITURE_WORDS.search(rest) and not ARCHITECTURAL_WORDS.search(rest):
                rephrased_count["near_from_at_furn"] += 1
            else:
                rephrased_count["kept"] += 1
        else:
            rephrased_count["kept"] += 1

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
    print(f"Rephrase: near_from_by={rephrased_count['near_from_by']}, near_from_at_furn={rephrased_count['near_from_at_furn']}, kept={rephrased_count['kept']}", flush=True)

    # Show stop vocab distribution
    stops = {}
    for ep in new_eps:
        txt = ep["instruction"]["instruction_text"]
        last = re.split(r'(?<=[.!?])\s+', txt.strip())[-1].lower()
        words = last.split()
        w2 = words[1].rstrip('.,') if len(words) > 1 else '?'
        stops[w2] = stops.get(w2, 0) + 1
    top = sorted(stops.items(), key=lambda x: -x[1])[:8]
    print(f"Stop vocab distribution: {top}", flush=True)

    print("Sample instructions:", flush=True)
    for ep in new_eps[:4]:
        print(f"  ep{ep['episode_id']}: {ep['instruction']['instruction_text'][:180]}", flush=True)


if __name__ == "__main__":
    main()
