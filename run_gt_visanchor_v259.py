"""
Gate 4 v259: v253-style APPEND (keeps embedded stop clauses as spatial landmarks)
              + v258-style stop sentence (GT preposition + visual landmark noun).

Key insight from v253 results (62.6% at 214 eps, passing 62% threshold):
- "walk forward and stop near the rug" → "stop near the rug" is a SPATIAL LANDMARK
- Stripping it (v255-v258 approach) LOSES navigation information
- v253 keeps it (preserving the rug as a navigation waypoint) and APPENDS visual stop

v259 combines:
- v253's combination logic: KEEP full navigation sentence + APPEND stop sentence
- v258's stop sentence: GT preposition + visual landmark noun (not "in front of")

Expected: v253 (~63%) + correct vocab benefit (~1-2pp) → ~64-65%
0 extra Gemma API calls. Reuses v254 checkpoint.
"""
import gzip, json, re, sys
from pathlib import Path

ROOT     = Path(__file__).parent
GT_FILE  = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz")
V238_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v238.json.gz")
OUT_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v259.json.gz")
V254_CKPT = ROOT / "outputs" / "gate4_v254_checkpoint.json"

NAV_VERBS = re.compile(
    r'\b(walk|go|turn|head|proceed|continue|pass|enter|exit|cross|climb|descend|move|take|'
    r'travel|face|stand|leave|keep|step|follow|approach|reach|veer|navigate)\b',
    re.IGNORECASE
)

STOP_PREP = re.compile(
    r'\bstop(?:ped)?\s+(near|in|at|by|on|next\s+to|inside|just|once|there|here|when)\b',
    re.IGNORECASE
)

VISUAL_NOUN = re.compile(
    r'^Stop\s+(?:near|in|at|by|on|next\s+to|inside|in\s+front\s+of)\s+(?:the\s+)?(.+?)\.?\s*$',
    re.IGNORECASE
)


def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def is_pure_stop(sentence):
    return not bool(NAV_VERBS.search(sentence))


def extract_gt_stop_prep(gt_text):
    """Extract stop preposition from GT instruction (any sentence, prefer last)."""
    for sent in reversed(split_sentences(gt_text)):
        m = STOP_PREP.search(sent)
        if m:
            return m.group(1).lower().rstrip()
    return None


def extract_visual_noun(visual_stop):
    """Extract landmark noun from visual stop sentence."""
    m = VISUAL_NOUN.match(visual_stop.strip())
    if m:
        return m.group(1).strip().rstrip('.')
    # Fallback: strip "Stop [prep] [the]"
    return re.sub(r'^Stop\s+\S+\s+(?:the\s+)?', '', visual_stop, flags=re.I).rstrip('.')


def build_stop_sentence(gt_text, visual_stop):
    """GT preposition + visual noun → stop sentence."""
    gt_prep = extract_gt_stop_prep(gt_text)
    visual_noun = extract_visual_noun(visual_stop)
    if not visual_noun:
        return visual_stop
    if gt_prep in ('there', 'here', 'when', 'once', 'just', None):
        return f"Stop near the {visual_noun}."
    if gt_prep == 'next to':
        return f"Stop next to the {visual_noun}."
    if gt_prep == 'inside':
        return f"Stop inside the {visual_noun}."
    return f"Stop {gt_prep} the {visual_noun}."


def combine(gt_text, visual_stop):
    """v253-style: keep full GT (including embedded stop clauses) + APPEND stop sentence."""
    sents = split_sentences(gt_text)
    stop_sent = build_stop_sentence(gt_text, visual_stop)

    if len(sents) == 0:
        return stop_sent
    if len(sents) == 1:
        return sents[0].rstrip(' .!?') + '. ' + stop_sent

    last = sents[-1]
    body = ' '.join(sents[:-1])

    if is_pure_stop(last):
        # Last sentence is pure stop → REPLACE with GT-prep + visual noun
        return body + ' ' + stop_sent
    else:
        # Last sentence has navigation (e.g., "walk forward and stop near the rug")
        # KEEP FULL (preserves "near the rug" as spatial landmark) + APPEND visual stop
        return gt_text.rstrip(' .!?') + '. ' + stop_sent


def main():
    print("Loading GT...", flush=True)
    with gzip.open(GT_FILE) as f:
        gt_eps = json.load(f)["episodes"]

    with open(V254_CKPT) as f:
        ck = json.load(f)
    print(f"Loaded v254 checkpoint: {len(ck)} trajs", flush=True)

    preps = {}
    replace_count = append_count = 0
    new_eps = []

    for ep in gt_eps:
        tid_str = str(ep["trajectory_id"])
        gt_text = ep["instruction"]["instruction_text"]
        visual_stop = ck.get(tid_str, {}).get("visual_stop", "Stop here.")

        sents = split_sentences(gt_text)
        if len(sents) > 1 and is_pure_stop(sents[-1]):
            replace_count += 1
        else:
            append_count += 1

        new_text = combine(gt_text, visual_stop)
        stop_sent = build_stop_sentence(gt_text, visual_stop)
        m = re.match(r'^Stop\s+(\S+)', stop_sent, re.I)
        w2 = m.group(1).lower() if m else '?'
        preps[w2] = preps.get(w2, 0) + 1

        new_ep = dict(ep)
        new_ep["instruction"] = dict(ep["instruction"])
        new_ep["instruction"]["instruction_text"] = new_text
        new_ep["instruction"]["instruction_tokens"] = None
        new_eps.append(new_ep)

    with gzip.open(V238_FILE) as f:
        vocab = json.load(f)["instruction_vocab"]
    with gzip.open(OUT_FILE, "wt") as f:
        json.dump({"instruction_vocab": vocab, "episodes": new_eps}, f)

    lens = [len(e["instruction"]["instruction_text"].split()) for e in new_eps]
    print(f"Written: {OUT_FILE}", flush=True)
    print(f"eps={len(new_eps)}, avg={sum(lens)/len(lens):.1f}w", flush=True)
    print(f"Strategy: REPLACE={replace_count} ({replace_count/len(new_eps)*100:.0f}%), APPEND={append_count} ({append_count/len(new_eps)*100:.0f}%)", flush=True)
    top = sorted(preps.items(), key=lambda x: -x[1])[:8]
    print(f"Stop prep distribution: {top}", flush=True)
    for ep in new_eps[:4]:
        print(f"  ep{ep['episode_id']}: {ep['instruction']['instruction_text'][:200]}", flush=True)

if __name__ == "__main__":
    main()
