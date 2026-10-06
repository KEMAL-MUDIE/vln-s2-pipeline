"""
Gate 4 v261: v260 base + "stop when you see X" format for visual stop anchor.

Hypothesis: The model decides to output STOP based on visual confirmation of a landmark.
"Stop when you see the grey armchair" may better match the model's decision process
than "Stop near the grey armchair" — it's a visual trigger rather than a spatial target.

R2R training data contains many "when you see / when you reach" stop conditions.
This may be more naturally aligned with the model's stop prediction learned from training.

v261 = v260's REPLACE/APPEND logic + "stop near" → "stop when you see" for visual anchor
"""
import gzip, json, re
from pathlib import Path

ROOT     = Path(__file__).parent
GT_FILE  = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz")
V238_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v238.json.gz")
OUT_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v261.json.gz")
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

EMBEDDED_STOP = re.compile(
    r'\bstop(?:ped)?\s+(near|in|at|by|on|next\s+to|beside|next)\b',
    re.IGNORECASE
)


def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def is_pure_stop(sentence):
    return not bool(NAV_VERBS.search(sentence))


def rewrite_embedded_stops(sent):
    return EMBEDDED_STOP.sub(lambda m: f"pass {m.group(1)}", sent)


def extract_visual_noun(visual_stop):
    m = VISUAL_NOUN.match(visual_stop.strip())
    if m:
        return m.group(1).strip().rstrip('.')
    return re.sub(r'^Stop\s+\S+\s+(?:the\s+)?', '', visual_stop, flags=re.I).rstrip('.')


def build_stop_sentence(visual_stop):
    """Build 'Stop when you see the X' format from visual stop anchor."""
    visual_noun = extract_visual_noun(visual_stop)
    if not visual_noun or visual_noun.lower() in ('here', 'there', 'you', 'it'):
        return "Stop here."
    # Use "Stop when you see the X" format
    return f"Stop when you see the {visual_noun}."


def combine(gt_text, visual_stop):
    sents = split_sentences(gt_text)
    stop_sent = build_stop_sentence(visual_stop)

    if len(sents) == 0:
        return stop_sent
    if len(sents) == 1:
        return sents[0].rstrip(' .!?') + '. ' + stop_sent

    last = sents[-1]

    if is_pure_stop(last):
        body = ' '.join(sents[:-1])
        return body + ' ' + stop_sent
    else:
        rewritten_sents = []
        for s in sents:
            if NAV_VERBS.search(s) and EMBEDDED_STOP.search(s):
                rewritten_sents.append(rewrite_embedded_stops(s))
            else:
                rewritten_sents.append(s)
        rewritten_text = ' '.join(rewritten_sents)
        return rewritten_text.rstrip(' .!?') + '. ' + stop_sent


def main():
    print("Loading GT...", flush=True)
    with gzip.open(GT_FILE) as f:
        gt_eps = json.load(f)["episodes"]

    with open(V254_CKPT) as f:
        ck = json.load(f)
    print(f"Loaded v254 checkpoint: {len(ck)} trajs", flush=True)

    new_eps = []
    replace_count = append_count = 0

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
    print(f"Strategy: REPLACE={replace_count}, APPEND={append_count}", flush=True)
    for ep in new_eps[:5]:
        print(f"  ep{ep['episode_id']}: {ep['instruction']['instruction_text'][:220]}", flush=True)

if __name__ == "__main__":
    main()
