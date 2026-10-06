"""
Gate 4 v264: v262 + fixes for SPECIFIC_OBJECTS and NAV_VERBS.

Bugs found in v262 via analysis of 204 vague-stop episodes:
1. SPECIFIC_OBJECTS missing: tub, shell, mat, curtain, clock, rack, frame, fan, cart, box
2. NAV_VERBS missing: make (for "make a left/right"), wait (treated as nav-stop sometimes)
3. Edge case: "towards the X" (no verb) misclassified as pure stop

Fixes:
1. Expanded SPECIFIC_OBJECTS to cover ~30 more object types
2. Added "make" to NAV_VERBS
3. Better handling of incomplete sentences

Expected: ~10 fewer episodes incorrectly replaced → cleaner 13% modified set
"""
import gzip, json, re
from pathlib import Path

ROOT     = Path(__file__).parent
GT_FILE  = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz")
V238_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v238.json.gz")
OUT_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v264.json.gz")
V254_CKPT = ROOT / "outputs" / "gate4_v254_checkpoint.json"

# EXPANDED — added tub, shell, mat, curtain, clock, rack, frame, fan, bench variants
NAV_VERBS = re.compile(
    r'\b(walk|go|turn|head|proceed|continue|pass|enter|exit|cross|climb|descend|move|take|'
    r'travel|face|stand|leave|keep|step|follow|approach|reach|veer|navigate|make)\b',
    re.IGNORECASE
)

SPECIFIC_OBJECTS = re.compile(
    r'\b(couch|sofa|chair|chairs|armchair|armchairs|table|tables|counter|counters|stool|stools|'
    r'ottoman|bed|beds|door|doors|doorway|window|windows|fireplace|bookshelf|bookshelves|dresser|'
    r'shelf|shelves|cabinet|cabinets|lamp|lamps|plant|plants|potted|stairs|staircase|stairway|'
    r'railing|elevator|television|tv|rug|rugs|carpet|mirror|desk|bench|benches|'
    r'hallway|bathroom|bedroom|kitchen|living\s+room|dining|pool\s+table|sink|toilet|'
    r'bathtub|tub|refrigerator|fridge|pillar|column|island|picture|painting|artwork|'
    r'bridge|tree|fountain|sign|exercise|workout|fitness|lobby|atrium|foyer|entrance|exit|'
    r'archway|alcove|mat|curtain|curtains|clock|rack|frame|fan|cart|box|'
    r'shell|shells|vase|statue|sculpture|column|sofa|loveseat|couch|bookcase|'
    r'recliner|hutch|console|credenza|wardrobe|armoire|chest|nightstand|headboard|'
    r'banister|newel|balcony|porch|terrace|deck|landing|foyer|vestibule|'
    r'eye\s+chart|whiteboard|chalkboard|bulletin)\b',
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

STOP_PREP = re.compile(
    r'\bstop(?:ped)?\s+(near|in|at|by|on|next\s+to|inside|just|once|there|here|when)\b',
    re.IGNORECASE
)


def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def is_pure_stop(sentence):
    return not bool(NAV_VERBS.search(sentence))


def has_specific_stop_object(text):
    return bool(SPECIFIC_OBJECTS.search(text))


def rewrite_embedded_stops(sent):
    return EMBEDDED_STOP.sub(lambda m: f"pass {m.group(1)}", sent)


def extract_gt_stop_prep(gt_text):
    for sent in reversed(split_sentences(gt_text)):
        m = STOP_PREP.search(sent)
        if m:
            return m.group(1).lower().rstrip()
    return None


def extract_visual_noun(visual_stop):
    m = VISUAL_NOUN.match(visual_stop.strip())
    if m:
        return m.group(1).strip().rstrip('.')
    return re.sub(r'^Stop\s+\S+\s+(?:the\s+)?', '', visual_stop, flags=re.I).rstrip('.')


def build_stop_sentence(gt_text, visual_stop):
    """GT preposition + visual noun (same as v262)."""
    gt_prep = extract_gt_stop_prep(gt_text)
    visual_noun = extract_visual_noun(visual_stop)
    if not visual_noun or visual_noun.lower() in ('here', 'there', 'you', 'it'):
        return visual_stop
    if gt_prep in ('there', 'here', 'when', 'once', 'just', None):
        return f"Stop near the {visual_noun}."
    if gt_prep == 'next to':
        return f"Stop next to the {visual_noun}."
    if gt_prep == 'inside':
        return f"Stop inside the {visual_noun}."
    return f"Stop {gt_prep} the {visual_noun}."


def combine(gt_text, visual_stop):
    """v264: v262 logic with expanded SPECIFIC_OBJECTS + NAV_VERBS."""
    sents = split_sentences(gt_text)
    if len(sents) == 0:
        return gt_text
    last = sents[-1]

    if is_pure_stop(last):
        if has_specific_stop_object(last):
            return gt_text  # KEEP
        else:
            stop_sent = build_stop_sentence(gt_text, visual_stop)
            if len(sents) == 1:
                return stop_sent
            return ' '.join(sents[:-1]) + ' ' + stop_sent
    else:
        if has_specific_stop_object(gt_text):
            return gt_text  # KEEP
        else:
            rewritten = []
            for s in sents:
                if NAV_VERBS.search(s) and EMBEDDED_STOP.search(s):
                    rewritten.append(rewrite_embedded_stops(s))
                else:
                    rewritten.append(s)
            stop_sent = build_stop_sentence(gt_text, visual_stop)
            return ' '.join(rewritten).rstrip(' .!?') + '. ' + stop_sent


def main():
    print("Loading GT...", flush=True)
    with gzip.open(GT_FILE) as f:
        gt_eps = json.load(f)["episodes"]
    with open(V254_CKPT) as f:
        ck = json.load(f)
    print(f"Loaded checkpoint: {len(ck)} trajs", flush=True)

    kept = modified = 0
    new_eps = []
    for ep in gt_eps:
        tid_str = str(ep["trajectory_id"])
        gt_text = ep["instruction"]["instruction_text"]
        visual_stop = ck.get(tid_str, {}).get("visual_stop", "Stop here.")
        new_text = combine(gt_text, visual_stop)
        if new_text == gt_text:
            kept += 1
        else:
            modified += 1
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
    total = len(new_eps)
    print(f"Written: {OUT_FILE}, eps={total}, avg={sum(lens)/total:.1f}w", flush=True)
    print(f"kept={kept} ({kept/total*100:.0f}%), modified={modified} ({modified/total*100:.0f}%)", flush=True)

if __name__ == "__main__":
    main()
