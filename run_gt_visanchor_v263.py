"""
Gate 4 v263: v262 selectivity + v261 "stop when you see X" format.

v262 recap: 87% kept as pure GT, 13% modified (vague stops only).
Problem with v262's modified 13%: "Stop near the X" is a distance target — model
must estimate proximity to X, which is uncertain.

v263 improvement: For the modified 13% (vague stops where visual anchor is used),
change format to "Stop when you see the X" — a RECOGNITION trigger.
- Model stops when it visually identifies X, not when it estimates it's close
- R2R training data uses "when you see" and "when you reach" constructions often
- Keeps the same v262 selectivity for non-modified episodes (87% pure GT)

v263 = v262 logic + v261's "stop when you see X" for modified episodes only
"""
import gzip, json, re
from pathlib import Path

ROOT     = Path(__file__).parent
GT_FILE  = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz")
V238_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v238.json.gz")
OUT_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v263.json.gz")
V254_CKPT = ROOT / "outputs" / "gate4_v254_checkpoint.json"

NAV_VERBS = re.compile(
    r'\b(walk|go|turn|head|proceed|continue|pass|enter|exit|cross|climb|descend|move|take|'
    r'travel|face|stand|leave|keep|step|follow|approach|reach|veer|navigate)\b',
    re.IGNORECASE
)

SPECIFIC_OBJECTS = re.compile(
    r'\b(couch|sofa|chair|chairs|table|tables|counter|counters|stool|stools|ottoman|bed|'
    r'door|doorway|window|fireplace|bookshelf|bookshelves|dresser|shelf|shelves|cabinet|'
    r'lamp|plant|potted|stairs|staircase|stairway|railing|elevator|television|tv|rug|carpet|'
    r'mirror|desk|bench|hallway|bathroom|bedroom|kitchen|living\s+room|dining|'
    r'pool\s+table|sink|toilet|bathtub|refrigerator|fridge|pillar|column|island|'
    r'picture|painting|artwork|bridge|tree|fountain|sign|exercise|workout|fitness|'
    r'lobby|atrium|foyer|entrance|exit|archway|alcove)\b',
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


def has_specific_stop_object(text):
    return bool(SPECIFIC_OBJECTS.search(text))


def rewrite_embedded_stops(sent):
    return EMBEDDED_STOP.sub(lambda m: f"pass {m.group(1)}", sent)


def extract_visual_noun(visual_stop):
    m = VISUAL_NOUN.match(visual_stop.strip())
    if m:
        return m.group(1).strip().rstrip('.')
    noun = re.sub(r'^Stop\s+\S+\s+(?:the\s+)?', '', visual_stop, flags=re.I).rstrip('.')
    return noun if noun and noun.lower() not in ('here', 'there', 'you', 'it') else None


def build_stop_sentence(visual_stop):
    """Build 'Stop when you see the X' from visual anchor."""
    visual_noun = extract_visual_noun(visual_stop)
    if not visual_noun:
        return "Stop here."
    return f"Stop when you see the {visual_noun}."


def combine(gt_text, visual_stop):
    """v263: v262 selectivity + v261 'stop when you see' format."""
    sents = split_sentences(gt_text)
    if len(sents) == 0:
        return gt_text
    last = sents[-1]

    if is_pure_stop(last):
        if has_specific_stop_object(last):
            return gt_text  # KEEP specific GT stop
        else:
            # REPLACE vague stop with "stop when you see X"
            stop_sent = build_stop_sentence(visual_stop)
            if len(sents) == 1:
                return stop_sent
            return ' '.join(sents[:-1]) + ' ' + stop_sent
    else:
        if has_specific_stop_object(gt_text):
            return gt_text  # KEEP GT when it names the stop target
        else:
            # APPEND "stop when you see X" to vague nav-last
            rewritten_sents = []
            for s in sents:
                if NAV_VERBS.search(s) and EMBEDDED_STOP.search(s):
                    rewritten_sents.append(rewrite_embedded_stops(s))
                else:
                    rewritten_sents.append(s)
            rewritten_text = ' '.join(rewritten_sents)
            stop_sent = build_stop_sentence(visual_stop)
            return rewritten_text.rstrip(' .!?') + '. ' + stop_sent


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
    print(f"Written: {OUT_FILE}", flush=True)
    print(f"eps={len(new_eps)}, avg={sum(lens)/len(lens):.1f}w", flush=True)
    print(f"kept={kept} ({kept/len(new_eps)*100:.0f}%), modified={modified} ({modified/len(new_eps)*100:.0f}%)", flush=True)
    for ep in new_eps[:3]:
        t = ep['instruction']['instruction_text']
        print(f"  {t[:200]}", flush=True)

if __name__ == "__main__":
    main()
