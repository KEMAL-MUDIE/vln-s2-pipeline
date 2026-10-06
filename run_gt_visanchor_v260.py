"""
Gate 4 v260: v259 base + rewrite EMBEDDED stop verbs in navigation sentences.

Problem: v259 APPEND episodes keep "walk forward and stop near the rug. Stop near the armchair."
         Model may stop AT THE RUG (wrong) instead of treating it as a spatial waypoint.

Fix: Rewrite embedded stop verbs to navigation verbs ("pass", "walk past", "go past"):
     "walk forward and PASS near the rug. Stop near the grey armchair."
     Spatial landmark (rug) preserved. Stop semantic removed from navigation body.

Only rewrites embedded stops in the NAVIGATION body, never the final stop sentence.
For REPLACE episodes (pure stop last), no change needed.
"""
import gzip, json, re, sys
from pathlib import Path

ROOT     = Path(__file__).parent
GT_FILE  = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz")
V238_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v238.json.gz")
OUT_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v260.json.gz")
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

# Matches "stop [prep] the X" or "stop [prep] X" embedded in navigation sentences
# We want to rewrite this to "pass [prep] the X" to keep the spatial landmark
EMBEDDED_STOP = re.compile(
    r'\bstop(?:ped)?\s+(near|in|at|by|on|next\s+to|beside|next)\b',
    re.IGNORECASE
)


def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def is_pure_stop(sentence):
    return not bool(NAV_VERBS.search(sentence))


def rewrite_embedded_stops(sent):
    """In a navigation sentence, replace 'stop near/at/by/in X' → 'pass near/at/by/in X'.
    Preserves spatial landmark but removes stop semantics."""
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
    """v260 logic:
    - REPLACE: pure-stop last sentence → replace with visual stop
    - APPEND: navigation-last sentence → rewrite embedded stops in body, then APPEND visual stop
    """
    sents = split_sentences(gt_text)
    stop_sent = build_stop_sentence(gt_text, visual_stop)

    if len(sents) == 0:
        return stop_sent
    if len(sents) == 1:
        return sents[0].rstrip(' .!?') + '. ' + stop_sent

    last = sents[-1]

    if is_pure_stop(last):
        # REPLACE: remove pure-stop last sentence, add visual stop
        body = ' '.join(sents[:-1])
        return body + ' ' + stop_sent
    else:
        # APPEND: rewrite embedded stops in ENTIRE text, then append visual stop
        # Rewrite each sentence's embedded stops to navigation verbs
        rewritten_sents = []
        for s in sents:
            # Only rewrite if it has nav verbs AND embedded stop (not standalone stop)
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

    preps = {}
    replace_count = append_count = rewritten_count = 0
    new_eps = []

    for ep in gt_eps:
        tid_str = str(ep["trajectory_id"])
        gt_text = ep["instruction"]["instruction_text"]
        visual_stop = ck.get(tid_str, {}).get("visual_stop", "Stop here.")

        sents = split_sentences(gt_text)
        if len(sents) > 1 and is_pure_stop(sents[-1]):
            replace_count += 1
            was_rewritten = False
        else:
            append_count += 1
            # Check if any sentence has embedded stop
            was_rewritten = any(
                NAV_VERBS.search(s) and EMBEDDED_STOP.search(s) for s in sents
            )
            if was_rewritten:
                rewritten_count += 1

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
    print(f"Strategy: REPLACE={replace_count}, APPEND={append_count} ({rewritten_count} with embedded-stop rewrite)", flush=True)
    top = sorted(preps.items(), key=lambda x: -x[1])[:8]
    print(f"Stop prep distribution: {top}", flush=True)
    # Show before/after for rewritten examples
    shown = 0
    for ep in new_eps:
        gt_text = ep["instruction"]["instruction_text"]
        if "pass near" in gt_text or "pass by" in gt_text or "pass at" in gt_text:
            if shown < 4:
                print(f"  rewritten: {gt_text[:220]}", flush=True)
                shown += 1

if __name__ == "__main__":
    main()
