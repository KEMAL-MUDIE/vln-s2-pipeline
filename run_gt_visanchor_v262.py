"""
Gate 4 v262: Conservative SELECTIVE visual anchor replacement.

KEY INSIGHT (v253 postmortem):
- Scene 8194nk5LbLH: v253 scored 4/11 = 36.4% vs v238 expected ~74%
- Root cause: Gemma visual anchors are WRONG for specific GT stops
  e.g., GT="stop behind the chair at the black table" → Gemma="Stop by the orange sofa"
  e.g., GT="stop inside the workout room" → Gemma="Stop in the hallway by the grey wall"
- poses.json goal frames don't always show the actual stop target object

v262 FIX: Selective approach
- If GT stop sentence names a SPECIFIC visual object → KEEP GT AS-IS (no anchor)
- If GT stop is VAGUE ("here", "there", "when you reach", "wait") → REPLACE with visual anchor
- For nav-last sentences with SPECIFIC stop object → KEEP GT AS-IS
- For nav-last sentences with NO specific stop object → APPEND visual anchor

This preserves the high-quality specific GT stops while only augmenting vague ones.
Expected: restores scene 8194 to ~74% while improving vague-stop episodes.
"""
import gzip, json, re
from pathlib import Path

ROOT     = Path(__file__).parent
GT_FILE  = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz")
V238_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v238.json.gz")
OUT_FILE = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_v262.json.gz")
V254_CKPT = ROOT / "outputs" / "gate4_v254_checkpoint.json"

NAV_VERBS = re.compile(
    r'\b(walk|go|turn|head|proceed|continue|pass|enter|exit|cross|climb|descend|move|take|'
    r'travel|face|stand|leave|keep|step|follow|approach|reach|veer|navigate)\b',
    re.IGNORECASE
)

# Specific visual objects that indicate a GT stop is already informative
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

# Vague stop phrases that benefit from visual anchors
VAGUE_STOP = re.compile(
    r'\b(here|there|this\s+point|this\s+spot|this\s+area|this\s+place|this\s+room|'
    r'when\s+you\s+(?:get|reach|arrive|see|are)|once\s+you|at\s+the\s+(?:end|top|bottom|'
    r'corner|entrance|exit)|in\s+front|ahead|forward)\b',
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


def has_specific_stop_object(text):
    """Does text already name a specific visual object for stopping?"""
    return bool(SPECIFIC_OBJECTS.search(text))


def is_vague_stop(sentence):
    """Is this stop sentence vague (no specific visual object)?"""
    return not has_specific_stop_object(sentence) and (
        bool(VAGUE_STOP.search(sentence)) or
        sentence.strip().rstrip('.!?').lower() in ('stop', 'wait', 'stop here', 'wait here',
                                                     'stop there', 'wait there', 'stop and wait')
    )


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
    """v262 selective logic."""
    sents = split_sentences(gt_text)

    if len(sents) == 0:
        return gt_text

    last = sents[-1]
    body_sents = sents[:-1]

    if is_pure_stop(last):
        if has_specific_stop_object(last):
            # GT already has a specific stop target — KEEP AS-IS
            return gt_text
        else:
            # GT stop is vague — REPLACE with visual anchor
            stop_sent = build_stop_sentence(gt_text, visual_stop)
            if len(body_sents) == 0:
                return stop_sent
            return ' '.join(body_sents) + ' ' + stop_sent
    else:
        # Navigation-last sentence
        if has_specific_stop_object(gt_text):
            # GT already describes the stop target somewhere — KEEP AS-IS
            return gt_text
        else:
            # No specific stop object in GT — APPEND visual anchor
            # Also rewrite any embedded stop verbs in nav body
            rewritten_sents = []
            for s in sents:
                if NAV_VERBS.search(s) and EMBEDDED_STOP.search(s):
                    rewritten_sents.append(rewrite_embedded_stops(s))
                else:
                    rewritten_sents.append(s)
            rewritten_text = ' '.join(rewritten_sents)
            stop_sent = build_stop_sentence(gt_text, visual_stop)
            return rewritten_text.rstrip(' .!?') + '. ' + stop_sent


def main():
    print("Loading GT...", flush=True)
    with gzip.open(GT_FILE) as f:
        gt_eps = json.load(f)["episodes"]

    with open(V254_CKPT) as f:
        ck = json.load(f)
    print(f"Loaded v254 checkpoint: {len(ck)} trajs", flush=True)

    kept_specific = kept_nav_specific = replaced_vague = appended_nav_vague = 0
    preps = {}
    new_eps = []

    for ep in gt_eps:
        tid_str = str(ep["trajectory_id"])
        gt_text = ep["instruction"]["instruction_text"]
        visual_stop = ck.get(tid_str, {}).get("visual_stop", "Stop here.")

        sents = split_sentences(gt_text)
        last = sents[-1] if sents else ''

        if is_pure_stop(last):
            if has_specific_stop_object(last):
                kept_specific += 1
            else:
                replaced_vague += 1
        else:
            if has_specific_stop_object(gt_text):
                kept_nav_specific += 1
            else:
                appended_nav_vague += 1

        new_text = combine(gt_text, visual_stop)

        # Track stop word distribution for modified episodes
        if new_text != gt_text:
            last_new = split_sentences(new_text)[-1] if split_sentences(new_text) else ''
            m = re.match(r'^Stop\s+(\S+)', last_new, re.I)
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
    total = len(new_eps)
    print(f"Written: {OUT_FILE}", flush=True)
    print(f"eps={total}, avg={sum(lens)/len(lens):.1f}w", flush=True)
    print(f"Strategy breakdown:", flush=True)
    print(f"  kept_specific_pure_stop={kept_specific} ({kept_specific/total*100:.0f}%)", flush=True)
    print(f"  kept_nav_has_specific={kept_nav_specific} ({kept_nav_specific/total*100:.0f}%)", flush=True)
    print(f"  replaced_vague_stop={replaced_vague} ({replaced_vague/total*100:.0f}%)", flush=True)
    print(f"  appended_nav_vague={appended_nav_vague} ({appended_nav_vague/total*100:.0f}%)", flush=True)
    top = sorted(preps.items(), key=lambda x: -x[1])[:8]
    print(f"Stop prep in modified eps: {top}", flush=True)

    # Spot-check scene 8194
    print("\n=== Scene 8194nk5LbLH spot-check ===", flush=True)
    scene_eps_new = [e for e in new_eps if '8194nk5LbLH' in e.get('scene_id','')]
    scene_eps_gt = [e for e in gt_eps if '8194nk5LbLH' in e.get('scene_id','')]
    for e_new, e_gt in zip(scene_eps_new[:6], scene_eps_gt[:6]):
        same = e_new['instruction']['instruction_text'] == e_gt['instruction']['instruction_text']
        print(f"  tid={e_new['trajectory_id']} kept={'YES' if same else 'MODIFIED'}: {e_new['instruction']['instruction_text'][:120]}", flush=True)

if __name__ == "__main__":
    main()
