#!/usr/bin/env python3
"""
v68: Fix 'destination/there/here' stop phrases in v67.

PROBLEM (v67): 179 episodes have generic stop phrases like:
  "Wait there." / "Stop there." / "Stop near the destination."
  
ROOT CAUSE: LLM generates BOTH the specific stop ("Stop at the circular tile mosaic")
  AND a generic follow-up ("Wait there."). The post-processing then:
  1. clean() or quality_ok strips the specific stop
  2. enforce_stop_phrase enforces "Wait there" as the stop (wrong)

FIX (v68): Post-processing lookup — for each episode with a generic stop,
  use Phase 1 perframe data to build the correct stop phrase.

Expected: 179 fixed episodes → better stop accuracy → higher SR.
"""
import gzip, json, re, sys
from pathlib import Path

ROOT = Path(__file__).parent
V67_PATH = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v67.json.gz"
V68_PATH = ROOT / "outputs" / "datasets" / "val_unseen_generated_gemma_visual_v68.json.gz"
PHASE1_DIR = ROOT / "outputs" / "gate3_perframe"

sys.path.insert(0, str(ROOT))
from gate5_tokenizer.tokenizer import VLNTokenizer, VOCAB_SOURCE

# Patterns to detect generic/bad stop phrases
_GENERIC_STOP_RE = re.compile(
    r'(?:Stop|Wait)\b[^.]*?\b(?:destination|there|here|area|location|arrival|spot)\b[^.]*?\.?\s*$',
    re.IGNORECASE | re.DOTALL,
)
# Also: "Stop near the room" (too generic room-only stops)
_ROOM_ONLY_STOP_RE = re.compile(
    r'(?:Stop|Wait)\s+(?:in|at|near|by)\s+the\s+(?:entryway|foyer|area|entrance|room|space|hallway|lobby|landing)\.\s*$',
    re.IGNORECASE,
)

def _choose_better_stop(ep_id: int, existing_text: str) -> str | None:
    """Return a better stop phrase from Phase 1 perframe data, or None if no improvement."""
    pf_file = PHASE1_DIR / f"episode_{int(ep_id):06d}.json"
    if not pf_file.exists():
        return None
    
    with open(pf_file) as f:
        pf = json.load(f)
    
    goal = pf.get("goal", {})
    stop_lm = (goal.get("stop_landmark", "") or goal.get("main_landmark", "")).strip()
    stop_desc = goal.get("stop_description", "").strip()
    room = goal.get("room", "").strip()
    
    # Require a real landmark (not generic)
    if not stop_lm or stop_lm.lower() in ("the destination", "destination", "the area", 
                                            "the room", "the entrance", ""):
        return None
    
    # Prefer stop_description for rich context
    if stop_desc and len(stop_desc.split()) >= 3:
        verb = "Wait" if re.search(r'\bwait\b', existing_text, re.I) else "Stop"
        return f"{verb} {stop_desc.rstrip('.').rstrip()}."
    
    # Fallback to landmark-only
    if stop_lm:
        verb = "Wait" if re.search(r'\bwait\b', existing_text, re.I) else "Stop"
        # Room context if available
        if room and room.lower() not in ("area", "room", "space", ""):
            return f"{verb} near the {stop_lm} in the {room}."
        return f"{verb} near the {stop_lm}."
    
    return None


def needs_fix(text: str) -> bool:
    """Return True if instruction has a generic/bad stop phrase."""
    return bool(_GENERIC_STOP_RE.search(text) or _ROOM_ONLY_STOP_RE.search(text))


def apply_dest_stop_fix(text: str, ep_id: int) -> tuple:
    """Fix destination/generic stop phrase. Returns (fixed_text, was_fixed)."""
    if not needs_fix(text):
        return text, False
    
    better = _choose_better_stop(ep_id, text)
    if not better:
        return text, False
    
    # Find the bad stop and replace it
    m = _GENERIC_STOP_RE.search(text)
    if not m:
        m = _ROOM_ONLY_STOP_RE.search(text)
    if not m:
        return text, False
    
    prefix = text[:m.start()].rstrip()
    fixed = (prefix + " " + better).strip()
    return fixed, True


def main():
    print("Loading v67...")
    with gzip.open(V67_PATH) as f:
        data = json.load(f)
    
    tokenizer = VLNTokenizer(VOCAB_SOURCE)
    episodes = data["episodes"]
    
    n_fixed = 0
    n_needs_fix = 0
    examples = []
    
    for ep in episodes:
        eid = ep["episode_id"]
        txt = ep["instruction"]["instruction_text"]
        
        if needs_fix(txt):
            n_needs_fix += 1
            new_txt, was_fixed = apply_dest_stop_fix(txt, eid)
            if was_fixed:
                n_fixed += 1
                if len(examples) < 3:
                    examples.append((txt[:80], new_txt[:80]))
                ep["instruction"]["instruction_text"] = new_txt
                ep["instruction"]["instruction_tokens"] = tokenizer.encode(new_txt)
    
    print(f"\n=== v68 Destination-Stop Fix ===")
    print(f"  Total episodes: {len(episodes)}")
    print(f"  Episodes with generic stop: {n_needs_fix}")
    print(f"  Fixed: {n_fixed} / {n_needs_fix}")
    print(f"\nExamples:")
    for before, after in examples:
        print(f"  BEFORE: {before}")
        print(f"  AFTER:  {after}")
        print()
    
    # Stats
    import re
    texts = [ep['instruction']['instruction_text'] for ep in episodes]
    n = len(texts)
    anchor_n = sum(1 for t in texts if re.search(r'turn\s+(left|right)\s+(at|past|through|into|around)\s+the\s+\w', t, re.I))
    stop_n = sum(1 for t in texts if re.search(r'\bstop\b', t, re.I))
    wait_n = sum(1 for t in texts if re.search(r'\bwait\b', t, re.I))
    anchor_err = abs(anchor_n/n - 0.165)
    stop_err = abs(stop_n/n - 0.508)
    wait_err = abs(wait_n/n - 0.305)
    gt_match = 1.0 - (anchor_err*2 + stop_err + wait_err)
    print(f"Post-fix stats: anchor={100*anchor_n/n:.1f}%, stop={100*stop_n/n:.1f}%, wait={100*wait_n/n:.1f}%, GT-match={gt_match:.3f}")
    print(f"  avg_words: {sum(len(t.split()) for t in texts)/n:.1f}")
    
    with gzip.open(V68_PATH, "wt") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"\nSaved: {V68_PATH}")


if __name__ == "__main__":
    main()
