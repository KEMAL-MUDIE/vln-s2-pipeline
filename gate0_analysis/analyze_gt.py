#!/usr/bin/env python3
"""
Gate 0: GT VLN-CE Dataset Analysis
Extracts statistics and style patterns from GT episodes to:
  - Understand what "GT quality" means quantitatively
  - Build quality metrics for Gate 7 evaluation
  - Identify landmark categories and instruction patterns
"""
import gzip
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

GT_PATH = "/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz"
OUTPUT_DIR = Path(__file__).parent


def quaternion_to_heading_deg(q):
    """Extract yaw (heading) in degrees from [qx, qy, qz, qw] quaternion."""
    qx, qy, qz, qw = q
    # Yaw around Y-axis (Habitat uses Y-up)
    siny_cosp = 2 * (qw * qy + qz * qx)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def heading_between(p1, p2):
    """Compute heading angle (degrees) from p1 to p2 in XZ plane."""
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2]
    return math.degrees(math.atan2(dx, -dz))


def angle_diff(a, b):
    """Signed angle difference a→b in (-180, 180]."""
    d = (b - a + 180) % 360 - 180
    return d


def segment_dist(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def analyze_path(reference_path, turn_threshold=30.0):
    """Return motion primitives for a reference path."""
    if len(reference_path) < 2:
        return []
    primitives = []
    prev_heading = heading_between(reference_path[0], reference_path[1])
    seg_dist = segment_dist(reference_path[0], reference_path[1])

    for i in range(1, len(reference_path) - 1):
        cur_heading = heading_between(reference_path[i], reference_path[i + 1])
        turn = angle_diff(prev_heading, cur_heading)
        d = segment_dist(reference_path[i], reference_path[i + 1])

        if abs(turn) > turn_threshold:
            primitives.append({"type": "straight", "distance_m": round(seg_dist, 2)})
            direction = "left" if turn < 0 else "right"
            primitives.append({"type": f"{direction}_turn", "angle_deg": round(abs(turn), 1)})
            seg_dist = d
        else:
            seg_dist += d
        prev_heading = cur_heading

    primitives.append({"type": "straight", "distance_m": round(seg_dist, 2)})
    primitives.append({"type": "stop"})
    return primitives


def extract_landmark_words(text):
    """Simple heuristic: nouns after 'the', 'a', 'an', 'past', 'near', 'by'."""
    # Remove trailing whitespace/period
    text = text.strip().rstrip(".")
    # Common VLN landmark patterns
    patterns = [
        r"\bthe (\w+(?:\s+\w+)?)\b",
        r"\bpast (?:the )?(\w+(?:\s+\w+)?)\b",
        r"\bnear (?:the )?(\w+(?:\s+\w+)?)\b",
        r"\bby (?:the )?(\w+(?:\s+\w+)?)\b",
        r"\bthrough (?:the )?(\w+(?:\s+\w+)?)\b",
        r"\binto (?:the )?(\w+(?:\s+\w+)?)\b",
    ]
    landmarks = []
    for pat in patterns:
        for m in re.finditer(pat, text.lower()):
            word = m.group(1).strip()
            if len(word) > 2 and word not in {"end", "top", "way", "left", "right", "front", "back", "side"}:
                landmarks.append(word)
    return landmarks


def count_sentences(text):
    return len([s for s in re.split(r"[.!?]+", text.strip()) if s.strip()])


def main():
    print("Loading GT dataset...")
    with gzip.open(GT_PATH, "rt") as f:
        data = json.load(f)

    episodes = data["episodes"]
    vocab = data.get("instruction_vocab", {})
    print(f"Episodes: {len(episodes)}")
    print(f"Vocabulary size: {vocab.get('num_vocab', 'N/A')}")

    # Instruction statistics
    texts = [e["instruction"]["instruction_text"] for e in episodes]
    word_counts = [len(t.split()) for t in texts]
    sentence_counts = [count_sentences(t) for t in texts]
    char_counts = [len(t) for t in texts]

    print("\n=== INSTRUCTION STATISTICS ===")
    for name, vals in [("word_count", word_counts), ("sentence_count", sentence_counts), ("char_count", char_counts)]:
        print(
            f"  {name}: min={min(vals)}, max={max(vals)}, "
            f"mean={statistics.mean(vals):.1f}, median={statistics.median(vals):.1f}, "
            f"stdev={statistics.stdev(vals):.1f}"
        )

    # Path statistics
    paths = [e["reference_path"] for e in episodes]
    path_wpt_counts = [len(p) for p in paths]
    geo_dists = [e["info"]["geodesic_distance"] for e in episodes]

    print("\n=== PATH STATISTICS ===")
    print(
        f"  waypoints: min={min(path_wpt_counts)}, max={max(path_wpt_counts)}, "
        f"mean={statistics.mean(path_wpt_counts):.1f}"
    )
    print(
        f"  geodesic_distance(m): min={min(geo_dists):.2f}, max={max(geo_dists):.2f}, "
        f"mean={statistics.mean(geo_dists):.2f}"
    )

    # Turn distribution
    all_primitives = [analyze_path(e["reference_path"]) for e in episodes]
    turn_counts = Counter()
    for prims in all_primitives:
        for p in prims:
            turn_counts[p["type"]] += 1

    print("\n=== MOTION PRIMITIVE DISTRIBUTION ===")
    for k, v in sorted(turn_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # Landmark analysis
    all_landmarks = []
    for t in texts:
        all_landmarks.extend(extract_landmark_words(t))
    top_landmarks = Counter(all_landmarks).most_common(30)

    print("\n=== TOP 30 LANDMARK WORDS IN GT INSTRUCTIONS ===")
    for word, count in top_landmarks:
        print(f"  {word}: {count}")

    # Instruction style patterns
    stop_patterns = Counter()
    for t in texts:
        t_lower = t.lower()
        if "stop" in t_lower:
            stop_patterns["stop"] += 1
        if "wait" in t_lower:
            stop_patterns["wait"] += 1
        if "stand" in t_lower:
            stop_patterns["stand"] += 1
        if "halt" in t_lower:
            stop_patterns["halt"] += 1

    print("\n=== STOP CONDITION PATTERNS ===")
    for k, v in stop_patterns.most_common():
        print(f"  '{k}': {v}/{len(texts)} ({100*v/len(texts):.1f}%)")

    # Common turn phrases
    turn_phrases = Counter()
    for t in texts:
        for phrase in ["turn left", "turn right", "make a left", "make a right",
                       "go left", "go right", "veer left", "veer right", "bear left", "bear right"]:
            if phrase in t.lower():
                turn_phrases[phrase] += 1

    print("\n=== TURN PHRASE DISTRIBUTION ===")
    for k, v in turn_phrases.most_common():
        print(f"  '{k}': {v}")

    # Sample instructions by word count range
    print("\n=== SAMPLE INSTRUCTIONS BY WORD COUNT ===")
    short = [t for t in texts if len(t.split()) < 15]
    medium = [t for t in texts if 15 <= len(t.split()) < 30]
    long_ = [t for t in texts if len(t.split()) >= 30]
    print(f"Short (<15 words): {len(short)} eps — example: {short[0][:100] if short else 'none'}")
    print(f"Medium (15-29): {len(medium)} eps — example: {medium[0][:120] if medium else 'none'}")
    print(f"Long (≥30): {len(long_)} eps — example: {long_[0][:150] if long_ else 'none'}")

    # Save statistics
    stats = {
        "n_episodes": len(episodes),
        "vocabulary_size": vocab.get("num_vocab"),
        "instruction": {
            "word_count": {"min": min(word_counts), "max": max(word_counts),
                           "mean": round(statistics.mean(word_counts), 2),
                           "median": statistics.median(word_counts),
                           "stdev": round(statistics.stdev(word_counts), 2)},
            "sentence_count": {"min": min(sentence_counts), "max": max(sentence_counts),
                                "mean": round(statistics.mean(sentence_counts), 2)},
        },
        "path": {
            "waypoint_count": {"min": min(path_wpt_counts), "max": max(path_wpt_counts),
                                "mean": round(statistics.mean(path_wpt_counts), 2)},
            "geodesic_distance_m": {"min": round(min(geo_dists), 3), "max": round(max(geo_dists), 3),
                                     "mean": round(statistics.mean(geo_dists), 3)},
        },
        "motion_primitives": dict(turn_counts),
        "top_landmarks": dict(top_landmarks),
        "stop_patterns": dict(stop_patterns),
        "turn_phrases": dict(turn_phrases),
    }
    out_path = OUTPUT_DIR / "gt_statistics.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
