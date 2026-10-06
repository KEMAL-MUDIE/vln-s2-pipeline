#!/usr/bin/env python3
"""
Organize top 5 annotator version datasets into outputs/annotated_datasets_top5/.

For each version (v264, v272, v273, v277, v278):
  unseen_{vN}.json.gz  ← ChronoNav dataset (from /mnt/nvme0, already exists)
  seen_{vN}.json.gz    ← auto-annotated (run_annotator_configured.py --split val_seen)
  train_{vN}.json.gz   ← auto-annotated (pending GPU2 free)

Run this script to copy/link the completed outputs into the top5 folder.
Also produces a manifest.json showing status of each file.
"""
import gzip
import json
import shutil
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent
OUT = PIPELINE_ROOT / "outputs" / "annotated_datasets_top5"
OUT.mkdir(parents=True, exist_ok=True)

HABITAT_DATA = Path("/mnt/nvme0/vln_habitat/habitat_data/datasets/vln/mp3d/r2r/v1")

TOP5_VERSIONS = ["v264", "v272", "v273", "v277", "v278"]

# Sources for each version
SOURCES = {
    "v264": {
        "unseen": HABITAT_DATA / "val_unseen/val_unseen_v264.json.gz",
        "seen":   PIPELINE_ROOT / "outputs/annotated_datasets_top5/seen_v264.json.gz",
        "train":  PIPELINE_ROOT / "outputs/annotated_datasets_top5/train_v264.json.gz",
    },
    "v272": {
        "unseen": HABITAT_DATA / "val_unseen/val_unseen_v272.json.gz",
        "seen":   PIPELINE_ROOT / "outputs/datasets/val_seen_complete_v272.json.gz",
        "train":  PIPELINE_ROOT / "outputs/annotated_datasets_top5/train_v272.json.gz",
    },
    "v273": {
        "unseen": HABITAT_DATA / "val_unseen/val_unseen_v273.json.gz",
        "seen":   PIPELINE_ROOT / "outputs/annotated_datasets_top5/seen_v273.json.gz",
        "train":  PIPELINE_ROOT / "outputs/annotated_datasets_top5/train_v273.json.gz",
    },
    "v277": {
        "unseen": HABITAT_DATA / "val_unseen/val_unseen_v277.json.gz",
        "seen":   PIPELINE_ROOT / "outputs/annotated_datasets_top5/seen_v277.json.gz",
        "train":  PIPELINE_ROOT / "outputs/annotated_datasets_top5/train_v277.json.gz",
    },
    "v278": {
        "unseen": HABITAT_DATA / "val_unseen/val_unseen_v278.json.gz",
        "seen":   PIPELINE_ROOT / "outputs/annotated_datasets_top5/seen_v278.json.gz",
        "train":  PIPELINE_ROOT / "outputs/annotated_datasets_top5/train_v278.json.gz",
    },
}

SR_MAP = {
    "v264": 65.14, "v272": 66.20, "v273": 65.15, "v277": 68.46, "v278": 72.82,
}


def episode_count(path: Path) -> int:
    try:
        with gzip.open(path, "rt") as f:
            d = json.load(f)
        return len(d.get("episodes", []))
    except Exception:
        return -1


def main():
    manifest = {}
    print("=" * 70)
    print("Organizing top 5 annotator datasets → outputs/annotated_datasets_top5/")
    print("=" * 70)

    for ver in TOP5_VERSIONS:
        manifest[ver] = {"sr_pct": SR_MAP[ver], "splits": {}}
        for split, src in SOURCES[ver].items():
            dest_name = f"{split[:6].rstrip('_')}_{ver}.json.gz"
            # Normalize: "val_unseen" → "unseen", "val_seen" → "seen", "train" → "train"
            if split == "val_unseen" or split == "unseen":
                dest_name = f"unseen_{ver}.json.gz"
            elif split == "val_seen" or split == "seen":
                dest_name = f"seen_{ver}.json.gz"
            else:
                dest_name = f"train_{ver}.json.gz"
            dest = OUT / dest_name

            status = "missing"
            n_eps = 0
            if src.exists():
                n_eps = episode_count(src)
                if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                    shutil.copy2(str(src), str(dest))
                    status = "copied"
                else:
                    status = "up-to-date"
            else:
                status = "PENDING" if split == "train" else "MISSING"

            manifest[ver]["splits"][dest_name] = {
                "status": status,
                "episodes": n_eps if n_eps > 0 else None,
                "source": str(src),
                "dest": str(dest),
            }
            marker = "✓" if status in ("copied", "up-to-date") else "…" if status == "PENDING" else "✗"
            print(f"  {marker} {dest_name:<28} {status:<12} {n_eps if n_eps>0 else '':>6} eps")

    manifest_path = OUT / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {manifest_path}")
    print("=" * 70)
    print("When all PENDING files are ready (train split after GPU2 free),")
    print("re-run this script to finalize and copy to NAS.")
    print("=" * 70)


if __name__ == "__main__":
    main()
