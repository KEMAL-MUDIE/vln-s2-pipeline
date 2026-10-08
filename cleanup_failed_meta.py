#!/usr/bin/env python3
"""
Remove meta files with empty generated_instruction.text so run_annotator_configured.py
--resume will re-annotate them.

Usage:
  python3 cleanup_failed_meta.py unseen          # clean all 5 unseen_v* meta dirs
  python3 cleanup_failed_meta.py unseen v278     # clean only unseen_v278_meta
  python3 cleanup_failed_meta.py seen            # clean seen_v* meta dirs
  python3 cleanup_failed_meta.py --dry-run unseen
"""
import argparse
import json
import sys
from pathlib import Path

TOP5 = Path(__file__).parent / "outputs/annotated_datasets_top5"
VERSIONS = ["v264", "v272", "v273", "v277", "v278"]
SPLIT_PREFIX = {"unseen": "unseen", "seen": "seen", "train": "train",
                "val_unseen": "unseen", "val_seen": "seen"}


def clean_dir(meta_dir: Path, dry_run: bool) -> tuple:
    if not meta_dir.exists():
        return 0, 0
    total = 0
    removed = 0
    for mf in meta_dir.glob("ep_*.json"):
        total += 1
        try:
            with open(mf) as f:
                m = json.load(f)
            text = m.get("generated_instruction", {}).get("text", "").strip()
            if not text:
                removed += 1
                if not dry_run:
                    mf.unlink()
        except Exception:
            removed += 1
            if not dry_run:
                mf.unlink()
    return total, removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", help="Split to clean: unseen / seen / train")
    parser.add_argument("version", nargs="?", default=None,
                        help="Optional version e.g. v278 (default: all 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be removed without deleting")
    args = parser.parse_args()

    prefix = SPLIT_PREFIX.get(args.split.lower(), args.split)
    versions = [args.version] if args.version else VERSIONS
    tag = "[DRY RUN] " if args.dry_run else ""

    print(f"{tag}Cleaning empty meta files for {args.split} / {versions}")
    grand_total = 0
    grand_removed = 0
    for v in versions:
        meta_dir = TOP5 / f"{prefix}_{v}_meta"
        total, removed = clean_dir(meta_dir, args.dry_run)
        grand_total += total
        grand_removed += removed
        if total:
            print(f"  {prefix}_{v}_meta: {removed}/{total} empty → {'will remove' if args.dry_run else 'removed'}")
        else:
            print(f"  {prefix}_{v}_meta: not found or empty dir")

    print(f"\n{tag}Total: {grand_removed}/{grand_total} empty files {'would be' if args.dry_run else ''} removed")
    if not args.dry_run and grand_removed:
        print(f"Now re-run:  bash run_all_unseen_top5.sh")


if __name__ == "__main__":
    main()
