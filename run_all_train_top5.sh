#!/usr/bin/env bash
# Run train annotation for all top 5 versions sequentially.
# MUST only run after:
#   1. gate1_renderer/render_train.sh has completed (all 10819 frames rendered)
#   2. GPU2 is confirmed free (eval chain finished)
#
# This is triggered automatically by gate1_renderer/watcher_render_train.sh
# when the v288 eval result appears.
#
# Manual usage:
#   bash run_all_train_top5.sh [--resume]
set -euo pipefail
cd "$(dirname "$0")"

LOGS="outputs/logs"
mkdir -p "$LOGS"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Safety check: ensure train frames exist
TRAIN_FRAMES="outputs/rendered_frames_train"
if [ ! -d "$TRAIN_FRAMES" ]; then
    echo "ERROR: $TRAIN_FRAMES does not exist."
    echo "       Run gate1_renderer/render_train.sh first."
    exit 1
fi

N_RENDERED=$(find "$TRAIN_FRAMES" -name "poses.json" 2>/dev/null | wc -l)
if [ "$N_RENDERED" -lt 1000 ]; then
    echo "WARNING: Only $N_RENDERED train episodes rendered (expected 10819)."
    echo "         Continue anyway? [y/N]"
    read -r ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
fi

echo "=== Train frames: $N_RENDERED episodes ==="

run_version() {
    local VER="$1"
    local CFG="configs/annotator_${VER}.yaml"
    local LOG="$LOGS/train_${VER}_${TIMESTAMP}.log"
    echo ""
    echo "=========================================="
    echo "  Annotating train with config $VER"
    echo "  Log: $LOG"
    echo "=========================================="
    python3 run_annotator_configured.py \
        --config "$CFG" \
        --split train \
        --resume \
        2>&1 | tee "$LOG"
    echo "=== Done: train_$VER ==="
}

echo "=== Top 5 train annotation run ==="
echo "  Running: v264, v272, v273, v277, v278"
echo ""

run_version v264
run_version v272
run_version v273
run_version v277
run_version v278

echo ""
echo "=== All train versions complete. Running organize script... ==="
python3 organize_top5_datasets.py

echo ""
echo "=== ALL SPLITS COMPLETE FOR TOP 5 VERSIONS ==="
echo ""
echo "Outputs in: outputs/annotated_datasets_top5/"
echo ""
echo "  unseen_v264.json.gz  ← ChronoNav (SR=65.14%)"
echo "  seen_v264.json.gz    ← VLM auto-annotated"
echo "  train_v264.json.gz   ← VLM auto-annotated"
echo ""
echo "  unseen_v272.json.gz  ← ChronoNav (SR=66.20%)"
echo "  seen_v272.json.gz    ← VLM auto-annotated"
echo "  train_v272.json.gz   ← VLM auto-annotated"
echo ""
echo "  unseen_v273.json.gz  ← ChronoNav (SR=65.15%)"
echo "  seen_v273.json.gz    ← VLM auto-annotated"
echo "  train_v273.json.gz   ← VLM auto-annotated"
echo ""
echo "  unseen_v277.json.gz  ← ChronoNav (SR=68.46%)"
echo "  seen_v277.json.gz    ← VLM auto-annotated"
echo "  train_v277.json.gz   ← VLM auto-annotated"
echo ""
echo "  unseen_v278.json.gz  ← ChronoNav (SR=72.82% BEST)"
echo "  seen_v278.json.gz    ← VLM auto-annotated"
echo "  train_v278.json.gz   ← VLM auto-annotated"
echo ""
echo "Ready to move to NAS. Notify user."
