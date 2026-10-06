#!/usr/bin/env bash
# Run val_seen annotation for all top 5 versions sequentially.
# v272 val_seen is already done (seen_v272 will be copied from complete_v272).
# This runs v264, v273, v277, v278 using run_annotator_configured.py.
#
# Prerequisites:
#   - vLLM server running at http://10.77.32.231:8000/v1
#   - rendered_frames_val_seen/ exists with all 778 episode dirs
#
# Usage:
#   bash run_all_seen_top5.sh [--resume]
set -euo pipefail
cd "$(dirname "$0")"

RESUME="${1:-}"
LOGS="outputs/logs"
mkdir -p "$LOGS"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_version() {
    local VER="$1"
    local CFG="configs/annotator_${VER}.yaml"
    local LOG="$LOGS/seen_${VER}_${TIMESTAMP}.log"
    echo ""
    echo "=========================================="
    echo "  Annotating val_seen with config $VER"
    echo "  Log: $LOG"
    echo "=========================================="
    python3 run_annotator_configured.py \
        --config "$CFG" \
        --split val_seen \
        --resume \
        2>&1 | tee "$LOG"
    echo "=== Done: seen_$VER ==="
}

echo "=== Top 5 val_seen annotation run ==="
echo "  v272: already done (complete_metadata_seen_v272)"
echo "  Running: v264, v273, v277, v278"
echo ""

run_version v264
run_version v273
run_version v277
run_version v278

echo ""
echo "=== All val_seen versions complete. Running organize script... ==="
python3 organize_top5_datasets.py

echo ""
echo "=== DONE. Check outputs/annotated_datasets_top5/ ==="
echo "  seen_v264.json.gz"
echo "  seen_v272.json.gz  (from complete_v272)"
echo "  seen_v273.json.gz"
echo "  seen_v277.json.gz"
echo "  seen_v278.json.gz"
echo ""
echo "Train splits still PENDING — will auto-run after eval chain completes."
