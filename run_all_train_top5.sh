#!/usr/bin/env bash
# Run train annotation for all top 5 versions IN PARALLEL.
# All 5 versions hit the remote vLLM server simultaneously —
# reduces annotation from ~7.5 hrs (sequential) to ~1.5 hrs.
#
# Annotation uses remote vLLM at 10.77.32.231:8000 — no local GPU needed.
# Workers per version: 6 (5 versions × 6 = 30 concurrent vLLM requests).
#
# Manual usage:
#   bash run_all_train_top5.sh
set -euo pipefail
cd "$(dirname "$0")"

LOGS="outputs/logs"
mkdir -p "$LOGS"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

TRAIN_FRAMES="outputs/rendered_frames_train"
if [ ! -d "$TRAIN_FRAMES" ]; then
    echo "ERROR: $TRAIN_FRAMES does not exist. Run render_train.sh first."
    exit 1
fi

N_RENDERED=$(find "$TRAIN_FRAMES" -name "poses.json" 2>/dev/null | wc -l)
echo "=== Train frames: $N_RENDERED / 10819 episodes ==="

run_version_bg() {
    local VER="$1"
    local LOG="$LOGS/train_${VER}_${TIMESTAMP}.log"
    echo "  Launching $VER → $LOG"
    python3 run_annotator_configured.py \
        --config "configs/annotator_${VER}.yaml" \
        --split train \
        --workers 6 \
        --resume \
        > "$LOG" 2>&1 &
    echo $!
}

echo ""
echo "=== Launching all 5 versions in PARALLEL ==="
echo "  (5 × 6 workers = 30 concurrent vLLM requests)"
echo ""

PID_264=$(run_version_bg v264)
PID_272=$(run_version_bg v272)
PID_273=$(run_version_bg v273)
PID_277=$(run_version_bg v277)
PID_278=$(run_version_bg v278)

echo "  PIDs: v264=$PID_264  v272=$PID_272  v273=$PID_273  v277=$PID_277  v278=$PID_278"
echo ""
echo "Monitor progress:"
echo "  watch -n30 'for v in v264 v272 v273 v277 v278; do"
echo "    n=\$(ls outputs/annotated_datasets_top5/train_\${v}_meta/ 2>/dev/null | wc -l)"
echo "    echo \"train_\$v: \$n/10819\"; done'"

# Wait for all
wait $PID_264 && echo "v264 done" || echo "v264 FAILED"
wait $PID_272 && echo "v272 done" || echo "v272 FAILED"
wait $PID_273 && echo "v273 done" || echo "v273 FAILED"
wait $PID_277 && echo "v277 done" || echo "v277 FAILED"
wait $PID_278 && echo "v278 done" || echo "v278 FAILED"

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
