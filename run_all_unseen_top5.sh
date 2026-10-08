#!/usr/bin/env bash
# Run val_unseen VLM annotation for all top 5 versions IN PARALLEL.
# Replaces ChronoNav unseen_v*.json.gz with VLM-annotated versions.
# ChronoNav originals backed up as unseen_v*_chrono.json.gz.
#
# All 5 versions use GT val_unseen as episode source + VLM primary instruction.
# Annotation calls remote vLLM at 10.77.32.231:8000 — no local GPU needed.
# Workers per version: 6  (5 × 6 = 30 concurrent vLLM requests)
#
# Manual usage:
#   bash run_all_unseen_top5.sh
set -euo pipefail
cd "$(dirname "$0")"

LOGS="outputs/logs"
mkdir -p "$LOGS"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

FRAMES="outputs/rendered_frames"
N_RENDERED=$(find "$FRAMES" -name "poses.json" 2>/dev/null | wc -l)
echo "=== Val_unseen rendered frames: $N_RENDERED / 1839 ==="
if [ "$N_RENDERED" -lt 1800 ]; then
    echo "ERROR: too few rendered frames. Expected 1839."
    exit 1
fi

run_version_bg() {
    local VER="$1"
    local LOG="$LOGS/unseen_${VER}_${TIMESTAMP}.log"
    echo "  Launching $VER → $LOG"
    python3 run_annotator_configured.py \
        --config "configs/annotator_${VER}.yaml" \
        --split val_unseen \
        --workers 6 \
        --resume \
        > "$LOG" 2>&1 &
    echo $!
}

echo ""
echo "=== Launching all 5 versions in PARALLEL ==="
echo "  (5 × 6 workers = 30 concurrent vLLM requests)"
echo "  VLM instruction is PRIMARY (GT episode structure, VLM-generated text)"
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
echo "    n=\$(ls outputs/annotated_datasets_top5/unseen_\${v}_meta/ 2>/dev/null | wc -l)"
echo "    echo \"unseen_\$v: \$n/1839\"; done'"
echo ""
echo "Or watch logs:"
echo "  tail -f outputs/logs/unseen_v278_${TIMESTAMP}.log"

# Wait for all
wait $PID_264 && echo "v264 done" || echo "v264 FAILED"
wait $PID_272 && echo "v272 done" || echo "v272 FAILED"
wait $PID_273 && echo "v273 done" || echo "v273 FAILED"
wait $PID_277 && echo "v277 done" || echo "v277 FAILED"
wait $PID_278 && echo "v278 done" || echo "v278 FAILED"

echo ""
echo "=== All unseen versions complete. Running organize script... ==="
python3 organize_top5_datasets.py

echo ""
echo "=== VAL_UNSEEN VLM ANNOTATION COMPLETE ==="
echo ""
echo "  unseen_v264.json.gz  ← VLM-annotated (was ChronoNav SR=65.14%)"
echo "  unseen_v272.json.gz  ← VLM-annotated (was ChronoNav SR=66.20%)"
echo "  unseen_v273.json.gz  ← VLM-annotated (was ChronoNav SR=65.15%)"
echo "  unseen_v277.json.gz  ← VLM-annotated (was ChronoNav SR=68.46%)"
echo "  unseen_v278.json.gz  ← VLM-annotated (was ChronoNav SR=72.82% BEST)"
echo ""
echo "  ChronoNav backups:   unseen_v*_chrono.json.gz"
echo ""
echo "Run quality analysis:"
echo "  python3 analyze_metadata_quality.py"
