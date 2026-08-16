#!/usr/bin/env bash
# Gate 1: Batch frame renderer using Habitat-Sim
# Renders key-frame RGB images along reference_paths (start + turns + goal)
# Output: outputs/rendered_frames/episode_{id:06d}/ + poses.json
#
# GPU:   2 (RTX 3090, 22GB free — elastic_curie coexists at 2.3GB)
# Image: vlnav/habitat-eval:rebuilt
# Usage: bash render_batch.sh [N_EPISODES] [START_IDX] [dry]
#        bash render_batch.sh 50        → validate: render first 50 episodes
#        bash render_batch.sh 1839 0    → full run: all 1839 episodes
set -euo pipefail
cd "$(dirname "$0")/.."

N_EPISODES="${1:-50}"
START_IDX="${2:-0}"
DRY_RUN="${3:-}"   # pass "dry" as 3rd arg for dry run

INTERNAV=/home/kemal/VLNav/VLNav/workspaces/model/InternNav
HABITAT_DATA=/mnt/nvme0/vln_habitat/habitat_data
OUTPUT_DIR="$(pwd)/outputs/rendered_frames"
LOGS="$(pwd)/logs"
mkdir -p "$OUTPUT_DIR" "$LOGS"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOGS/gate1_render_${TIMESTAMP}.log"

echo "=== Gate 1: Habitat Frame Renderer (GPU 2) ==="
echo "  Episodes:  $N_EPISODES (start=$START_IDX)"
echo "  Output:    $OUTPUT_DIR"
echo "  Log:       $LOGFILE"
echo ""

docker rm -f vlnav_gate1_renderer 2>/dev/null || true

DRY_ARG=""
if [ "$DRY_RUN" = "dry" ]; then
    DRY_ARG="--dry-run"
fi

docker run --rm \
  --name vlnav_gate1_renderer \
  --gpus '"device=2"' \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HABITAT_SIM_EGL_DEVICE_ID=0 \
  -e PYTHONPATH="/workspace/InternNav/third_party/habitat-sim/src_python:/workspace/InternNav/third_party/habitat-sim/build/cp311-cp311-linux_x86_64/RelWithDebInfo/lib:/workspace/InternNav/third_party/habitat-sim/build/cp311-cp311-linux_x86_64/deps/magnum-bindings/src/python" \
  --network host \
  --shm-size 8g \
  -v "$INTERNAV":/workspace/InternNav:ro \
  -v "$HABITAT_DATA/scene_datasets":/habitat_scenes:ro \
  -v "$HABITAT_DATA/datasets":/habitat_datasets:ro \
  -v "$(pwd)":/workspace/s2_pipeline:rw \
  vlnav/habitat-eval:rebuilt \
  bash -c "
set -e
echo '=== Verifying habitat-sim ==='
python3 -c 'import habitat_sim; print(\"habitat-sim\", habitat_sim.__version__)'

echo '=== Running Gate 1 renderer ==='
python3 /workspace/s2_pipeline/gate1_renderer/run_renderer.py \
    --gt-path /habitat_datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen_patched.json.gz \
    --scenes-root /habitat_scenes \
    --output-dir /workspace/s2_pipeline/outputs/rendered_frames \
    --n-episodes $N_EPISODES \
    --start $START_IDX \
    $DRY_ARG
" 2>&1 | tee "$LOGFILE"

RC=${PIPESTATUS[0]}
echo ""
if [ $RC -eq 0 ]; then
    N_RENDERED=$(find "$OUTPUT_DIR" -name "poses.json" 2>/dev/null | wc -l)
    echo "=== Render complete: $N_RENDERED episodes rendered ==="
    echo "Next: run Gate 3 (landmark detection)"
    echo "  python3 gate3_landmarks/run_landmark_batch.py --frames-dir $OUTPUT_DIR"
else
    echo "=== ERROR: renderer exited with code $RC ==="
    echo "Check log: $LOGFILE"
fi
exit $RC
