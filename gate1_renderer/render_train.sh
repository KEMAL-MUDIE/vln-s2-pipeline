#!/usr/bin/env bash
# Gate 1: Train split frame renderer
# Renders key-frame RGB images for all 10819 train episodes
# GPU: device=2 (RTX 3090) — ONLY run when eval chain is idle (eval uses GPU2 too)
#
# Usage:
#   bash gate1_renderer/render_train.sh [N_EPISODES] [START_IDX]
#   bash gate1_renderer/render_train.sh 100 0     # test first 100
#   bash gate1_renderer/render_train.sh 10819 0   # full run
set -euo pipefail
cd "$(dirname "$0")/.."

N_EPISODES="${1:-10819}"
START_IDX="${2:-0}"

INTERNAV=/home/kemal/VLNav/VLNav/workspaces/model/InternNav
HABITAT_DATA=/mnt/nvme0/vln_habitat/habitat_data
OUTPUT_DIR="$(pwd)/outputs/rendered_frames_train"
LOGS="$(pwd)/logs"
mkdir -p "$OUTPUT_DIR" "$LOGS"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOGS/gate1_render_train_${TIMESTAMP}.log"

echo "=== Gate 1: Train Frame Renderer (GPU 2) ==="
echo "  Episodes:  $N_EPISODES (start=$START_IDX)"
echo "  Output:    $OUTPUT_DIR"
echo "  Log:       $LOGFILE"
echo ""
echo "WARNING: This uses GPU2. Ensure no eval container is running."
echo "  Check: docker ps | grep habitat_valunseen"
echo ""

docker rm -f vlnav_gate1_train_renderer 2>/dev/null || true

docker run --rm \
  --name vlnav_gate1_train_renderer \
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

echo '=== Running Gate 1 renderer (train split) ==='
python3 /workspace/s2_pipeline/gate1_renderer/run_renderer.py \
    --gt-path /habitat_datasets/vln/mp3d/r2r/v1/train/train.json.gz \
    --scenes-root /habitat_scenes \
    --output-dir /workspace/s2_pipeline/outputs/rendered_frames_train \
    --n-episodes $N_EPISODES \
    --start $START_IDX
" 2>&1 | tee "$LOGFILE"

RC=${PIPESTATUS[0]}
if [ $RC -eq 0 ]; then
    N_RENDERED=$(find "$OUTPUT_DIR" -name "poses.json" 2>/dev/null | wc -l)
    echo "=== Render complete: $N_RENDERED train episodes rendered ==="
    echo "Next: run batch annotator"
    echo "  python3 run_batch_annotator.py --split train --workers 8 --resume"
else
    echo "=== ERROR: renderer exited with code $RC ==="
fi
exit $RC
