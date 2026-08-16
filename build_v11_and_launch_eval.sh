#!/usr/bin/env bash
# Build v11 dataset and launch Habitat eval
#
# Steps:
#   1. Wait for Gate 3 v2 to finish (if still running)
#   2. Run Gate 4 v11 on all 1839 episodes
#   3. Build val_unseen_v11.json.gz
#   4. Launch Habitat eval on GPU 1
#
# Usage: bash build_v11_and_launch_eval.sh
set -euo pipefail
cd "$(dirname "$0")"

VENV=/home/kemal/VLNav/vlnav_env/bin/activate
EVAL_SCRIPT=/home/kemal/VLNav/habitat_eval/scripts/run_eval_gate4_v11.sh
G3PF_DIR=outputs/gate3_perframe
TOTAL_EPS=1839

echo "======================================================================"
echo "v11 Build + Eval Pipeline"
echo "======================================================================"

# Step 1: Wait for Gate 3 v2 if running
G3_DONE=$(ls "$G3PF_DIR"/*.json 2>/dev/null | wc -l)
echo "[1/3] Gate 3 v2: $G3_DONE/$TOTAL_EPS episodes done"

if [ "$G3_DONE" -lt "$TOTAL_EPS" ]; then
    echo "  Waiting for Gate 3 v2 to complete (PID: $(pgrep -f run_gate3_perframe_v2 || echo 'not running'))..."
    while true; do
        G3_DONE=$(ls "$G3PF_DIR"/*.json 2>/dev/null | wc -l)
        if [ "$G3_DONE" -ge "$TOTAL_EPS" ]; then
            echo "  Gate 3 v2 DONE: $G3_DONE episodes"
            break
        fi
        echo "  Gate 3 v2: $G3_DONE/$TOTAL_EPS..."
        sleep 30
    done
fi

# Step 2: Run Gate 4 v11
echo ""
echo "[2/3] Running Gate 4 v11 instruction generator..."
source "$VENV"
python3 run_gate4_visual_v11.py --concurrency 12

V11_PATH=outputs/datasets/val_unseen_generated_gemma_visual_v11.json.gz
N_EPS=$(python3 -c "
import gzip, json
with gzip.open('$V11_PATH', 'rt') as f: d = json.load(f)
print(len(d['episodes']))
")
echo "v11 dataset: $N_EPS episodes"

# Step 3: Launch Habitat eval
echo ""
echo "[3/3] Launching Habitat eval on GPU 1..."
bash "$EVAL_SCRIPT"
