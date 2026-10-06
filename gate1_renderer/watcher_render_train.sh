#!/usr/bin/env bash
# Waits for the eval chain to complete (v288 result.json), then renders train frames
RESULT_288="/home/kemal/VLNav/habitat_eval/logs_v288/habitat/cvml07_valUNseen_v288/result.json"
LOG="/home/kemal/VLNav/s2_pipeline_new/logs/render_train_watcher.log"
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "Watcher: waiting for v288 to complete (end of eval chain)..."
until [ -f "$RESULT_288" ]; do sleep 120; done
log "v288 done — eval chain complete. Checking GPU2 is free..."
until ! docker ps --format '{{.Names}}' | grep -q 'vlnav_habitat_valunseen'; do
    log "Waiting for any eval container to finish..."; sleep 60
done
log "GPU2 free. Starting train frame render (10819 episodes)..."
cd /home/kemal/VLNav/s2_pipeline_new
bash gate1_renderer/render_train.sh 10819 0 2>&1 | tee -a "$LOG"
log "Train render complete. Starting train annotation (v272 base + all top 5 versions)..."
python3 run_batch_annotator.py --split train --workers 8 --resume 2>&1 | tee -a "$LOG"
log "v272 train annotation complete. Running all top 5 version configs..."
bash run_all_train_top5.sh 2>&1 | tee -a "$LOG"
log "ALL TOP 5 TRAIN ANNOTATIONS COMPLETE. Ready for NAS transfer."
