#!/usr/bin/env bash
# Waits for train render to finish, then runs all 5 version annotations.
# Started alongside render_train.sh. Notifies when everything is NAS-ready.
set -euo pipefail
cd "$(dirname "$0")"

LOG="outputs/logs/watcher_render_to_annotation.log"
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "Watcher started: waiting for train render to complete..."
log "  Monitor: docker logs vlnav_gate1_train_renderer"

# Wait for render container to finish
until ! docker ps --format '{{.Names}}' | grep -q 'vlnav_gate1_train_renderer'; do
    N=$(find outputs/rendered_frames_train -name "poses.json" 2>/dev/null | wc -l)
    log "  Render progress: $N / 10819 episodes"
    sleep 300
done

N_RENDERED=$(find outputs/rendered_frames_train -name "poses.json" 2>/dev/null | wc -l)
log "Render complete: $N_RENDERED episodes rendered."

if [ "$N_RENDERED" -lt 5000 ]; then
    log "ERROR: too few frames ($N_RENDERED). Check render log."
    exit 1
fi

log "Starting train annotation for all 5 versions..."
bash run_all_train_top5.sh 2>&1 | tee -a "$LOG"

log ""
log "=========================================="
log "ALL DONE — ready to move to NAS:"
log ""
log "  outputs/annotated_datasets_top5/"
log "    unseen_v264/272/273/277/278.json.gz  (done)"
log "    seen_v264/272/273/277/278.json.gz    (done)"
log "    train_v264/272/273/277/278.json.gz   (just finished)"
log "=========================================="
