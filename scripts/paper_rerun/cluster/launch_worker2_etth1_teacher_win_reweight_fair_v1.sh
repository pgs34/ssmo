#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker2_etth1_teacher_win_reweight_fair_v1.lock"
flock -n 9 || {
  echo "[worker2_etth1_teacher_win_reweight_fair_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_teacher_win_reweight_fair_v1/worker2}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker2/etth1_teacher_win_reweight_fair_v1" \
  "$LOG_DIR/launcher.out" \
  env \
    GPU="${TIME_SERIES_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_teacher_win_reweight_fair_v1/worker2_gpu0}" \
    LOG_DIR="${INNER_LOG_DIR:-results/logs/time_series_etth1_teacher_win_reweight_fair_v1/worker2/cases}" \
    SEEDS="${SEEDS:-0 1 2}" \
    EPOCHS="${EPOCHS:-80}" \
    NUM_WORKERS="${NUM_WORKERS:-4}" \
    BATCH_SIZE="${BATCH_SIZE:-768}" \
    bash scripts/paper_rerun/run_time_series_etth1_teacher_win_reweight_fair_v1.sh

echo "[worker2_etth1_teacher_win_reweight_fair_v1] job finished"
