#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker3_etth1_teacher_ft_v2.lock"
flock -n 9 || {
  echo "[worker3_etth1_teacher_ft_v2] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_teacher_ft_v2/worker3}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker3/etth1_teacher_ft_v2" \
  "$LOG_DIR/launcher.out" \
  env \
    GPU="${TIME_SERIES_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    BATCH_SIZE="${BATCH_SIZE:-768}" \
    NUM_WORKERS="${NUM_WORKERS:-4}" \
    OUTPUT_ROOT="results/time_series_etth1_teacher_ft_v2/worker3_gpu0" \
    LOG_DIR="$LOG_DIR/cases" \
    CASE_SPECS="${CASE_SPECS:-trs13_res04_lr2e4:0.0002:0.010:0.00001:64:0.00:-0.25:10:24:0.10:0.20:0.12:13:0.05:1.3:0.4:4:20:40:0.10}" \
    bash scripts/paper_rerun/run_time_series_etth1_teacher_ft_v2.sh

echo "[worker3_etth1_teacher_ft_v2] job finished"
