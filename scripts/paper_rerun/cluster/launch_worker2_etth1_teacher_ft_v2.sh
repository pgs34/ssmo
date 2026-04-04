#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker2_etth1_teacher_ft_v2.lock"
flock -n 9 || {
  echo "[worker2_etth1_teacher_ft_v2] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_teacher_ft_v2/worker2}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker2/etth1_teacher_ft_v2" \
  "$LOG_DIR/launcher.out" \
  env \
    GPU="${TIME_SERIES_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    BATCH_SIZE="${BATCH_SIZE:-768}" \
    NUM_WORKERS="${NUM_WORKERS:-4}" \
    OUTPUT_ROOT="results/time_series_etth1_teacher_ft_v2/worker2_gpu0" \
    LOG_DIR="$LOG_DIR/cases" \
    CASE_SPECS="${CASE_SPECS:-trs18_res01_lr3e4:0.0003:0.012:0.00002:96:0.00:-0.45:12:28:0.15:0.20:0.18:13:0.04:1.8:0.1:5:22:42:0.10}" \
    bash scripts/paper_rerun/run_time_series_etth1_teacher_ft_v2.sh

echo "[worker2_etth1_teacher_ft_v2] job finished"
