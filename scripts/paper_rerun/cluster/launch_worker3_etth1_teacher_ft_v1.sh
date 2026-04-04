#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker3_etth1_teacher_ft_v1.lock"
flock -n 9 || {
  echo "[worker3_etth1_teacher_ft_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_teacher_ft_v1/worker3}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker3/etth1_teacher_ft_v1" \
  "$LOG_DIR/launcher.out" \
  env \
    GPU="${TIME_SERIES_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_teacher_ft_v1}" \
    LOG_DIR="${INNER_LOG_DIR:-results/logs/time_series_etth1_teacher_ft_v1/worker3/cases}" \
    SEEDS="${SEEDS:-0 1 2}" \
    EPOCHS="${EPOCHS:-90}" \
    NUM_WORKERS="${NUM_WORKERS:-4}" \
    BATCH_SIZE="${BATCH_SIZE:-768}" \
    CASE_SPECS="${CASE_SPECS:-tft_tail10_reg15_l010_lr2e4:0.0002:0.010:0.00001:64:0.00:-0.20:10:26:0.10:0.15:0.15:13:0.05 tft_tail40_reg30_l025_lr5e4:0.0005:0.025:0.00005:128:0.00:-0.90:14:34:0.40:0.30:0.30:13:0.20}" \
    bash scripts/paper_rerun/run_time_series_etth1_teacher_ft_v1.sh

echo "[worker3_etth1_teacher_ft_v1] job finished"
