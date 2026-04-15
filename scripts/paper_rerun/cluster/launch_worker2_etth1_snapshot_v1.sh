#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker2_etth1_snapshot_v1.lock"
flock -n 9 || {
  echo "[worker2_etth1_snapshot_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_infinite_loop_v1/worker2}"
mkdir -p "$LOG_DIR"

GPU="${TARGET_GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"

run_logged_job \
  "worker2/etth1_teacher_ft_snapshot_handoff_v1" \
  "$LOG_DIR/snapshot_handoff.log" \
  env \
    GPU="$GPU" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="$SEEDS" \
    BATCH_SIZE="${BATCH_SIZE:-768}" \
    NUM_WORKERS="${NUM_WORKERS:-4}" \
    OUTPUT_ROOT="results/time_series_etth1_teacher_ft_snapshot_handoff_v1/worker2_gpu0" \
    LOG_DIR="results/logs/time_series_etth1_teacher_ft_snapshot_handoff_v1/worker2/cases" \
    CASE_SPECS="${SNAPSHOT_CASE_SPECS:-tft_h22_t12_l010_lr2e4:0.00020:0.010:64:0.00:-0.20:10:22:22:0.12:0.18:0.15:13:0.05}" \
    bash scripts/paper_rerun/run_time_series_etth1_teacher_ft_snapshot_handoff_v1.sh

echo "[worker2_etth1_snapshot_v1] finished"
