#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_etth1_snapshot_v1_gpu1.lock"
flock -n 9 || {
  echo "[node0_etth1_snapshot_v1_gpu1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_infinite_loop_v1/node0_gpu1}"
mkdir -p "$LOG_DIR"

GPU="${TARGET_GPU:-1}"
SEEDS="${SEEDS:-0 1 2}"

run_logged_job \
  "node0/etth1_teacher_ft_snapshot_handoff_v1_gpu${GPU}" \
  "$LOG_DIR/snapshot_handoff.log" \
  env \
    GPU="$GPU" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="$SEEDS" \
    BATCH_SIZE="${BATCH_SIZE_4090:-768}" \
    NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
    OUTPUT_ROOT="results/time_series_etth1_teacher_ft_snapshot_handoff_v1/node0_gpu1" \
    LOG_DIR="results/logs/time_series_etth1_teacher_ft_snapshot_handoff_v1/node0_gpu1/cases" \
    CASE_SPECS="${SNAPSHOT_CASE_SPECS:-tft_h18_t10_l008_lr15e4:0.00015:0.008:64:0.00:-0.20:8:18:18:0.10:0.15:0.15:13:0.05}" \
    bash scripts/paper_rerun/run_time_series_etth1_teacher_ft_snapshot_handoff_v1.sh

echo "[node0_etth1_snapshot_v1_gpu1] finished"
