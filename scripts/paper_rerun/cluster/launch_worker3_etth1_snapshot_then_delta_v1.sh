#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker3_etth1_snapshot_then_delta_v1.lock"
flock -n 9 || {
  echo "[worker3_etth1_snapshot_then_delta_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_infinite_loop_v1/worker3}"
mkdir -p "$LOG_DIR"

GPU="${TARGET_GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"

run_logged_job \
  "worker3/etth1_teacher_ft_snapshot_handoff_v1" \
  "$LOG_DIR/snapshot_handoff.log" \
  env \
    GPU="$GPU" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="$SEEDS" \
    BATCH_SIZE="${BATCH_SIZE:-768}" \
    NUM_WORKERS="${NUM_WORKERS:-4}" \
    OUTPUT_ROOT="results/time_series_etth1_teacher_ft_snapshot_handoff_v1/worker3_gpu0" \
    LOG_DIR="results/logs/time_series_etth1_teacher_ft_snapshot_handoff_v1/worker3/cases" \
    CASE_SPECS="${SNAPSHOT_CASE_SPECS:-tft_h26_t14_l012_lr2e4:0.00020:0.012:64:0.00:-0.20:12:26:26:0.14:0.18:0.15:13:0.05}" \
    bash scripts/paper_rerun/run_time_series_etth1_teacher_ft_snapshot_handoff_v1.sh

run_logged_job \
  "worker3/etth1_delta_fusion_v1" \
  "$LOG_DIR/delta_fusion.log" \
  env \
    GPU="$GPU" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="$SEEDS" \
    BATCH_SIZE="${BATCH_SIZE:-768}" \
    NUM_WORKERS="${NUM_WORKERS:-4}" \
    OUTPUT_ROOT="results/time_series_etth1_delta_fusion_v1/worker3_gpu0" \
    LOG_DIR="results/logs/time_series_etth1_delta_fusion_v1/worker3/cases" \
    CASE_SPECS="${DELTA_CASE_SPECS:-fusion_tail35_g12_lr2e4:0.00020:0.010:0.35:1.20}" \
    bash scripts/paper_rerun/run_time_series_etth1_delta_fusion_v1.sh

echo "[worker3_etth1_snapshot_then_delta_v1] all jobs finished"
