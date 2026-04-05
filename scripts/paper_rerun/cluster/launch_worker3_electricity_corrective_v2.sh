#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker3_electricity_corrective_v2.lock"
flock -n 9 || {
  echo "[worker3_electricity_corrective_v2] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_electricity_corrective_v2/worker3}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker3/electricity_corrective_v2" \
  "$LOG_DIR/launcher.out" \
  env \
    GPU="${TIME_SERIES_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    BATCH_SIZE="${BATCH_SIZE:-64}" \
    NUM_WORKERS="${NUM_WORKERS:-4}" \
    OUTPUT_ROOT="results/time_series_electricity_corrective_v2/worker3_gpu0" \
    LOG_DIR="$LOG_DIR/cases" \
    CASE_SPECS="${CASE_SPECS:-late_b08_q85_m25_sp15e4:0.10:0.0015:32:0.00:8:8:12:16:30:0.00:0.85:0.0025:3:0.08:6:12:0.0001}" \
    bash scripts/paper_rerun/run_time_series_electricity_corrective_v2.sh

echo "[worker3_electricity_corrective_v2] job finished"
