#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_etth1_independent_rerun_20260405_v1_gpu1.lock"
flock -n 9 || {
  echo "[node0_etth1_independent_rerun_20260405_v1_gpu1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_independent_rerun_20260405_v1}"
mkdir -p "$LOG_DIR"

GPU="${TARGET_GPU:-1}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"

run_logged_job \
  "node0/etth1_independent_rerun_20260405_v1_gpu${GPU}" \
  "$LOG_DIR/launcher_gpu${GPU}.out" \
  env \
    GPU="$GPU" \
    SEEDS="$SEEDS" \
    EPOCHS="${EPOCHS:-80}" \
    BATCH_SIZE="${BATCH_SIZE_4090:-64}" \
    NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
    OUTPUT_ROOT="results/time_series_etth1_independent_rerun_20260405_v1" \
    LOG_DIR="$LOG_DIR" \
    bash scripts/paper_rerun/run_time_series_etth1_independent_rerun_20260405_v1.sh

echo "[node0_etth1_independent_rerun_20260405_v1_gpu1] finished"
