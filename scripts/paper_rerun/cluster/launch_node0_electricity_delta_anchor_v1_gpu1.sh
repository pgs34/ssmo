#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_electricity_delta_anchor_v1_gpu1.lock"
flock -n 9 || {
  echo "[node0_electricity_delta_anchor_v1_gpu1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_electricity_delta_anchor_v1/node0_gpu1}"
mkdir -p "$LOG_DIR"

GPU="${TARGET_GPU:-1}"
SEEDS="${SEEDS:-0 1 2}"

run_logged_job \
  "node0/electricity_delta_anchor_v1_gpu${GPU}" \
  "$LOG_DIR/launcher.out" \
  env \
    GPU="$GPU" \
    SEEDS="$SEEDS" \
    EPOCHS="${EPOCHS:-60}" \
    BATCH_SIZE="${BATCH_SIZE_4090:-64}" \
    NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
    OUTPUT_ROOT="results/time_series_electricity_delta_anchor_v1/node0_gpu1" \
    LOG_DIR="results/logs/time_series_electricity_delta_anchor_v1/node0_gpu1/cases" \
    CASE_SPECS="${CASE_SPECS:-dfa_g10_t20_w20:1.00:20:0.20:0.50:0.20}" \
    bash scripts/paper_rerun/run_time_series_electricity_delta_anchor_v1.sh

echo "[node0_electricity_delta_anchor_v1_gpu1] finished"
