#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker2_etth1_dense_regime_horizon_seeded_v3.lock"
flock -n 9 || {
  echo "[worker2_etth1_dense_regime_horizon_seeded_v3] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_dense_regime_horizon_seeded_v3/worker2}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker2/etth1_dense_regime_horizon_seeded_v3" \
  "$LOG_DIR/time_series_gpu${TIME_SERIES_GPU:-0}.log" \
  env \
    GPU="${TIME_SERIES_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    BATCH_SIZE="${BATCH_SIZE_3090TI:-64}" \
    NUM_WORKERS="${NUM_WORKERS_3090TI:-4}" \
    OUTPUT_ROOT="results/time_series_etth1_dense_regime_horizon_seeded_v3/worker2" \
    LOG_DIR="$LOG_DIR/cases" \
    CASE_SPECS="${CASE_SPECS:-}" \
    bash scripts/paper_rerun/run_time_series_etth1_dense_regime_horizon_seeded_v3.sh

echo "[worker2_etth1_dense_regime_horizon_seeded_v3] job finished"
