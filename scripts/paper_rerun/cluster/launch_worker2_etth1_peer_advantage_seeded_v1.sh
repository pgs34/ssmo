#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker2_etth1_peer_advantage_seeded_v1.lock"
flock -n 9 || {
  echo "[worker2_etth1_peer_advantage_seeded_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_peer_advantage_seeded_v1/worker2}"
mkdir -p "$LOG_DIR"

CASE_SPECS="${CASE_SPECS:-advq70_tail35_rg0_min0_f8:0.0008:0.18:0.0004:128:0.00:-2.5:2:8:8:0.35:0.00:0.70:0.000:3:9:18:48:0.30 advq75_tail30_rg0_min0_f8:0.0008:0.18:0.0004:128:0.00:-2.5:2:8:8:0.30:0.00:0.75:0.000:3:9:18:48:0.30}"

run_logged_job \
  "worker2/etth1_peer_advantage_seeded_v1" \
  "$LOG_DIR/time_series_gpu${TIME_SERIES_GPU:-0}.log" \
  env \
    GPU="${TIME_SERIES_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    BATCH_SIZE="${BATCH_SIZE_3090TI:-160}" \
    NUM_WORKERS="${NUM_WORKERS_3090TI:-4}" \
    OUTPUT_ROOT="results/time_series_etth1_peer_advantage_seeded_v1/worker2" \
    LOG_DIR="$LOG_DIR/cases" \
    CASE_SPECS="$CASE_SPECS" \
    bash scripts/paper_rerun/run_time_series_etth1_peer_advantage_seeded_v1.sh

echo "[worker2_etth1_peer_advantage_seeded_v1] job finished"
