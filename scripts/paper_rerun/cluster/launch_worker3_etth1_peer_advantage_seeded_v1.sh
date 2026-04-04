#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker3_etth1_peer_advantage_seeded_v1.lock"
flock -n 9 || {
  echo "[worker3_etth1_peer_advantage_seeded_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_peer_advantage_seeded_v1/worker3}"
mkdir -p "$LOG_DIR"

CASE_SPECS="${CASE_SPECS:-advq70_tail30_rg0_min5e3_f8:0.0010:0.15:0.0005:96:0.00:-3.0:2:8:8:0.30:0.00:0.70:0.005:3:9:18:44:0.35 advq80_tail55_rg0_min5e3:0.0008:0.20:0.0004:96:0.00:-2.5:2:12:16:0.55:0.00:0.80:0.005:5:13:24:48:0.30}"

run_logged_job \
  "worker3/etth1_peer_advantage_seeded_v1" \
  "$LOG_DIR/time_series_gpu${TIME_SERIES_GPU:-0}.log" \
  env \
    GPU="${TIME_SERIES_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    BATCH_SIZE="${BATCH_SIZE_3090TI:-160}" \
    NUM_WORKERS="${NUM_WORKERS_3090TI:-4}" \
    OUTPUT_ROOT="results/time_series_etth1_peer_advantage_seeded_v1/worker3" \
    LOG_DIR="$LOG_DIR/cases" \
    CASE_SPECS="$CASE_SPECS" \
    bash scripts/paper_rerun/run_time_series_etth1_peer_advantage_seeded_v1.sh

echo "[worker3_etth1_peer_advantage_seeded_v1] job finished"
