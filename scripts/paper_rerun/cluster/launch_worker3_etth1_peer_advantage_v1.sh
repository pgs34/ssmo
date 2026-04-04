#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_peer_advantage_v1/worker3}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker3/etth1_peer_advantage_v1" \
  "$LOG_DIR/time_series_gpu${TIME_SERIES_GPU:-0}.log" \
  env \
    GPU="${TIME_SERIES_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_ROOT="results/time_series_etth1_peer_advantage_v1/worker3" \
    bash scripts/paper_rerun/run_time_series_etth1_peer_advantage_v1.sh

echo "[worker3_etth1_peer_advantage_v1] job finished"
