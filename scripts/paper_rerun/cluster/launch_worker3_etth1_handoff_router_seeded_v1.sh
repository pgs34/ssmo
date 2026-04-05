#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker3_etth1_handoff_router_seeded_v1.lock"
flock -n 9 || {
  echo "[worker3_etth1_handoff_router_seeded_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_handoff_router_seeded_v1/worker3}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker3/etth1_handoff_router_seeded_v1" \
  "$LOG_DIR/launcher.out" \
  env \
    GPU="${TIME_SERIES_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_handoff_router_seeded_v1/worker3_gpu0}" \
    LOG_DIR="${INNER_LOG_DIR:-results/logs/time_series_etth1_handoff_router_seeded_v1/worker3/cases}" \
    SEEDS="${SEEDS:-0 1 2}" \
    EPOCHS="${EPOCHS:-90}" \
    NUM_WORKERS="${NUM_WORKERS:-4}" \
    BATCH_SIZE="${BATCH_SIZE:-768}" \
    CASE_SPECS="${CASE_SPECS:-hr_long_q85_b08_h34:0.00030:0.015:0.00002:96:0.00:-0.45:12:30:0.20:0.25:0.15:9:0.08:34:0.80:0.08:1.15:0.10}" \
    bash scripts/paper_rerun/run_time_series_etth1_handoff_router_seeded_v1.sh

echo "[worker3_etth1_handoff_router_seeded_v1] job finished"
