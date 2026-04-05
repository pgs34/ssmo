#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker3_burgers_fno_polish_fair_v2.lock"
flock -n 9 || {
  echo "[worker3_burgers_fno_polish_fair_v2] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/operator_burgers_fno_polish_fair_v2/worker3}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker3/operator_burgers_fno_polish_fair_v2" \
  "$LOG_DIR/launcher.out" \
  env \
    GPU="${TARGET_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-results/operator_burgers_fno_polish_fair_v2/worker3_gpu0}" \
    LOG_DIR="${INNER_LOG_DIR:-results/logs/operator_burgers_fno_polish_fair_v2/worker3/cases}" \
    SEEDS="${SEEDS:-0 1 2}" \
    EPOCHS="${EPOCHS:-180}" \
    BATCH_SIZE="${BATCH_SIZE:-16}" \
    NUM_WORKERS="${NUM_WORKERS:-2}" \
    bash scripts/paper_rerun/run_operator_burgers_fno_polish_fair_v2.sh

echo "[worker3_burgers_fno_polish_fair_v2] job finished"
