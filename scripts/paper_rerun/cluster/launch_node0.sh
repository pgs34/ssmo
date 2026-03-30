#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$(paper_results_root)/logs/node0}"
mkdir -p "$LOG_DIR"

run_classification() {
  CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU:-0}" \
  METHODS="${CLASSIFICATION_METHODS:-independent dml ssml}" \
  SEEDS="${CLASSIFICATION_SEEDS:-0 1 2}" \
  OUTPUT_DIR="${CLASSIFICATION_OUTPUT_DIR:-$(paper_results_root)/classification}" \
  run_logged_job \
    "node0/classification" \
    "$LOG_DIR/classification_gpu${CLASSIFICATION_GPU:-0}.log" \
    bash scripts/paper_rerun/run_core_classification.sh
}

run_classification &
PID_CLASSIFICATION=$!

echo "[node0] started classification pid=$PID_CLASSIFICATION"
echo "[node0] results_root=$(paper_results_root)"

wait "$PID_CLASSIFICATION"
echo "[node0] all jobs finished"
