#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$(paper_results_root)/logs/worker2}"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${OPERATOR_GPU:-0}" \
METHODS="${OPERATOR_METHODS:-independent dml ssml}" \
DATASETS="${OPERATOR_DATASETS:-darcy}" \
MODEL_PAIRS="${OPERATOR_MODEL_PAIRS:-fno:deeponet}" \
BATCH_SIZE="${OPERATOR_BATCH_SIZE:-8}" \
NUM_WORKERS="${OPERATOR_NUM_WORKERS:-2}" \
OUTPUT_DIR="${OPERATOR_OUTPUT_DIR:-$(paper_results_root)/operator}" \
run_logged_job \
  "worker2/operator" \
  "$LOG_DIR/operator_gpu${OPERATOR_GPU:-0}.log" \
  bash scripts/paper_rerun/run_core_operator.sh
echo "[worker2] results_root=$(paper_results_root)"
echo "[worker2] job finished"
