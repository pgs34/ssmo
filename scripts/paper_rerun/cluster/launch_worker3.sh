#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$(paper_results_root)/logs/worker3}"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
METHODS="${TIME_SERIES_METHODS:-independent dml ssml}" \
DATASETS="${TIME_SERIES_DATASETS:-etth1 electricity weather}" \
BATCH_SIZE="${TIME_SERIES_BATCH_SIZE:-32}" \
NUM_WORKERS="${TIME_SERIES_NUM_WORKERS:-2}" \
OUTPUT_DIR="${TIME_SERIES_OUTPUT_DIR:-$(paper_results_root)/time_series}" \
run_logged_job \
  "worker3/time_series" \
  "$LOG_DIR/time_series_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_core_time_series.sh
echo "[worker3] results_root=$(paper_results_root)"
echo "[worker3] job finished"
