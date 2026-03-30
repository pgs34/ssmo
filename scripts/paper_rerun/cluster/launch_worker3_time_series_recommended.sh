#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/time_series_ssml_topk_v1/logs/worker3}"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
METHODS="${TIME_SERIES_METHODS:-independent dml ssml}" \
DATASETS="${TIME_SERIES_DATASETS:-etth1 electricity weather}" \
BATCH_SIZE="${TIME_SERIES_BATCH_SIZE:-32}" \
NUM_WORKERS="${TIME_SERIES_NUM_WORKERS:-2}" \
OUTPUT_DIR="${TIME_SERIES_OUTPUT_DIR:-results/time_series_ssml_topk_v1}" \
LAMBDA_IMITATION="${TIME_SERIES_LAMBDA:-0.3}" \
MARGIN="${TIME_SERIES_MARGIN:-0.05}" \
WARMUP_EPOCHS="${TIME_SERIES_WARMUP:-5}" \
SSML_TOPK_RATIO="${TIME_SERIES_TOPK_RATIO:-0.3}" \
run_logged_job \
  "worker3/time_series_recommended" \
  "$LOG_DIR/time_series_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_time_series_recommended.sh

echo "[worker3] time_series_recommended output_dir=${TIME_SERIES_OUTPUT_DIR:-results/time_series_ssml_topk_v1}"
echo "[worker3] job finished"
