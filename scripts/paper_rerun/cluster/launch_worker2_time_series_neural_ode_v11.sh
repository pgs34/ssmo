#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker2}"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
GPU="${TIME_SERIES_GPU:-0}" \
DATASETS="${TIME_SERIES_DATASETS:-weather}" \
MODEL_PAIRS="${TIME_SERIES_MODEL_PAIRS:-neural_ode:dlinear neural_ode:transformer_wide}" \
OUTPUT_ROOT="${TIME_SERIES_OUTPUT_ROOT:-results/time_series_neural_ode_weather_v11}" \
BATCH_SIZE="${TIME_SERIES_BATCH_SIZE:-48}" \
CASE_SPECS="${TIME_SERIES_CASE_SPECS:-node_a01_t03:0.10:0.03:0.90 node_a02_t03:0.20:0.03:0.90}" \
run_logged_job \
  "worker2/time_series_neural_ode_v11_gpu${TIME_SERIES_GPU:-0}" \
  "$LOG_DIR/time_series_neural_ode_v11_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_time_series_neural_ode_v11.sh

echo "[worker2_time_series_neural_ode_v11] job finished"
