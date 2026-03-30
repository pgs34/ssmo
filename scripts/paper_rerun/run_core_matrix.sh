#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

GPU_CLASSIFICATION="${GPU_CLASSIFICATION:-0}"
GPU_TIME_SERIES="${GPU_TIME_SERIES:-1}"
GPU_OPERATOR="${GPU_OPERATOR:-0}"
RUN_OPERATOR="${RUN_OPERATOR:-0}"

echo "[paper_rerun] results_root=$(paper_results_root)"
echo "[paper_rerun] classification on GPU $GPU_CLASSIFICATION"
CUDA_VISIBLE_DEVICES="$GPU_CLASSIFICATION" bash scripts/paper_rerun/run_core_classification.sh

if [[ "$RUN_OPERATOR" == "1" ]]; then
  echo "[paper_rerun] operator on GPU $GPU_OPERATOR"
  CUDA_VISIBLE_DEVICES="$GPU_OPERATOR" bash scripts/paper_rerun/run_core_operator.sh
fi

echo "[paper_rerun] time_series on GPU $GPU_TIME_SERIES"
CUDA_VISIBLE_DEVICES="$GPU_TIME_SERIES" bash scripts/paper_rerun/run_core_time_series.sh
