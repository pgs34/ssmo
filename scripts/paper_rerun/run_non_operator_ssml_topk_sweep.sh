#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

TIME_SERIES_GPU="${TIME_SERIES_GPU:-${GPU:-0}}"
CLASSIFICATION_GPU="${CLASSIFICATION_GPU:-${GPU:-0}}"

echo "[non_operator_ssml_topk_sweep] time_series_gpu=$TIME_SERIES_GPU classification_gpu=$CLASSIFICATION_GPU"
echo "[non_operator_ssml_topk_sweep] starting time-series"
GPU="$TIME_SERIES_GPU" bash scripts/paper_rerun/run_time_series_ssml_topk_sweep.sh

echo "[non_operator_ssml_topk_sweep] starting classification"
GPU="$CLASSIFICATION_GPU" bash scripts/paper_rerun/run_classification_ssml_topk_sweep.sh
