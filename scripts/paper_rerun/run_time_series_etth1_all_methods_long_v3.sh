#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

export LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_all_methods_long_v3}"
mkdir -p "$LOG_DIR"

export OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_all_methods_long_v3}"
export SUMMARY_PLOT_ROOT="${SUMMARY_PLOT_ROOT:-results/plots/time_series_etth1_all_methods_long_v3}"
export SEEDS="${SEEDS:-0 1 2 3 4 5}"
export MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
export INDEPENDENT_MODELS="${INDEPENDENT_MODELS:-transformer dlinear}"
export EPOCHS="${EPOCHS:-80}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-huber}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-8}"
export IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-20}"
export IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-70}"
export IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.15}"
export LAMBDA_IMITATION="${LAMBDA_IMITATION:-0.001}"
export MARGIN="${MARGIN:-0.02}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
export EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-0}"
export EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
export REFRESH_TOP_LEVEL="${REFRESH_TOP_LEVEL:-0}"

echo "[time_series_etth1_all_methods_long_v3] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_all_methods_long_v3] summary_plot_root=$SUMMARY_PLOT_ROOT"
echo "[time_series_etth1_all_methods_long_v3] gpu=${GPU:-0} seeds=$SEEDS epochs=$EPOCHS"
echo "[time_series_etth1_all_methods_long_v3] model_pairs=$MODEL_PAIRS independent_models=$INDEPENDENT_MODELS"

bash "$SCRIPT_DIR/run_time_series_etth1_all_methods_long_v2.sh"

