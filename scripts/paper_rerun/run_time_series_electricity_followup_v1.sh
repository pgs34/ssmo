#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_electricity_followup_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_electricity_followup_v1}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-mse}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
CASE_SPECS="${CASE_SPECS:-best_known:0.30:0.05:5:20:50:0.10:1:0.30 best_known_topk25:0.30:0.05:5:20:50:0.10:1:0.25 best_known_margin03:0.30:0.03:5:20:50:0.10:1:0.30}"

run_case() {
  local label="$1"
  shift
  run_logged_job \
    "time_series_electricity_followup_v1/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      OUTPUT_DIR="$OUTPUT_ROOT/$label" \
      DATASETS="electricity" \
      METHODS="independent dml ssml" \
      MODEL_PAIRS="transformer:dlinear" \
      REQUIRE_DISTINCT_PEER="1" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      SEQ_LEN="$SEQ_LEN" \
      PRED_LENS="$PRED_LENS" \
      REGRESSION_IMITATION_LOSS="$REGRESSION_IMITATION_LOSS" \
      FEATURE_MODE="$FEATURE_MODE" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_electricity_followup_v1] output_root=$OUTPUT_ROOT"
echo "[time_series_electricity_followup_v1] gpu=$GPU seeds=$SEEDS"
echo "[time_series_electricity_followup_v1] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label lambda margin warmup decay_start decay_end decay_min_scale one_way topk <<< "$spec"
  run_case \
    "$label" \
    LAMBDA_IMITATION="$lambda" \
    MARGIN="$margin" \
    WARMUP_EPOCHS="$warmup" \
    IMITATION_DECAY_START_EPOCH="$decay_start" \
    IMITATION_DECAY_END_EPOCH="$decay_end" \
    IMITATION_DECAY_MIN_SCALE="$decay_min_scale" \
    HETERO_SSML_ONE_WAY="$one_way" \
    SSML_TOPK_RATIO="$topk"
done
