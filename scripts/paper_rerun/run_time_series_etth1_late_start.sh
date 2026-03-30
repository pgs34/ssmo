#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_late_start}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_late_start_v1}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"

EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear dlinear:transformer}"

LAMBDAS="${LAMBDAS:-0.01 0.02}"
MARGINS="${MARGINS:-0.05 0.1}"
TOPKS="${TOPKS:-0.01 0.02 0.05}"
LOSSES="${LOSSES:-huber mae}"
WARMUPS="${WARMUPS:-20 30}"
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-25}"
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-35}"
IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.0}"

slug_float() {
  local value="${1//./p}"
  value="${value//-/m}"
  printf '%s\n' "$value"
}

run_job() {
  local label="$1"
  shift
  run_logged_job \
    "etth1_late_start/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      DATASETS="etth1" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="1" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      SEQ_LEN="$SEQ_LEN" \
      PRED_LENS="$PRED_LENS" \
      FEATURE_MODE="$FEATURE_MODE" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_etth1_late_start] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_late_start] seeds=$SEEDS lambdas=$LAMBDAS margins=$MARGINS topks=$TOPKS losses=$LOSSES warmups=$WARMUPS"

if [[ "$INCLUDE_BASELINE" == "1" ]]; then
  run_job \
    "etth1_baseline" \
    METHODS="independent" \
    LAMBDA_IMITATION="0.0" \
    MARGIN="0.0" \
    WARMUP_EPOCHS="0" \
    OUTPUT_DIR="$OUTPUT_ROOT/baseline"
fi

for loss_name in $LOSSES; do
  for lambda_imitation in $LAMBDAS; do
    lambda_slug="$(slug_float "$lambda_imitation")"
    for margin in $MARGINS; do
      margin_slug="$(slug_float "$margin")"
      for topk_ratio in $TOPKS; do
        topk_slug="$(slug_float "$topk_ratio")"
        for warmup in $WARMUPS; do
          run_job \
            "etth1_ssml_${loss_name}_l${lambda_slug}_m${margin_slug}_t${topk_slug}_w${warmup}" \
            METHODS="ssml" \
            REGRESSION_IMITATION_LOSS="$loss_name" \
            HETERO_SSML_ONE_WAY="0" \
            LAMBDA_IMITATION="$lambda_imitation" \
            MARGIN="$margin" \
            WARMUP_EPOCHS="$warmup" \
            IMITATION_DECAY_START_EPOCH="$IMITATION_DECAY_START_EPOCH" \
            IMITATION_DECAY_END_EPOCH="$IMITATION_DECAY_END_EPOCH" \
            IMITATION_DECAY_MIN_SCALE="$IMITATION_DECAY_MIN_SCALE" \
            SSML_TOPK_RATIO="$topk_ratio" \
            OUTPUT_DIR="$OUTPUT_ROOT/${loss_name}/ssml_l${lambda_slug}_m${margin_slug}_t${topk_slug}_w${warmup}"
        done
      done
    done
  done
done
