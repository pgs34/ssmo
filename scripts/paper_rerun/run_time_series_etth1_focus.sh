#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_focus}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_focus_v1}"

EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear dlinear:transformer}"

VERIFY_LAMBDAS="${VERIFY_LAMBDAS:-0.1 0.3}"
VERIFY_MARGINS="${VERIFY_MARGINS:-0.0 0.02}"
VERIFY_TOPKS="${VERIFY_TOPKS:-0.1 0.2}"
VERIFY_WARMUP="${VERIFY_WARMUP:-5}"
VERIFY_DECAY_START="${VERIFY_DECAY_START:-15}"
VERIFY_DECAY_END="${VERIFY_DECAY_END:-45}"
VERIFY_DECAY_MIN="${VERIFY_DECAY_MIN:-0.2}"

FOCUS_LAMBDAS="${FOCUS_LAMBDAS:-0.02 0.05}"
FOCUS_MARGINS="${FOCUS_MARGINS:-0.05 0.1}"
FOCUS_TOPKS="${FOCUS_TOPKS:-0.02 0.05}"
FOCUS_LOSSES="${FOCUS_LOSSES:-huber mae}"
FOCUS_WARMUP="${FOCUS_WARMUP:-8}"
FOCUS_DECAY_START="${FOCUS_DECAY_START:-15}"
FOCUS_DECAY_END="${FOCUS_DECAY_END:-40}"
FOCUS_DECAY_MIN="${FOCUS_DECAY_MIN:-0.02}"

slug_float() {
  local value="${1//./p}"
  value="${value//-/m}"
  printf '%s\n' "$value"
}

run_job() {
  local label="$1"
  shift
  run_logged_job \
    "etth1_focus/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      METHODS="ssml" \
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

echo "[time_series_etth1_focus] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_focus] seeds=$SEEDS model_pairs=$MODEL_PAIRS"
echo "[time_series_etth1_focus] verify: lambdas=$VERIFY_LAMBDAS margins=$VERIFY_MARGINS topks=$VERIFY_TOPKS"
echo "[time_series_etth1_focus] focus: lambdas=$FOCUS_LAMBDAS margins=$FOCUS_MARGINS topks=$FOCUS_TOPKS losses=$FOCUS_LOSSES"

# Phase 1: reproduce the previous story-screen sweet spot locally on this node.
for lambda_imitation in $VERIFY_LAMBDAS; do
  lambda_slug="$(slug_float "$lambda_imitation")"
  for margin in $VERIFY_MARGINS; do
    margin_slug="$(slug_float "$margin")"
    for topk_ratio in $VERIFY_TOPKS; do
      topk_slug="$(slug_float "$topk_ratio")"
      run_job \
        "verify_l${lambda_slug}_m${margin_slug}_t${topk_slug}" \
        REGRESSION_IMITATION_LOSS="mse" \
        HETERO_SSML_ONE_WAY="1" \
        LAMBDA_IMITATION="$lambda_imitation" \
        MARGIN="$margin" \
        WARMUP_EPOCHS="$VERIFY_WARMUP" \
        IMITATION_DECAY_START_EPOCH="$VERIFY_DECAY_START" \
        IMITATION_DECAY_END_EPOCH="$VERIFY_DECAY_END" \
        IMITATION_DECAY_MIN_SCALE="$VERIFY_DECAY_MIN" \
        SSML_TOPK_RATIO="$topk_ratio" \
        OUTPUT_DIR="$OUTPUT_ROOT/verify/ssml_l${lambda_slug}_m${margin_slug}_t${topk_slug}"
    done
  done
done

# Phase 2: more aggressive sparse-gate rescue around the current best local setting.
for loss_name in $FOCUS_LOSSES; do
  for lambda_imitation in $FOCUS_LAMBDAS; do
    lambda_slug="$(slug_float "$lambda_imitation")"
    for margin in $FOCUS_MARGINS; do
      margin_slug="$(slug_float "$margin")"
      for topk_ratio in $FOCUS_TOPKS; do
        topk_slug="$(slug_float "$topk_ratio")"
        run_job \
          "focus_${loss_name}_l${lambda_slug}_m${margin_slug}_t${topk_slug}" \
          REGRESSION_IMITATION_LOSS="$loss_name" \
          HETERO_SSML_ONE_WAY="0" \
          LAMBDA_IMITATION="$lambda_imitation" \
          MARGIN="$margin" \
          WARMUP_EPOCHS="$FOCUS_WARMUP" \
          IMITATION_DECAY_START_EPOCH="$FOCUS_DECAY_START" \
          IMITATION_DECAY_END_EPOCH="$FOCUS_DECAY_END" \
          IMITATION_DECAY_MIN_SCALE="$FOCUS_DECAY_MIN" \
          SSML_TOPK_RATIO="$topk_ratio" \
          OUTPUT_DIR="$OUTPUT_ROOT/focus/${loss_name}/ssml_l${lambda_slug}_m${margin_slug}_t${topk_slug}"
      done
    done
  done
done
