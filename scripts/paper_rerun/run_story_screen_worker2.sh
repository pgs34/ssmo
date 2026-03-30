#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
PAPER_RERUN_TAG="${PAPER_RERUN_TAG:-story_screen_v2}"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$(paper_results_root)/logs/worker2_story}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
OPERATOR_GPU="${OPERATOR_GPU:-0}"
SEEDS="${SEEDS:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(paper_results_root)/story_screen/operator}"

OPERATOR_DATASETS_HOMO="${OPERATOR_DATASETS_HOMO:-darcy}"
OPERATOR_DATASETS_HETERO="${OPERATOR_DATASETS_HETERO:-darcy}"
OPERATOR_MODEL_PAIRS_HOMO="${OPERATOR_MODEL_PAIRS_HOMO:-fno:fno deeponet:deeponet}"
OPERATOR_MODEL_PAIRS_HETERO="${OPERATOR_MODEL_PAIRS_HETERO:-fno:deeponet}"
OPERATOR_EPOCHS="${OPERATOR_EPOCHS:-150}"
OPERATOR_BATCH_SIZE="${OPERATOR_BATCH_SIZE:-8}"
OPERATOR_NUM_WORKERS="${OPERATOR_NUM_WORKERS:-2}"
OPERATOR_REGRESSION_IMITATION_LOSS="${OPERATOR_REGRESSION_IMITATION_LOSS:-mse}"
OPERATOR_DML_LAMBDAS="${OPERATOR_DML_LAMBDAS:-0.05 0.1 0.3}"
OPERATOR_SSML_LAMBDAS="${OPERATOR_SSML_LAMBDAS:-0.1 0.3 0.5}"
OPERATOR_SSML_MARGINS="${OPERATOR_SSML_MARGINS:-0.0 0.02}"
OPERATOR_WARMUP_EPOCHS="${OPERATOR_WARMUP_EPOCHS:-5}"
OPERATOR_IMITATION_DECAY_START_EPOCH="${OPERATOR_IMITATION_DECAY_START_EPOCH:-50}"
OPERATOR_IMITATION_DECAY_END_EPOCH="${OPERATOR_IMITATION_DECAY_END_EPOCH:-120}"
OPERATOR_IMITATION_DECAY_MIN_SCALE="${OPERATOR_IMITATION_DECAY_MIN_SCALE:-0.2}"
OPERATOR_HETERO_SSML_ONE_WAY="${OPERATOR_HETERO_SSML_ONE_WAY:-1}"

slug_float() {
  local value="${1//./p}"
  value="${value//-/m}"
  printf '%s\n' "$value"
}

run_operator_job() {
  local label="$1"
  shift
  run_logged_job \
    "worker2/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$OPERATOR_GPU" \
      DEVICE="$DEVICE" \
      SEEDS="$SEEDS" \
      EPOCHS="$OPERATOR_EPOCHS" \
      BATCH_SIZE="$OPERATOR_BATCH_SIZE" \
      NUM_WORKERS="$OPERATOR_NUM_WORKERS" \
      REGRESSION_IMITATION_LOSS="$OPERATOR_REGRESSION_IMITATION_LOSS" \
      IMITATION_DECAY_START_EPOCH="$OPERATOR_IMITATION_DECAY_START_EPOCH" \
      IMITATION_DECAY_END_EPOCH="$OPERATOR_IMITATION_DECAY_END_EPOCH" \
      IMITATION_DECAY_MIN_SCALE="$OPERATOR_IMITATION_DECAY_MIN_SCALE" \
      HETERO_SSML_ONE_WAY="$OPERATOR_HETERO_SSML_ONE_WAY" \
      "$@" \
      bash scripts/paper_rerun/run_core_operator.sh
}

echo "[worker2_story] results_root=$(paper_results_root)"
echo "[worker2_story] output_root=$OUTPUT_ROOT"
echo "[worker2_story] gpu=$OPERATOR_GPU seeds=$SEEDS"

run_operator_job \
  "operator_baseline_seed${SEEDS// /_}" \
  METHODS="independent" \
  DATASETS="$OPERATOR_DATASETS_HOMO $OPERATOR_DATASETS_HETERO" \
  MODEL_PAIRS="$OPERATOR_MODEL_PAIRS_HOMO $OPERATOR_MODEL_PAIRS_HETERO" \
  REQUIRE_DISTINCT_PEER="0" \
  OUTPUT_DIR="$OUTPUT_ROOT/baseline"

for lambda_imitation in $OPERATOR_DML_LAMBDAS; do
  lambda_slug="$(slug_float "$lambda_imitation")"
  run_operator_job \
    "operator_homo_dml_l${lambda_slug}" \
    METHODS="dml" \
    DATASETS="$OPERATOR_DATASETS_HOMO" \
    MODEL_PAIRS="$OPERATOR_MODEL_PAIRS_HOMO" \
    REQUIRE_DISTINCT_PEER="0" \
    LAMBDA_IMITATION="$lambda_imitation" \
    MARGIN="0.0" \
    WARMUP_EPOCHS="$OPERATOR_WARMUP_EPOCHS" \
    OUTPUT_DIR="$OUTPUT_ROOT/homogeneous/dml_l${lambda_slug}"

  run_operator_job \
    "operator_hetero_dml_l${lambda_slug}" \
    METHODS="dml" \
    DATASETS="$OPERATOR_DATASETS_HETERO" \
    MODEL_PAIRS="$OPERATOR_MODEL_PAIRS_HETERO" \
    REQUIRE_DISTINCT_PEER="1" \
    LAMBDA_IMITATION="$lambda_imitation" \
    MARGIN="0.0" \
    WARMUP_EPOCHS="$OPERATOR_WARMUP_EPOCHS" \
    OUTPUT_DIR="$OUTPUT_ROOT/heterogeneous/dml_l${lambda_slug}"
done

for lambda_imitation in $OPERATOR_SSML_LAMBDAS; do
  lambda_slug="$(slug_float "$lambda_imitation")"
  for margin in $OPERATOR_SSML_MARGINS; do
    margin_slug="$(slug_float "$margin")"
    run_operator_job \
      "operator_homo_ssml_l${lambda_slug}_m${margin_slug}" \
      METHODS="ssml" \
      DATASETS="$OPERATOR_DATASETS_HOMO" \
      MODEL_PAIRS="$OPERATOR_MODEL_PAIRS_HOMO" \
      REQUIRE_DISTINCT_PEER="0" \
      LAMBDA_IMITATION="$lambda_imitation" \
      MARGIN="$margin" \
      WARMUP_EPOCHS="$OPERATOR_WARMUP_EPOCHS" \
      OUTPUT_DIR="$OUTPUT_ROOT/homogeneous/ssml_l${lambda_slug}_m${margin_slug}"

    run_operator_job \
      "operator_hetero_ssml_l${lambda_slug}_m${margin_slug}" \
      METHODS="ssml" \
      DATASETS="$OPERATOR_DATASETS_HETERO" \
      MODEL_PAIRS="$OPERATOR_MODEL_PAIRS_HETERO" \
      REQUIRE_DISTINCT_PEER="1" \
      LAMBDA_IMITATION="$lambda_imitation" \
      MARGIN="$margin" \
      WARMUP_EPOCHS="$OPERATOR_WARMUP_EPOCHS" \
      OUTPUT_DIR="$OUTPUT_ROOT/heterogeneous/ssml_l${lambda_slug}_m${margin_slug}"
  done
done

echo "[worker2_story] all jobs finished"
