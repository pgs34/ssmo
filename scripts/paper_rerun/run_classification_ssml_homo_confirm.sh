#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_ssml_homo_confirm}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
DATASETS="${DATASETS:-cifar10 cifar100}"
HOMO_MODEL_PAIRS="${HOMO_MODEL_PAIRS:-resnet18:resnet18 vit_b16:vit_b16}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_ssml_homo_confirm_v1}"
DOWNLOAD="${DOWNLOAD:-1}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"

LAMBDAS="${LAMBDAS:-0.02 0.05}"
TEMPERATURES="${TEMPERATURES:-4.0 8.0}"
MARGIN="${MARGIN:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-30}"
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-80}"
IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.05}"

slug_float() {
  local value="${1//./p}"
  value="${value//-/m}"
  printf '%s\n' "$value"
}

run_job() {
  local label="$1"
  shift
  run_logged_job \
    "classification_homo_confirm/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      DATASETS="$DATASETS" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      DOWNLOAD="$DOWNLOAD" \
      "$@" \
      bash scripts/paper_rerun/run_core_classification.sh
}

echo "[classification_ssml_homo_confirm] output_root=$OUTPUT_ROOT"
echo "[classification_ssml_homo_confirm] datasets=$DATASETS seeds=$SEEDS"
echo "[classification_ssml_homo_confirm] homo_model_pairs=$HOMO_MODEL_PAIRS"
echo "[classification_ssml_homo_confirm] lambdas=$LAMBDAS temperatures=$TEMPERATURES margin=$MARGIN warmup=$WARMUP_EPOCHS"

if [[ "$INCLUDE_BASELINE" == "1" ]]; then
  run_job \
    "classification_homo_baseline" \
    METHODS="independent" \
    MODEL_PAIRS="$HOMO_MODEL_PAIRS" \
    REQUIRE_DISTINCT_PEER="0" \
    LAMBDA_IMITATION="0.0" \
    MARGIN="0.0" \
    WARMUP_EPOCHS="0" \
    OUTPUT_DIR="$OUTPUT_ROOT/baseline"
fi

for lambda_imitation in $LAMBDAS; do
  lambda_slug="$(slug_float "$lambda_imitation")"
  for temperature in $TEMPERATURES; do
    temp_slug="$(slug_float "$temperature")"
    run_job \
      "classification_homo_ssml_l${lambda_slug}_t${temp_slug}" \
      METHODS="ssml" \
      MODEL_PAIRS="$HOMO_MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="0" \
      HETERO_SSML_ONE_WAY="0" \
      LAMBDA_IMITATION="$lambda_imitation" \
      MARGIN="$MARGIN" \
      WARMUP_EPOCHS="$WARMUP_EPOCHS" \
      DISTILL_TEMPERATURE="$temperature" \
      IMITATION_DECAY_START_EPOCH="$IMITATION_DECAY_START_EPOCH" \
      IMITATION_DECAY_END_EPOCH="$IMITATION_DECAY_END_EPOCH" \
      IMITATION_DECAY_MIN_SCALE="$IMITATION_DECAY_MIN_SCALE" \
      OUTPUT_DIR="$OUTPUT_ROOT/ssml_l${lambda_slug}_t${temp_slug}"
  done
done
