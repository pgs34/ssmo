#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_ssml_rescue}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0}"
DATASETS="${DATASETS:-cifar10 cifar100}"
HOMO_MODEL_PAIRS="${HOMO_MODEL_PAIRS:-resnet18:resnet18 vit_b16:vit_b16}"
HETERO_MODEL_PAIRS="${HETERO_MODEL_PAIRS:-resnet18:vit_b16 vit_b16:resnet18}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_ssml_rescue_v1}"
DOWNLOAD="${DOWNLOAD:-1}"

LAMBDAS="${LAMBDAS:-0.02 0.05}"
MARGINS="${MARGINS:-0.05 0.1}"
TEMPERATURES="${TEMPERATURES:-4.0 8.0}"
WARMUPS="${WARMUPS:-10}"
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-30}"
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-55}"
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
    "classification_rescue/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      METHODS="ssml" \
      DATASETS="$DATASETS" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      DOWNLOAD="$DOWNLOAD" \
      IMITATION_DECAY_START_EPOCH="$IMITATION_DECAY_START_EPOCH" \
      IMITATION_DECAY_END_EPOCH="$IMITATION_DECAY_END_EPOCH" \
      IMITATION_DECAY_MIN_SCALE="$IMITATION_DECAY_MIN_SCALE" \
      "$@" \
      bash scripts/paper_rerun/run_core_classification.sh
}

echo "[classification_ssml_rescue] output_root=$OUTPUT_ROOT"
echo "[classification_ssml_rescue] datasets=$DATASETS seeds=$SEEDS"
echo "[classification_ssml_rescue] homo_model_pairs=$HOMO_MODEL_PAIRS"
echo "[classification_ssml_rescue] hetero_model_pairs=$HETERO_MODEL_PAIRS"
echo "[classification_ssml_rescue] lambdas=$LAMBDAS margins=$MARGINS temperatures=$TEMPERATURES warmups=$WARMUPS"
echo "[classification_ssml_rescue] decay=[$IMITATION_DECAY_START_EPOCH,$IMITATION_DECAY_END_EPOCH] min_scale=$IMITATION_DECAY_MIN_SCALE"

for lambda_imitation in $LAMBDAS; do
  lambda_slug="$(slug_float "$lambda_imitation")"
  for margin in $MARGINS; do
    margin_slug="$(slug_float "$margin")"
    for temperature in $TEMPERATURES; do
      temp_slug="$(slug_float "$temperature")"
      for warmup in $WARMUPS; do
        run_job \
          "cls_hetero_ssml_l${lambda_slug}_m${margin_slug}_t${temp_slug}_w${warmup}" \
          MODEL_PAIRS="$HETERO_MODEL_PAIRS" \
          REQUIRE_DISTINCT_PEER="1" \
          HETERO_SSML_ONE_WAY="0" \
          LAMBDA_IMITATION="$lambda_imitation" \
          MARGIN="$margin" \
          WARMUP_EPOCHS="$warmup" \
          DISTILL_TEMPERATURE="$temperature" \
          OUTPUT_DIR="$OUTPUT_ROOT/heterogeneous/ssml_l${lambda_slug}_m${margin_slug}_t${temp_slug}_w${warmup}"

        run_job \
          "cls_homo_ssml_l${lambda_slug}_m${margin_slug}_t${temp_slug}_w${warmup}" \
          MODEL_PAIRS="$HOMO_MODEL_PAIRS" \
          REQUIRE_DISTINCT_PEER="0" \
          HETERO_SSML_ONE_WAY="0" \
          LAMBDA_IMITATION="$lambda_imitation" \
          MARGIN="$margin" \
          WARMUP_EPOCHS="$warmup" \
          DISTILL_TEMPERATURE="$temperature" \
          OUTPUT_DIR="$OUTPUT_ROOT/homogeneous/ssml_l${lambda_slug}_m${margin_slug}_t${temp_slug}_w${warmup}"
      done
    done
  done
done
