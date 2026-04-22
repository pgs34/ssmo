#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../common" && pwd)/_common.sh"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/operator_ssml_tuned_v1}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/logs/darcy/run.log}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"

run_locked_job "darcy_operator_ssml_tuned_v1" "darcy" "$LOG_FILE" env \
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  DATASETS="${DATASETS:-darcy}" \
  METHODS="${METHODS:-independent dml ssml}" \
  MODEL_PAIRS="${MODEL_PAIRS:-fno:deeponet}" \
  INDEPENDENT_MODELS="${INDEPENDENT_MODELS:-fno deeponet}" \
  REQUIRE_DISTINCT_PEER="${REQUIRE_DISTINCT_PEER:-1}" \
  SEEDS="${SEEDS:-0 1 2}" \
  EPOCHS="${EPOCHS:-150}" \
  BATCH_SIZE="${BATCH_SIZE:-16}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  DEVICE="${DEVICE:-cuda}" \
  LR="${LR:-0.001}" \
  WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}" \
  REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-mse}" \
  LAMBDA_IMITATION="${LAMBDA_IMITATION:-0.3}" \
  MARGIN="${MARGIN:-0.02}" \
  WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}" \
  IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-30}" \
  IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-120}" \
  IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.10}" \
  HETERO_SSML_ONE_WAY="${HETERO_SSML_ONE_WAY:-1}" \
  DOWNLOAD="${DOWNLOAD:-0}" \
  bash "$ROOT_DIR/scripts/common/run_core_operator.sh"
