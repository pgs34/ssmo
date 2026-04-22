#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../common" && pwd)/_common.sh"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/instruction_matrix_v1}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/logs/weather/run.log}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"

run_locked_job "weather_instruction_matrix_v1" "weather" "$LOG_FILE" env \
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  DATASETS="${DATASETS:-weather}" \
  METHODS="${METHODS:-independent dml ssml}" \
  MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}" \
  INDEPENDENT_MODELS="${INDEPENDENT_MODELS:-transformer dlinear}" \
  REQUIRE_DISTINCT_PEER="${REQUIRE_DISTINCT_PEER:-1}" \
  SEEDS="${SEEDS:-0 1 2}" \
  EPOCHS="${EPOCHS:-60}" \
  BATCH_SIZE="${BATCH_SIZE:-64}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  DEVICE="${DEVICE:-cuda}" \
  LR="${LR:-1e-3}" \
  WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}" \
  SEQ_LEN="${SEQ_LEN:-96}" \
  PRED_LENS="${PRED_LENS:-24}" \
  FEATURE_MODE="${FEATURE_MODE:-multivariate}" \
  REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-mse}" \
  bash "$ROOT_DIR/scripts/common/run_core_time_series.sh"
