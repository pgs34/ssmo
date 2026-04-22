#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../common" && pwd)/_common.sh"

CORE_OUTPUT_DIR="${CORE_OUTPUT_DIR:-$ROOT_DIR/results/instruction_matrix_v1}"
DML_OUTPUT_DIR="${DML_OUTPUT_DIR:-$ROOT_DIR/results/classification_cifar10_homo_dml_long_v1}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/cifar10}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"
CIFAR10_STAGE_MAX_PARALLEL_RUNS="${CIFAR10_STAGE_MAX_PARALLEL_RUNS:-all}"

parallel_exec_init "$CIFAR10_STAGE_MAX_PARALLEL_RUNS"
trap parallel_exec_cleanup INT TERM

parallel_exec_submit "cifar10.core" run_locked_job "cifar10_instruction_matrix_v1" "cifar10.core" "$LOG_DIR/core.log" env \
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  OUTPUT_DIR="$CORE_OUTPUT_DIR" \
  DATASETS="${DATASETS:-cifar10}" \
  METHODS="${METHODS:-independent ssml}" \
  MODEL_PAIRS="${MODEL_PAIRS:-resnet18:resnet18}" \
  INDEPENDENT_MODELS="${INDEPENDENT_MODELS:-resnet18}" \
  REQUIRE_DISTINCT_PEER="${REQUIRE_DISTINCT_PEER:-0}" \
  LABEL_NOISE_CONDITIONS="${LABEL_NOISE_CONDITIONS:-none:0.0}" \
  SEEDS="${SEEDS:-0 1 2}" \
  EPOCHS="${EPOCHS:-100}" \
  BATCH_SIZE="${BATCH_SIZE:-128}" \
  NUM_WORKERS="${NUM_WORKERS:-8}" \
  DEVICE="${DEVICE:-cuda}" \
  DOWNLOAD="${DOWNLOAD:-1}" \
  bash "$ROOT_DIR/scripts/common/run_core_classification.sh"

parallel_exec_submit "cifar10.dml" run_locked_job "cifar10_homo_dml_long_v1" "cifar10.dml" "$LOG_DIR/dml.log" env \
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  OUTPUT_DIR="$DML_OUTPUT_DIR" \
  DATASETS="${DATASETS:-cifar10}" \
  METHODS="${METHODS:-dml}" \
  MODEL_PAIRS="${MODEL_PAIRS:-resnet18:resnet18}" \
  REQUIRE_DISTINCT_PEER="${REQUIRE_DISTINCT_PEER:-0}" \
  LABEL_NOISE_CONDITIONS="${LABEL_NOISE_CONDITIONS:-none:0.0}" \
  SEEDS="${SEEDS:-0 1 2}" \
  EPOCHS="${EPOCHS:-100}" \
  BATCH_SIZE="${BATCH_SIZE:-256}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  DEVICE="${DEVICE:-cuda}" \
  DOWNLOAD="${DOWNLOAD:-1}" \
  CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}" \
  DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-4.0}" \
  LAMBDA_IMITATION="${LAMBDA_IMITATION:-0.02}" \
  MARGIN="${MARGIN:-0.05}" \
  SSML_TOPK_RATIO="${SSML_TOPK_RATIO:-0.0}" \
  WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}" \
  IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-30}" \
  IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-85}" \
  IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.05}" \
  bash "$ROOT_DIR/scripts/common/run_core_classification.sh"

parallel_exec_wait_all
