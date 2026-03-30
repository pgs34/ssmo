#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_ssml_topk_sweep}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
DATASETS="${DATASETS:-cifar10 cifar100}"
MODEL_PAIRS="${MODEL_PAIRS:-resnet18:resnet18 vit_b16:vit_b16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_ssml_topk_sweep_v1}"
DOWNLOAD="${DOWNLOAD:-1}"
INCLUDE_BASELINES="${INCLUDE_BASELINES:-1}"

CIFAR10_TOPKS="${CIFAR10_TOPKS:-0.1 0.2 0.3}"
CIFAR100_TOPKS="${CIFAR100_TOPKS:-0.01 0.02 0.05}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-4.0}"
LAMBDA_IMITATION="${LAMBDA_IMITATION:-0.02}"
MARGIN="${MARGIN:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-30}"
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-80}"
IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.05}"
SSML_SUPERVISED_HOTSPOT_ALPHA="${SSML_SUPERVISED_HOTSPOT_ALPHA:-1.0}"

slug_float() {
  local value="${1//./p}"
  value="${value//-/m}"
  printf '%s\n' "$value"
}

run_core_job() {
  local label="$1"
  shift
  run_logged_job \
    "classification_topk/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="0" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      DOWNLOAD="$DOWNLOAD" \
      "$@" \
      bash scripts/paper_rerun/run_core_classification.sh
}

dataset_enabled() {
  array_contains "$1" $DATASETS
}

echo "[classification_ssml_topk_sweep] output_root=$OUTPUT_ROOT"
echo "[classification_ssml_topk_sweep] gpu=$GPU"
echo "[classification_ssml_topk_sweep] datasets=$DATASETS"
echo "[classification_ssml_topk_sweep] cifar10_topks=$CIFAR10_TOPKS"
echo "[classification_ssml_topk_sweep] cifar100_topks=$CIFAR100_TOPKS"
echo "[classification_ssml_topk_sweep] ssml_supervised_hotspot_alpha=$SSML_SUPERVISED_HOTSPOT_ALPHA"

if [[ "$INCLUDE_BASELINES" == "1" ]] && dataset_enabled "cifar10"; then
  run_core_job \
    "cifar10_baseline_dml" \
    DATASETS="cifar10" \
    METHODS="independent dml" \
    CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
    DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE" \
    LAMBDA_IMITATION="$LAMBDA_IMITATION" \
    MARGIN="$MARGIN" \
    SSML_SUPERVISED_HOTSPOT_ALPHA="$SSML_SUPERVISED_HOTSPOT_ALPHA" \
    SSML_TOPK_RATIO="0.0" \
    WARMUP_EPOCHS="$WARMUP_EPOCHS" \
    IMITATION_DECAY_START_EPOCH="$IMITATION_DECAY_START_EPOCH" \
    IMITATION_DECAY_END_EPOCH="$IMITATION_DECAY_END_EPOCH" \
    IMITATION_DECAY_MIN_SCALE="$IMITATION_DECAY_MIN_SCALE" \
    OUTPUT_DIR="$OUTPUT_ROOT/cifar10/baseline_dml"

fi

if [[ "$INCLUDE_BASELINES" == "1" ]] && dataset_enabled "cifar100"; then
  run_core_job \
    "cifar100_baseline_dml" \
    DATASETS="cifar100" \
    METHODS="independent dml" \
    CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
    DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE" \
    LAMBDA_IMITATION="$LAMBDA_IMITATION" \
    MARGIN="$MARGIN" \
    SSML_SUPERVISED_HOTSPOT_ALPHA="$SSML_SUPERVISED_HOTSPOT_ALPHA" \
    SSML_TOPK_RATIO="0.0" \
    WARMUP_EPOCHS="$WARMUP_EPOCHS" \
    IMITATION_DECAY_START_EPOCH="$IMITATION_DECAY_START_EPOCH" \
    IMITATION_DECAY_END_EPOCH="$IMITATION_DECAY_END_EPOCH" \
    IMITATION_DECAY_MIN_SCALE="$IMITATION_DECAY_MIN_SCALE" \
    OUTPUT_DIR="$OUTPUT_ROOT/cifar100/baseline_dml"
fi

if dataset_enabled "cifar10"; then
  for topk in $CIFAR10_TOPKS; do
    topk_slug="$(slug_float "$topk")"
    run_core_job \
      "cifar10_ssml_t${topk_slug}" \
      DATASETS="cifar10" \
      METHODS="ssml" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE" \
      LAMBDA_IMITATION="$LAMBDA_IMITATION" \
      MARGIN="$MARGIN" \
      SSML_SUPERVISED_HOTSPOT_ALPHA="$SSML_SUPERVISED_HOTSPOT_ALPHA" \
      SSML_TOPK_RATIO="$topk" \
      WARMUP_EPOCHS="$WARMUP_EPOCHS" \
      IMITATION_DECAY_START_EPOCH="$IMITATION_DECAY_START_EPOCH" \
      IMITATION_DECAY_END_EPOCH="$IMITATION_DECAY_END_EPOCH" \
      IMITATION_DECAY_MIN_SCALE="$IMITATION_DECAY_MIN_SCALE" \
      OUTPUT_DIR="$OUTPUT_ROOT/cifar10/ssml_t${topk_slug}"
  done
fi

if dataset_enabled "cifar100"; then
  for topk in $CIFAR100_TOPKS; do
    topk_slug="$(slug_float "$topk")"
    run_core_job \
      "cifar100_ssml_t${topk_slug}" \
      DATASETS="cifar100" \
      METHODS="ssml" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE" \
      LAMBDA_IMITATION="$LAMBDA_IMITATION" \
      MARGIN="$MARGIN" \
      SSML_SUPERVISED_HOTSPOT_ALPHA="$SSML_SUPERVISED_HOTSPOT_ALPHA" \
      SSML_TOPK_RATIO="$topk" \
      WARMUP_EPOCHS="$WARMUP_EPOCHS" \
      IMITATION_DECAY_START_EPOCH="$IMITATION_DECAY_START_EPOCH" \
      IMITATION_DECAY_END_EPOCH="$IMITATION_DECAY_END_EPOCH" \
      IMITATION_DECAY_MIN_SCALE="$IMITATION_DECAY_MIN_SCALE" \
      OUTPUT_DIR="$OUTPUT_ROOT/cifar100/ssml_t${topk_slug}"
  done
fi
