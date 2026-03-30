#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_ssml_reweight_v6}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
DATASETS="${DATASETS:-cifar10 cifar100}"
MODEL_PAIRS="${MODEL_PAIRS:-resnet18:resnet18 vit_b16:vit_b16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_ssml_reweight_v6}"
DOWNLOAD="${DOWNLOAD:-1}"

CIFAR10_TOPKS="${CIFAR10_TOPKS:-0.05 0.1}"
CIFAR100_TOPKS="${CIFAR100_TOPKS:-0.01 0.02}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-4.0}"
LAMBDA_IMITATION="${LAMBDA_IMITATION:-0.0}"
MARGIN="${MARGIN:-0.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-40}"
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-90}"
IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.2}"
SSML_SUPERVISED_HOTSPOT_ALPHA="${SSML_SUPERVISED_HOTSPOT_ALPHA:-0.5}"
SSML_GATE_SCORE_MODE="${SSML_GATE_SCORE_MODE:-peer_better_student_error}"
SSML_SCORE_TRANSFORM="${SSML_SCORE_TRANSFORM:-log1p}"
SSML_GUIDANCE_MODE="${SSML_GUIDANCE_MODE:-reweight_only}"
SSML_PEER_CORRECT_ONLY="${SSML_PEER_CORRECT_ONLY:-1}"
SSML_STUDENT_INCORRECT_ONLY="${SSML_STUDENT_INCORRECT_ONLY:-1}"
SSML_PEER_TRUE_PROB_THRESHOLD="${SSML_PEER_TRUE_PROB_THRESHOLD:-0.4}"

slug_float() {
  local value="${1//./p}"
  value="${value//-/m}"
  printf '%s\n' "$value"
}

dataset_enabled() {
  array_contains "$1" $DATASETS
}

run_job() {
  local label="$1"
  shift
  run_logged_job \
    "classification_reweight_v6/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      METHODS="ssml" \
      DATASETS="$1" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="0" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      DOWNLOAD="$DOWNLOAD" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE" \
      LAMBDA_IMITATION="$LAMBDA_IMITATION" \
      MARGIN="$MARGIN" \
      WARMUP_EPOCHS="$WARMUP_EPOCHS" \
      IMITATION_DECAY_START_EPOCH="$IMITATION_DECAY_START_EPOCH" \
      IMITATION_DECAY_END_EPOCH="$IMITATION_DECAY_END_EPOCH" \
      IMITATION_DECAY_MIN_SCALE="$IMITATION_DECAY_MIN_SCALE" \
      SSML_SUPERVISED_HOTSPOT_ALPHA="$SSML_SUPERVISED_HOTSPOT_ALPHA" \
      SSML_GATE_SCORE_MODE="$SSML_GATE_SCORE_MODE" \
      SSML_SCORE_TRANSFORM="$SSML_SCORE_TRANSFORM" \
      SSML_GUIDANCE_MODE="$SSML_GUIDANCE_MODE" \
      SSML_PEER_CORRECT_ONLY="$SSML_PEER_CORRECT_ONLY" \
      SSML_STUDENT_INCORRECT_ONLY="$SSML_STUDENT_INCORRECT_ONLY" \
      SSML_PEER_TRUE_PROB_THRESHOLD="$SSML_PEER_TRUE_PROB_THRESHOLD" \
      "${@:2}" \
      bash scripts/paper_rerun/run_core_classification.sh
}

echo "[classification_ssml_reweight_v6] output_root=$OUTPUT_ROOT"
echo "[classification_ssml_reweight_v6] gpu=$GPU datasets=$DATASETS seeds=$SEEDS"
echo "[classification_ssml_reweight_v6] cifar10_topks=$CIFAR10_TOPKS"
echo "[classification_ssml_reweight_v6] cifar100_topks=$CIFAR100_TOPKS"
echo "[classification_ssml_reweight_v6] warmup=$WARMUP_EPOCHS decay=${IMITATION_DECAY_START_EPOCH}->${IMITATION_DECAY_END_EPOCH} min_scale=$IMITATION_DECAY_MIN_SCALE"
echo "[classification_ssml_reweight_v6] hotspot_alpha=$SSML_SUPERVISED_HOTSPOT_ALPHA gate=$SSML_GATE_SCORE_MODE transform=$SSML_SCORE_TRANSFORM"
echo "[classification_ssml_reweight_v6] peer_correct_only=$SSML_PEER_CORRECT_ONLY student_incorrect_only=$SSML_STUDENT_INCORRECT_ONLY peer_true_prob_threshold=$SSML_PEER_TRUE_PROB_THRESHOLD"

if dataset_enabled "cifar10"; then
  for topk in $CIFAR10_TOPKS; do
    topk_slug="$(slug_float "$topk")"
    run_job \
      "cifar10_ssml_reweight_t${topk_slug}" \
      "cifar10" \
      SSML_TOPK_RATIO="$topk" \
      OUTPUT_DIR="$OUTPUT_ROOT/cifar10/ssml_reweight_t${topk_slug}"
  done
fi

if dataset_enabled "cifar100"; then
  for topk in $CIFAR100_TOPKS; do
    topk_slug="$(slug_float "$topk")"
    run_job \
      "cifar100_ssml_reweight_t${topk_slug}" \
      "cifar100" \
      SSML_TOPK_RATIO="$topk" \
      OUTPUT_DIR="$OUTPUT_ROOT/cifar100/ssml_reweight_t${topk_slug}"
  done
fi
