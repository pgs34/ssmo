#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

REFERENCE_MODE="${REFERENCE_MODE:-independent}"
LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_scaled_fair_reference_v2}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-3072}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD="${DOWNLOAD:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_cifar100_scaled_fair_reference_v2/${REFERENCE_MODE}}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
PROTOCOL_ID="${PROTOCOL_ID:-scaled_fair_bs${BATCH_SIZE}}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-unknown}"
if [[ -z "${BEST_CKPT_TEMPLATE:-}" ]]; then
  BEST_CKPT_TEMPLATE="results/classification_cifar100_bestckpt_pool_v1/classification/classification/cifar100/{model}_independent_${CLASSIFICATION_IMITATION_LOSS}_seed{seed}/best_model.pt"
fi

METHODS="independent"
INDEPENDENT_MODELS="resnet34_gelu"
MODEL_PAIRS="resnet34_gelu:resnet34_gelu"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-6.0}"
LAMBDA_IMITATION="${LAMBDA_IMITATION:-0.04}"
MARGIN="${MARGIN:-0.0}"

case "$REFERENCE_MODE" in
  independent)
    METHODS="independent"
    ;;
  dml)
    METHODS="dml"
    ;;
  *)
    echo "[classification_cifar100_scaled_fair_reference_v2] unknown REFERENCE_MODE=$REFERENCE_MODE" >&2
    exit 1
    ;;
esac

echo "[classification_cifar100_scaled_fair_reference_v2] mode=$REFERENCE_MODE gpu=$GPU seeds=$SEEDS epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "[classification_cifar100_scaled_fair_reference_v2] protocol_id=$PROTOCOL_ID hardware_profile=$HARDWARE_PROFILE"
echo "[classification_cifar100_scaled_fair_reference_v2] output_root=$OUTPUT_ROOT"
echo "[classification_cifar100_scaled_fair_reference_v2] best_ckpt_template=$BEST_CKPT_TEMPLATE"

run_logged_job \
  "classification_cifar100_scaled_fair_reference_v2/$REFERENCE_MODE" \
  "$LOG_DIR/${REFERENCE_MODE}.log" \
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    DEVICE="$DEVICE" \
    METHODS="$METHODS" \
    DATASETS="cifar100" \
    INDEPENDENT_MODELS="$INDEPENDENT_MODELS" \
    MODEL_PAIRS="$MODEL_PAIRS" \
    REQUIRE_DISTINCT_PEER="0" \
    SEEDS="$SEEDS" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    DOWNLOAD="$DOWNLOAD" \
    OUTPUT_DIR="$OUTPUT_ROOT" \
    PROTOCOL_ID="$PROTOCOL_ID" \
    HARDWARE_PROFILE="$HARDWARE_PROFILE" \
    CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
    DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE" \
    LAMBDA_IMITATION="$LAMBDA_IMITATION" \
    MARGIN="$MARGIN" \
    INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
    PEER_INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
    WARMUP_EPOCHS="0" \
    IMITATION_DECAY_START_EPOCH="-1" \
    IMITATION_DECAY_END_EPOCH="-1" \
    IMITATION_DECAY_MIN_SCALE="1.0" \
    bash scripts/paper_rerun/run_core_classification.sh

echo "[classification_cifar100_scaled_fair_reference_v2] done"
