#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_bestckpt_pool_v2}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD="${DOWNLOAD:-0}"
POOL_ROOT="${POOL_ROOT:-results/classification_cifar100_bestckpt_pool_v2}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
INDEPENDENT_MODELS="${INDEPENDENT_MODELS:-resnet34_gelu resnet34_cifar_gelu}"
PROTOCOL_ID="${PROTOCOL_ID:-bestckpt_pool_v2}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-unknown}"

echo "[classification_cifar100_bestckpt_pool_v2] gpu=$GPU seeds=$SEEDS"
echo "[classification_cifar100_bestckpt_pool_v2] independent_models=$INDEPENDENT_MODELS"
echo "[classification_cifar100_bestckpt_pool_v2] pool_root=$POOL_ROOT"

run_logged_job \
  "classification_cifar100_bestckpt_pool_v2" \
  "$LOG_DIR/pool.log" \
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    DEVICE="$DEVICE" \
    METHODS="independent" \
    DATASETS="cifar100" \
    INDEPENDENT_MODELS="$INDEPENDENT_MODELS" \
    MODEL_PAIRS="resnet34_gelu:resnet34_gelu" \
    REQUIRE_DISTINCT_PEER="0" \
    SEEDS="$SEEDS" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    DOWNLOAD="$DOWNLOAD" \
    OUTPUT_DIR="$POOL_ROOT/classification" \
    PROTOCOL_ID="$PROTOCOL_ID" \
    HARDWARE_PROFILE="$HARDWARE_PROFILE" \
    OPTIMIZER="sgd_nesterov" \
    MOMENTUM="0.9" \
    LR_SCHEDULER="cosine" \
    SCHEDULER_WARMUP_EPOCHS="5" \
    SCHEDULER_MIN_SCALE="0.01" \
    LABEL_SMOOTHING="0.1" \
    MODEL_EMA_DECAY="0.999" \
    TRAIN_AUG_MODE="strong" \
    LR="${LR:-0.1}" \
    WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}" \
    CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
    bash scripts/paper_rerun/run_core_classification.sh

echo "[classification_cifar100_bestckpt_pool_v2] done"
