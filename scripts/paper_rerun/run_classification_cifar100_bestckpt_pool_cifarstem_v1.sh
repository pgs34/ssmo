#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_bestckpt_pool_cifarstem_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1536}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD="${DOWNLOAD:-0}"
POOL_ROOT="${POOL_ROOT:-results/classification_cifar100_bestckpt_pool_cifarstem_v1}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
PROTOCOL_ID="${PROTOCOL_ID:-bestckpt_pool_cifarstem_v1}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-unknown}"
LR="${LR:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"

echo "[classification_cifar100_bestckpt_pool_cifarstem_v1] gpu=$GPU seeds=$SEEDS"
echo "[classification_cifar100_bestckpt_pool_cifarstem_v1] pool_root=$POOL_ROOT"

run_logged_job \
  "classification_cifar100_bestckpt_pool_cifarstem_v1" \
  "$LOG_DIR/pool.log" \
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    DEVICE="$DEVICE" \
    METHODS="independent" \
    DATASETS="cifar100" \
    INDEPENDENT_MODELS="resnet34_cifar_gelu" \
    MODEL_PAIRS="resnet34_cifar_gelu:resnet34_cifar_gelu" \
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
    LR="$LR" \
    WEIGHT_DECAY="$WEIGHT_DECAY" \
    LR_SCHEDULER="cosine" \
    SCHEDULER_WARMUP_EPOCHS="5" \
    SCHEDULER_MIN_SCALE="0.10" \
    LABEL_SMOOTHING="0.1" \
    MODEL_EMA_DECAY="0.999" \
    GRAD_CLIP="1.0" \
    TRAIN_AUG_MODE="strong" \
    FREEZE_BN_STATS="1" \
    CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
    bash scripts/paper_rerun/run_core_classification.sh

echo "[classification_cifar100_bestckpt_pool_cifarstem_v1] done"
