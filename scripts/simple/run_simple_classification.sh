#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

METHODS="${METHODS:-independent dml ssml}"
DATASETS="${DATASETS:-cifar10 cifar100}"
SEEDS="${SEEDS:-0}"
if [[ -n "${MODELS:-}" && -z "${MODEL_PAIRS:-}" ]]; then
  MODEL_PAIRS=""
  for MODEL_NAME in $MODELS; do
    MODEL_PAIRS+="${MODEL_NAME}:${MODEL_NAME} "
  done
fi
MODEL_PAIRS="${MODEL_PAIRS:-resnet18:resnet18 resnet18:vit_b16}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-results/run_simple}"
CLASSIFICATION_IMITATION_LOSSES="${CLASSIFICATION_IMITATION_LOSSES:-kl}"
LAMBDA_IMITATION="${LAMBDA_IMITATION:-1.0}"
MARGIN="${MARGIN:-0.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
LABEL_NOISE_CONDITIONS="${LABEL_NOISE_CONDITIONS:-none:0.0 symmetric:0.2 symmetric:0.4 symmetric:0.6 asymmetric:0.2 asymmetric:0.4 asymmetric:0.6}"
DOWNLOAD_CLASSIFICATION="${DOWNLOAD_CLASSIFICATION:-1}"
TRAIN_SUBSET_SIZE="${TRAIN_SUBSET_SIZE:-}"
VAL_SUBSET_SIZE="${VAL_SUBSET_SIZE:-}"

for PAIR in $MODEL_PAIRS; do
  IFS=':' read -r MODEL PEER_MODEL <<< "$PAIR"
  if [[ -z "${PEER_MODEL:-}" ]]; then
    PEER_MODEL="$MODEL"
  fi

  for DATASET in $DATASETS; do
    for NOISE_CONDITION in $LABEL_NOISE_CONDITIONS; do
      IFS=':' read -r NOISE_TYPE NOISE_RATE <<< "$NOISE_CONDITION"
      if [[ -z "${NOISE_RATE:-}" ]]; then
        NOISE_RATE="0.0"
      fi

      for LOSS in $CLASSIFICATION_IMITATION_LOSSES; do
        for METHOD in $METHODS; do
          for SEED in $SEEDS; do
            CMD=(
              python -m runners.run_classification
              --method "$METHOD"
              --model "$MODEL"
              --dataset "$DATASET"
              --epochs "$EPOCHS"
              --batch-size "$BATCH_SIZE"
              --num-workers "$NUM_WORKERS"
              --seed "$SEED"
              --device "$DEVICE"
              --output-dir "$OUTPUT_DIR"
              --classification-imitation-loss "$LOSS"
              --lambda-imitation "$LAMBDA_IMITATION"
              --margin "$MARGIN"
              --warmup-epochs "$WARMUP_EPOCHS"
            )

            if [[ "$METHOD" != "independent" ]]; then
              CMD+=(--peer-model "$PEER_MODEL")
            fi
            if [[ "$NOISE_TYPE" != "none" && "$NOISE_RATE" != "0" && "$NOISE_RATE" != "0.0" ]]; then
              CMD+=(--label-noise-type "$NOISE_TYPE" --label-noise-rate "$NOISE_RATE")
            fi
            if [[ -n "$TRAIN_SUBSET_SIZE" ]]; then
              CMD+=(--train-subset-size "$TRAIN_SUBSET_SIZE")
            fi
            if [[ -n "$VAL_SUBSET_SIZE" ]]; then
              CMD+=(--val-subset-size "$VAL_SUBSET_SIZE")
            fi
            if [[ "$DOWNLOAD_CLASSIFICATION" == "1" ]]; then
              CMD+=(--download)
            fi

            "${CMD[@]}"
          done
        done
      done
    done
  done
done
