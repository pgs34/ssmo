#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

METHODS="${METHODS:-naive dml studygroup}"
SEEDS="${SEEDS:-0}"
if [[ -n "${MODELS:-}" && -z "${MODEL_PAIRS:-}" ]]; then
  MODEL_PAIRS=""
  for MODEL_NAME in $MODELS; do
    MODEL_PAIRS+="${MODEL_NAME}:${MODEL_NAME} "
  done
fi
MODEL_PAIRS="${MODEL_PAIRS:-unet:unet deeplabv3_resnet50:deeplabv3_resnet50 unet:deeplabv3_resnet50}"
if [[ -z "${DATASET_PAIRS:-}" && -n "${DATASETS:-}" ]]; then
  DATASET_PAIRS=""
  if [[ -n "${VAL_DATASETS:-}" ]]; then
    for TRAIN_DATASET_NAME in $DATASETS; do
      for VAL_DATASET_NAME in $VAL_DATASETS; do
        DATASET_PAIRS+="${TRAIN_DATASET_NAME}:${VAL_DATASET_NAME} "
      done
    done
  else
    for TRAIN_DATASET_NAME in $DATASETS; do
      DATASET_PAIRS+="${TRAIN_DATASET_NAME}:${TRAIN_DATASET_NAME} "
    done
  fi
fi
DATASET_PAIRS="${DATASET_PAIRS:-voc:voc}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-results/run_simple}"

HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
SEGMENTATION_IMITATION_LOSSES="${SEGMENTATION_IMITATION_LOSSES:-kl}"
LAMBDA_IMITATION="${LAMBDA_IMITATION:-1.0}"
MARGIN="${MARGIN:-0.0}"
WARMUP_STUDYGROUP="${WARMUP_STUDYGROUP:-5}"

TRAIN_CORRUPTION="${TRAIN_CORRUPTION:-none}"
TRAIN_CORRUPTION_SEVERITY="${TRAIN_CORRUPTION_SEVERITY:-1}"
TRAIN_RESOLUTION_SCALE="${TRAIN_RESOLUTION_SCALE:-1.0}"
VAL_CONDITIONS="${VAL_CONDITIONS:-none:1:1.0 gaussian_noise:3:1.0 none:1:0.75}"
DOWNLOAD_VOC="${DOWNLOAD_VOC:-1}"

for PAIR in $MODEL_PAIRS; do
  IFS=':' read -r MODEL PEER_MODEL <<< "$PAIR"
  if [[ -z "${PEER_MODEL:-}" ]]; then
    PEER_MODEL="$MODEL"
  fi

  for DATASET_PAIR in $DATASET_PAIRS; do
    IFS=':' read -r TRAIN_DATASET VAL_DATASET <<< "$DATASET_PAIR"
    if [[ -z "${VAL_DATASET:-}" ]]; then
      VAL_DATASET="$TRAIN_DATASET"
    fi

    for VAL_CONDITION in $VAL_CONDITIONS; do
      IFS=':' read -r VAL_CORRUPTION VAL_CORRUPTION_SEVERITY VAL_RESOLUTION_SCALE <<< "$VAL_CONDITION"
      if [[ -z "${VAL_CORRUPTION_SEVERITY:-}" ]]; then
        VAL_CORRUPTION_SEVERITY="1"
      fi
      if [[ -z "${VAL_RESOLUTION_SCALE:-}" ]]; then
        VAL_RESOLUTION_SCALE="1.0"
      fi

      for LOSS in $SEGMENTATION_IMITATION_LOSSES; do
        for METHOD in $METHODS; do
          for SEED in $SEEDS; do
            CMD=(
              python -m runners.run_segmentation
              --method "$METHOD"
              --model "$MODEL"
              --train-dataset "$TRAIN_DATASET"
              --val-dataset "$VAL_DATASET"
              --epochs "$EPOCHS"
              --batch-size "$BATCH_SIZE"
              --num-workers "$NUM_WORKERS"
              --seed "$SEED"
              --device "$DEVICE"
              --output-dir "$OUTPUT_DIR"
              --height "$HEIGHT"
              --width "$WIDTH"
              --segmentation-imitation-loss "$LOSS"
              --lambda-imitation "$LAMBDA_IMITATION"
              --margin "$MARGIN"
              --warmup-epochs "$WARMUP_STUDYGROUP"
              --train-corruption "$TRAIN_CORRUPTION"
              --train-corruption-severity "$TRAIN_CORRUPTION_SEVERITY"
              --train-resolution-scale "$TRAIN_RESOLUTION_SCALE"
              --val-corruption "$VAL_CORRUPTION"
              --val-corruption-severity "$VAL_CORRUPTION_SEVERITY"
              --val-resolution-scale "$VAL_RESOLUTION_SCALE"
            )

            if [[ "$METHOD" != "naive" ]]; then
              CMD+=(--peer-model "$PEER_MODEL")
            fi
            if [[ "$DOWNLOAD_VOC" == "1" ]]; then
              CMD+=(--download-voc)
            fi

            "${CMD[@]}"
          done
        done
      done
    done
  done
done
