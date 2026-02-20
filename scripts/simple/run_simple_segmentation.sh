#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

METHODS="independent studygroup"
DATASETS="voc"
SEEDS="0"
MODELS="unet deeplabv3_resnet50"
EPOCHS="1"
BATCH_SIZE="2"
NUM_WORKERS="4"
DEVICE="cuda"
OUTPUT_DIR="results/run_simple"

HEIGHT="512"
WIDTH="512"
SEGMENTATION_IMITATION_LOSS="kl"
LAMBDA_IMITATION="1.0"
MARGIN="0.0"
WARMUP_STUDYGROUP="5"
MAX_TRAIN_BATCHES="2"
MAX_VAL_BATCHES="2"
VAL_DATASETS="voc"

for MODEL in $MODELS; do
  for DATASET in $DATASETS; do
    for METHOD in $METHODS; do
      for SEED in $SEEDS; do
        for VAL_DATASET in $VAL_DATASETS; do
          CMD=(
            python -m runners.run_segmentation
            --method "$METHOD"
            --model "$MODEL"
            --train-dataset "$DATASET"
            --val-dataset "$VAL_DATASET"
            --epochs "$EPOCHS"
            --batch-size "$BATCH_SIZE"
            --num-workers "$NUM_WORKERS"
            --seed "$SEED"
            --device "$DEVICE"
            --output-dir "$OUTPUT_DIR"
            --height "$HEIGHT"
            --width "$WIDTH"
            --segmentation-imitation-loss "$SEGMENTATION_IMITATION_LOSS"
            --lambda-imitation "$LAMBDA_IMITATION"
            --margin "$MARGIN"
            --warmup-epochs "$WARMUP_STUDYGROUP"
            --max-train-batches "$MAX_TRAIN_BATCHES"
            --max-val-batches "$MAX_VAL_BATCHES"
            --download-voc
          )

          "${CMD[@]}"
        done
      done
    done
  done
done
