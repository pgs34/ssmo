#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

METHODS="naive dml studygroup"
DATASETS="voc"
SEEDS="0"
MODELS="unet deeplabv3_resnet50"
EPOCHS="300"
BATCH_SIZE="2"
NUM_WORKERS="4"
DEVICE="cuda"
OUTPUT_DIR="results/run_simple"

HEIGHT="512"
WIDTH="512"
SEGMENTATION_IMITATION_LOSSES="kl mse_logits js"
LAMBDA_IMITATION="1.0"
MARGIN="0.0"
WARMUP_STUDYGROUP="5"
VAL_DATASETS="voc"

for MODEL in $MODELS; do
  for DATASET in $DATASETS; do
    for LOSS in $SEGMENTATION_IMITATION_LOSSES; do
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
              --segmentation-imitation-loss "$LOSS"
              --lambda-imitation "$LAMBDA_IMITATION"
              --margin "$MARGIN"
              --warmup-epochs "$WARMUP_STUDYGROUP"
            )

            "${CMD[@]}"
          done
        done
      done
    done
  done
done
