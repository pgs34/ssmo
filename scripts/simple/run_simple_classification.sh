#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

METHODS="naive dml studygroup"
DATASETS="mnist cifar10 cifar100"
SEEDS="0"
MODELS="simple_cnn simple_mlp resnet18 vit_b16"
EPOCHS="200"
BATCH_SIZE="64"
NUM_WORKERS="4"
DEVICE="cuda"
OUTPUT_DIR="results/run_simple"
CLASSIFICATION_IMITATION_LOSSES="kl mse_logits"
LAMBDA_IMITATION="1.0"
MARGIN="0.0"
WARMUP_STUDYGROUP="5"

for MODEL in $MODELS; do
  for DATASET in $DATASETS; do
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
            --warmup-epochs "$WARMUP_STUDYGROUP"
          )

          "${CMD[@]}"
        done
      done
    done
  done
done
