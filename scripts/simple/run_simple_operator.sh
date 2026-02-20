#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

METHODS="independent studygroup"
DATASETS="darcy"
SEEDS="0"
MODELS="fno deeponet"
EPOCHS="1"
BATCH_SIZE="8"
NUM_WORKERS="4"
DEVICE="cuda"
OUTPUT_DIR="results/run_simple"

REGRESSION_IMITATION_LOSS="mse"
LAMBDA_IMITATION="1.0"
MARGIN="0.0"
WARMUP_STUDYGROUP="5"
MAX_TRAIN_BATCHES="3"
MAX_VAL_BATCHES="3"

for MODEL in $MODELS; do
  for DATASET in $DATASETS; do
    for METHOD in $METHODS; do
      for SEED in $SEEDS; do
        CMD=(
          python -m runners.run_operator
          --method "$METHOD"
          --model "$MODEL"
          --dataset "$DATASET"
          --epochs "$EPOCHS"
          --batch-size "$BATCH_SIZE"
          --num-workers "$NUM_WORKERS"
          --seed "$SEED"
          --device "$DEVICE"
          --output-dir "$OUTPUT_DIR"
          --regression-imitation-loss "$REGRESSION_IMITATION_LOSS"
          --lambda-imitation "$LAMBDA_IMITATION"
          --margin "$MARGIN"
          --warmup-epochs "$WARMUP_STUDYGROUP"
          --max-train-batches "$MAX_TRAIN_BATCHES"
          --max-val-batches "$MAX_VAL_BATCHES"
          --download
        )

        "${CMD[@]}"
      done
    done
  done
done
