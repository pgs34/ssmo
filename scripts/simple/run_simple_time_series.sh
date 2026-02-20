#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

METHODS="independent studygroup"
DATASETS="etth1"
SEEDS="0"
MODELS="dlinear transformer"
EPOCHS="1"
BATCH_SIZE="32"
NUM_WORKERS="4"
DEVICE="cuda"
OUTPUT_DIR="results/run_simple"

SEQ_LEN="96"
PRED_LEN="24"
FEATURE_MODE="multivariate"
REGRESSION_IMITATION_LOSS="mse"
LAMBDA_IMITATION="1.0"
MARGIN="0.0"
WARMUP_STUDYGROUP="3"
MAX_TRAIN_BATCHES="5"
MAX_VAL_BATCHES="5"

for MODEL in $MODELS; do
  for DATASET in $DATASETS; do
    for METHOD in $METHODS; do
      for SEED in $SEEDS; do
        CMD=(
          python -m runners.run_time_series
          --method "$METHOD"
          --model "$MODEL"
          --dataset "$DATASET"
          --epochs "$EPOCHS"
          --batch-size "$BATCH_SIZE"
          --num-workers "$NUM_WORKERS"
          --seed "$SEED"
          --device "$DEVICE"
          --output-dir "$OUTPUT_DIR"
          --seq-len "$SEQ_LEN"
          --pred-len "$PRED_LEN"
          --regression-imitation-loss "$REGRESSION_IMITATION_LOSS"
          --lambda-imitation "$LAMBDA_IMITATION"
          --margin "$MARGIN"
          --warmup-epochs "$WARMUP_STUDYGROUP"
          --feature-mode "$FEATURE_MODE"
          --max-train-batches "$MAX_TRAIN_BATCHES"
          --max-val-batches "$MAX_VAL_BATCHES"
        )

        "${CMD[@]}"
      done
    done
  done
done
