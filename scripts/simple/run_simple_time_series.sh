#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

METHODS="naive dml studygroup"
DATASETS="etth1"
SEEDS="0"
  ="dlinear transformer"
EPOCHS="500"
BATCH_SIZE="32"
NUM_WORKERS="4"
DEVICE="cuda"
OUTPUT_DIR="results/run_simple"

SEQ_LEN="96"
PRED_LEN="24"
FEATURE_MODE="multivariate"
REGRESSION_IMITATION_LOSSES="mse mae huber"
LAMBDA_IMITATION="1.0"
MARGIN="0.0"
WARMUP_STUDYGROUP="3"

for MODEL in $MODELS; do
  for DATASET in $DATASETS; do
    for LOSS in $REGRESSION_IMITATION_LOSSES; do
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
            --regression-imitation-loss "$LOSS"
            --lambda-imitation "$LAMBDA_IMITATION"
            --margin "$MARGIN"
            --warmup-epochs "$WARMUP_STUDYGROUP"
            --feature-mode "$FEATURE_MODE"
          )

          "${CMD[@]}"
        done
      done
    done
  done
done
