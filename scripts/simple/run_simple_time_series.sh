#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

METHODS="${METHODS:-independent dml ssml}"
DATASETS="${DATASETS:-etth1 electricity weather}"
SEEDS="${SEEDS:-0}"
if [[ -n "${MODELS:-}" && -z "${MODEL_PAIRS:-}" ]]; then
  MODEL_PAIRS=""
  for MODEL_NAME in $MODELS; do
    MODEL_PAIRS+="${MODEL_NAME}:${MODEL_NAME} "
  done
fi
MODEL_PAIRS="${MODEL_PAIRS:-transformer:transformer dlinear:dlinear transformer:dlinear}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-results/run_simple}"
RUN_VISUALIZATION="${RUN_VISUALIZATION:-1}"

SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24 96}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
TARGET_COLUMN="${TARGET_COLUMN:-}"
REGRESSION_IMITATION_LOSSES="${REGRESSION_IMITATION_LOSSES:-mse}"
LAMBDA_IMITATION="${LAMBDA_IMITATION:-1.0}"
MARGIN="${MARGIN:-0.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"

for PAIR in $MODEL_PAIRS; do
  IFS=':' read -r MODEL PEER_MODEL <<< "$PAIR"
  if [[ -z "${PEER_MODEL:-}" ]]; then
    PEER_MODEL="$MODEL"
  fi

  for DATASET in $DATASETS; do
    for PRED_LEN in $PRED_LENS; do
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
              --warmup-epochs "$WARMUP_EPOCHS"
              --feature-mode "$FEATURE_MODE"
            )

            if [[ "$METHOD" != "independent" ]]; then
              CMD+=(--peer-model "$PEER_MODEL")
            fi
            if [[ -n "$TARGET_COLUMN" ]]; then
              CMD+=(--target-column "$TARGET_COLUMN")
            fi

            "${CMD[@]}"
          done
        done
      done
    done
  done
done

if [[ "$RUN_VISUALIZATION" == "1" && -d "$OUTPUT_DIR/time_series" ]]; then
  python -m src.utils.visualization \
    --input-dir "$OUTPUT_DIR/time_series" \
    --output-dir "$OUTPUT_DIR"
fi
