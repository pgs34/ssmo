#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

METHODS="${METHODS:-independent dml ssml}"
DATASETS="${DATASETS:-burgers darcy navier_stokes}"
SEEDS="${SEEDS:-0}"
if [[ -n "${MODELS:-}" && -z "${MODEL_PAIRS:-}" ]]; then
  MODEL_PAIRS=""
  for MODEL_NAME in $MODELS; do
    MODEL_PAIRS+="${MODEL_NAME}:${MODEL_NAME} "
  done
fi
MODEL_PAIRS="${MODEL_PAIRS:-fno:fno deeponet:deeponet fno:deeponet}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-results/run_simple}"

REGRESSION_IMITATION_LOSSES="${REGRESSION_IMITATION_LOSSES:-mse}"
LAMBDA_IMITATION="${LAMBDA_IMITATION:-1.0}"
MARGIN="${MARGIN:-0.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
DOWNLOAD_OPERATOR="${DOWNLOAD_OPERATOR:-0}"

for PAIR in $MODEL_PAIRS; do
  IFS=':' read -r MODEL PEER_MODEL <<< "$PAIR"
  if [[ -z "${PEER_MODEL:-}" ]]; then
    PEER_MODEL="$MODEL"
  fi

  for DATASET in $DATASETS; do
    for LOSS in $REGRESSION_IMITATION_LOSSES; do
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
            --regression-imitation-loss "$LOSS"
            --lambda-imitation "$LAMBDA_IMITATION"
            --margin "$MARGIN"
            --warmup-epochs "$WARMUP_EPOCHS"
            --download
          )

          if [[ "$METHOD" != "independent" ]]; then
            CMD+=(--peer-model "$PEER_MODEL")
          fi
          if [[ "$DOWNLOAD_OPERATOR" == "1" ]]; then
            CMD+=(--download)
          fi

          "${CMD[@]}"
        done
      done
    done
  done
done
