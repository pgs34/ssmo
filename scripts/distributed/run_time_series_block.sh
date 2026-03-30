#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: bash scripts/distributed/run_time_series_block.sh <gpu_id> <datasets> <model_pairs>" >&2
  echo "example: bash scripts/distributed/run_time_series_block.sh 0 'etth1' 'transformer:dlinear'" >&2
  exit 1
fi

GPU_ID="$1"
DATASETS_ARG="$2"
MODEL_PAIRS_ARG="$3"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

METHODS="${METHODS:-independent dml ssml}"
PRED_LENS="${PRED_LENS:-24 96}"
REGRESSION_IMITATION_LOSSES="${REGRESSION_IMITATION_LOSSES:-mse}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEEDS="${SEEDS:-0}"
DEVICE="${DEVICE:-cuda}"
SEQ_LEN="${SEQ_LEN:-96}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
TARGET_COLUMN="${TARGET_COLUMN:-}"
LAMBDA_IMITATION="${LAMBDA_IMITATION:-1.0}"
MARGIN="${MARGIN:-0.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/time_series_method_diff}"

echo "[DIST] gpu=$GPU_ID datasets=$DATASETS_ARG model_pairs=$MODEL_PAIRS_ARG output_dir=$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
METHODS="$METHODS" \
DATASETS="$DATASETS_ARG" \
MODEL_PAIRS="$MODEL_PAIRS_ARG" \
PRED_LENS="$PRED_LENS" \
REGRESSION_IMITATION_LOSSES="$REGRESSION_IMITATION_LOSSES" \
EPOCHS="$EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
NUM_WORKERS="$NUM_WORKERS" \
SEEDS="$SEEDS" \
DEVICE="$DEVICE" \
SEQ_LEN="$SEQ_LEN" \
FEATURE_MODE="$FEATURE_MODE" \
TARGET_COLUMN="$TARGET_COLUMN" \
LAMBDA_IMITATION="$LAMBDA_IMITATION" \
MARGIN="$MARGIN" \
WARMUP_EPOCHS="$WARMUP_EPOCHS" \
OUTPUT_DIR="$OUTPUT_DIR" \
RUN_VISUALIZATION="0" \
bash scripts/simple/run_simple_time_series.sh
