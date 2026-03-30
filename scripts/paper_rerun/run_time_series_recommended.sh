#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

METHODS="${METHODS:-independent dml ssml}"
DATASETS="${DATASETS:-etth1 electricity weather}"
SEEDS="${SEEDS:-0 1 2}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
INDEPENDENT_MODELS="${INDEPENDENT_MODELS:-$(collect_unique_models "$MODEL_PAIRS")}"
REQUIRE_DISTINCT_PEER="${REQUIRE_DISTINCT_PEER:-1}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-results/time_series_ssml_topk_v1}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-mse}"
LAMBDA_IMITATION="${LAMBDA_IMITATION:-0.3}"
MARGIN="${MARGIN:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
SSML_TOPK_RATIO="${SSML_TOPK_RATIO:-0.3}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"

echo "[time_series_recommended] output_dir=$OUTPUT_DIR"
echo "[time_series_recommended] methods=$METHODS"
echo "[time_series_recommended] datasets=$DATASETS"
echo "[time_series_recommended] model_pairs=$MODEL_PAIRS"
echo "[time_series_recommended] lambda=$LAMBDA_IMITATION margin=$MARGIN warmup=$WARMUP_EPOCHS topk=$SSML_TOPK_RATIO"

METHODS="$METHODS" \
DATASETS="$DATASETS" \
SEEDS="$SEEDS" \
MODEL_PAIRS="$MODEL_PAIRS" \
INDEPENDENT_MODELS="$INDEPENDENT_MODELS" \
REQUIRE_DISTINCT_PEER="$REQUIRE_DISTINCT_PEER" \
EPOCHS="$EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
NUM_WORKERS="$NUM_WORKERS" \
DEVICE="$DEVICE" \
OUTPUT_DIR="$OUTPUT_DIR" \
SEQ_LEN="$SEQ_LEN" \
PRED_LENS="$PRED_LENS" \
REGRESSION_IMITATION_LOSS="$REGRESSION_IMITATION_LOSS" \
LAMBDA_IMITATION="$LAMBDA_IMITATION" \
MARGIN="$MARGIN" \
WARMUP_EPOCHS="$WARMUP_EPOCHS" \
SSML_TOPK_RATIO="$SSML_TOPK_RATIO" \
FEATURE_MODE="$FEATURE_MODE" \
bash scripts/paper_rerun/run_core_time_series.sh
