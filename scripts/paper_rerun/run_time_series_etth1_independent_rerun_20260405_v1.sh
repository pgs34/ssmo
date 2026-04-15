#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_independent_rerun_20260405_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_independent_rerun_20260405_v1}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-huber}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
INDEPENDENT_MODELS="${INDEPENDENT_MODELS:-transformer dlinear}"

echo "[time_series_etth1_independent_rerun_20260405_v1] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_independent_rerun_20260405_v1] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS"
echo "[time_series_etth1_independent_rerun_20260405_v1] independent_models=$INDEPENDENT_MODELS"

run_logged_job \
  "time_series_etth1_independent_rerun_20260405_v1" \
  "$LOG_DIR/launcher.out" \
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    DEVICE="$DEVICE" \
    OUTPUT_DIR="$OUTPUT_ROOT" \
    DATASETS="etth1" \
    METHODS="independent" \
    MODEL_PAIRS="$MODEL_PAIRS" \
    INDEPENDENT_MODELS="$INDEPENDENT_MODELS" \
    REQUIRE_DISTINCT_PEER="1" \
    SEEDS="$SEEDS" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    SEQ_LEN="$SEQ_LEN" \
    PRED_LENS="$PRED_LENS" \
    REGRESSION_IMITATION_LOSS="$REGRESSION_IMITATION_LOSS" \
    FEATURE_MODE="$FEATURE_MODE" \
    LIVE_PLOT_INTERVAL="10" \
    LAMBDA_IMITATION="0.001" \
    MARGIN="0.02" \
    WARMUP_EPOCHS="8" \
    IMITATION_DECAY_START_EPOCH="20" \
    IMITATION_DECAY_END_EPOCH="70" \
    IMITATION_DECAY_MIN_SCALE="0.15" \
    SSML_TOPK_RATIO="0.02" \
    SSML_SUPERVISED_HOTSPOT_ALPHA="0.5" \
    SSML_GATE_SCORE_MODE="peer_better_student_error" \
    SSML_SCORE_TRANSFORM="none" \
    SSML_GUIDANCE_MODE="reweight_only" \
    HETERO_SSML_ONE_WAY="1" \
    EARLY_STOP_PATIENCE="0" \
    EARLY_STOP_MIN_EPOCHS="0" \
    EARLY_STOP_MIN_DELTA="0.0" \
    bash scripts/paper_rerun/run_core_time_series.sh

echo "[time_series_etth1_independent_rerun_20260405_v1] done"
