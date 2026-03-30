#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_early_best_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-huber}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_early_best_v1}"
LIVE_PLOT_INTERVAL="${LIVE_PLOT_INTERVAL:-5}"

ONE_WAY="${ONE_WAY:-1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-3}"
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-8}"
IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.0}"
SSML_GUIDANCE_MODE="${SSML_GUIDANCE_MODE:-reweight_only}"
SSML_GATE_SCORE_MODE="${SSML_GATE_SCORE_MODE:-peer_better_student_error}"
SSML_SCORE_TRANSFORM="${SSML_SCORE_TRANSFORM:-none}"
CASE_SPECS="${CASE_SPECS:-a0p5_fastdecay:0.5:0.001:0.02:0.02 a025_fastdecay:0.25:0.001:0.02:0.02}"

run_job() {
  local label="$1"
  shift
  run_logged_job \
    "etth1_early_best_v1/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      METHODS="ssml" \
      DATASETS="etth1" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="1" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      SEQ_LEN="$SEQ_LEN" \
      PRED_LENS="$PRED_LENS" \
      FEATURE_MODE="$FEATURE_MODE" \
      REGRESSION_IMITATION_LOSS="$REGRESSION_IMITATION_LOSS" \
      WARMUP_EPOCHS="$WARMUP_EPOCHS" \
      IMITATION_DECAY_START_EPOCH="$IMITATION_DECAY_START_EPOCH" \
      IMITATION_DECAY_END_EPOCH="$IMITATION_DECAY_END_EPOCH" \
      IMITATION_DECAY_MIN_SCALE="$IMITATION_DECAY_MIN_SCALE" \
      HETERO_SSML_ONE_WAY="$ONE_WAY" \
      SSML_GATE_SCORE_MODE="$SSML_GATE_SCORE_MODE" \
      SSML_SCORE_TRANSFORM="$SSML_SCORE_TRANSFORM" \
      SSML_GUIDANCE_MODE="$SSML_GUIDANCE_MODE" \
      LIVE_PLOT_INTERVAL="$LIVE_PLOT_INTERVAL" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_etth1_early_best_v1] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_early_best_v1] gpu=$GPU seeds=$SEEDS"
echo "[time_series_etth1_early_best_v1] model_pairs=$MODEL_PAIRS"
echo "[time_series_etth1_early_best_v1] epochs=$EPOCHS live_plot_interval=$LIVE_PLOT_INTERVAL"
echo "[time_series_etth1_early_best_v1] warmup=$WARMUP_EPOCHS decay=${IMITATION_DECAY_START_EPOCH}->${IMITATION_DECAY_END_EPOCH} min_scale=$IMITATION_DECAY_MIN_SCALE"
echo "[time_series_etth1_early_best_v1] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label alpha lambda_imitation margin topk_ratio <<< "$spec"
  run_job \
    "$label" \
    LAMBDA_IMITATION="$lambda_imitation" \
    MARGIN="$margin" \
    SSML_TOPK_RATIO="$topk_ratio" \
    SSML_SUPERVISED_HOTSPOT_ALPHA="$alpha" \
    OUTPUT_DIR="$OUTPUT_ROOT/$label"
done
