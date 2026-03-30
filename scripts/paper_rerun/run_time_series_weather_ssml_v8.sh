#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_weather_ssml_v8}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear dlinear:transformer}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-mse}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_weather_ssml_rescue_v8}"

ONE_WAY="${ONE_WAY:-1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-20}"
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-45}"
IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.0}"
SSML_GUIDANCE_MODE="${SSML_GUIDANCE_MODE:-reweight_only}"
SSML_GATE_SCORE_MODE="${SSML_GATE_SCORE_MODE:-peer_better_student_error}"
SSML_SCORE_TRANSFORM="${SSML_SCORE_TRANSFORM:-log1p}"
SSML_TOPK_SCOPE="${SSML_TOPK_SCOPE:-positive}"
SSML_SUPERVISED_WEIGHT_MODE="${SSML_SUPERVISED_WEIGHT_MODE:-binary}"
CASE_SPECS="${CASE_SPECS:-bin_trim90_a01_t05:0.1:0.0:0.02:0.05:0.90 bin_trim90_a025_t03:0.25:0.0:0.02:0.03:0.90 bin_trim85_a01_t05:0.1:0.0:0.02:0.05:0.85}"

run_job() {
  local label="$1"
  shift
  run_logged_job \
    "weather_v8/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      METHODS="ssml" \
      DATASETS="weather" \
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
      SSML_TOPK_SCOPE="$SSML_TOPK_SCOPE" \
      SSML_SUPERVISED_WEIGHT_MODE="$SSML_SUPERVISED_WEIGHT_MODE" \
      SSML_GUIDANCE_MODE="$SSML_GUIDANCE_MODE" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_weather_ssml_v8] output_root=$OUTPUT_ROOT"
echo "[time_series_weather_ssml_v8] gpu=$GPU seeds=$SEEDS"
echo "[time_series_weather_ssml_v8] model_pairs=$MODEL_PAIRS"
echo "[time_series_weather_ssml_v8] warmup=$WARMUP_EPOCHS decay=${IMITATION_DECAY_START_EPOCH}->${IMITATION_DECAY_END_EPOCH} min_scale=$IMITATION_DECAY_MIN_SCALE"
echo "[time_series_weather_ssml_v8] one_way=$ONE_WAY regression_imitation_loss=$REGRESSION_IMITATION_LOSS"
echo "[time_series_weather_ssml_v8] gate=$SSML_GATE_SCORE_MODE transform=$SSML_SCORE_TRANSFORM topk_scope=$SSML_TOPK_SCOPE weight_mode=$SSML_SUPERVISED_WEIGHT_MODE guidance=$SSML_GUIDANCE_MODE"
echo "[time_series_weather_ssml_v8] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label alpha lambda_imitation margin topk_ratio upper_q <<< "$spec"
  run_job \
    "$label" \
    LAMBDA_IMITATION="$lambda_imitation" \
    MARGIN="$margin" \
    SSML_TOPK_RATIO="$topk_ratio" \
    SSML_SUPERVISED_HOTSPOT_ALPHA="$alpha" \
    SSML_POSITIVE_UPPER_QUANTILE="$upper_q" \
    OUTPUT_DIR="$OUTPUT_ROOT/$label"
done
