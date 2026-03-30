#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_weather_ssml_rescue}"
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
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_weather_ssml_rescue_v3}"

LAMBDAS="${LAMBDAS:-0.02 0.05}"
MARGINS="${MARGINS:-0.02 0.05}"
TOPKS="${TOPKS:-0.005 0.01}"
ONE_WAYS="${ONE_WAYS:-1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-15}"
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-35}"
IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.0}"
SSML_SUPERVISED_HOTSPOT_ALPHA="${SSML_SUPERVISED_HOTSPOT_ALPHA:-0.5}"
SSML_GATE_SCORE_MODE="${SSML_GATE_SCORE_MODE:-peer_better_student_error}"
SSML_SCORE_TRANSFORM="${SSML_SCORE_TRANSFORM:-log1p}"
SSML_GUIDANCE_MODE="${SSML_GUIDANCE_MODE:-reweight_only}"

slug_float() {
  local value="${1//./p}"
  value="${value//-/m}"
  printf '%s\n' "$value"
}

run_job() {
  local label="$1"
  shift
  run_logged_job \
    "weather_rescue/$label" \
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
      SSML_SUPERVISED_HOTSPOT_ALPHA="$SSML_SUPERVISED_HOTSPOT_ALPHA" \
      SSML_GATE_SCORE_MODE="$SSML_GATE_SCORE_MODE" \
      SSML_SCORE_TRANSFORM="$SSML_SCORE_TRANSFORM" \
      SSML_GUIDANCE_MODE="$SSML_GUIDANCE_MODE" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_weather_ssml_rescue] output_root=$OUTPUT_ROOT"
echo "[time_series_weather_ssml_rescue] gpu=$GPU seeds=$SEEDS"
echo "[time_series_weather_ssml_rescue] lambdas=$LAMBDAS margins=$MARGINS topks=$TOPKS one_ways=$ONE_WAYS"
echo "[time_series_weather_ssml_rescue] ssml_supervised_hotspot_alpha=$SSML_SUPERVISED_HOTSPOT_ALPHA"
echo "[time_series_weather_ssml_rescue] ssml_gate_score_mode=$SSML_GATE_SCORE_MODE"
echo "[time_series_weather_ssml_rescue] ssml_score_transform=$SSML_SCORE_TRANSFORM"
echo "[time_series_weather_ssml_rescue] ssml_guidance_mode=$SSML_GUIDANCE_MODE"

for one_way in $ONE_WAYS; do
  for lambda_imitation in $LAMBDAS; do
    lambda_slug="$(slug_float "$lambda_imitation")"
    for margin in $MARGINS; do
      margin_slug="$(slug_float "$margin")"
      for topk_ratio in $TOPKS; do
        topk_slug="$(slug_float "$topk_ratio")"
        run_job \
          "weather_ow${one_way}_l${lambda_slug}_m${margin_slug}_t${topk_slug}" \
          HETERO_SSML_ONE_WAY="$one_way" \
          LAMBDA_IMITATION="$lambda_imitation" \
          MARGIN="$margin" \
          SSML_TOPK_RATIO="$topk_ratio" \
          OUTPUT_DIR="$OUTPUT_ROOT/ow${one_way}/ssml_l${lambda_slug}_m${margin_slug}_t${topk_slug}"
      done
    done
  done
done
