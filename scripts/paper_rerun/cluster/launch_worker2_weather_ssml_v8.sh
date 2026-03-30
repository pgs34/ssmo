#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker2}"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
GPU="${TIME_SERIES_GPU:-0}" \
OUTPUT_ROOT="${WEATHER_V8_OUTPUT_ROOT:-results/time_series_weather_ssml_rescue_v8}" \
SEEDS="${WEATHER_V8_SEEDS:-0 1 2}" \
CASE_SPECS="${WEATHER_V8_CASE_SPECS:-bin_trim90_a01_t05:0.1:0.0:0.02:0.05:0.90 bin_trim90_a025_t03:0.25:0.0:0.02:0.03:0.90 bin_trim85_a01_t05:0.1:0.0:0.02:0.05:0.85}" \
MODEL_PAIRS="${WEATHER_V8_MODEL_PAIRS:-transformer:dlinear dlinear:transformer}" \
REGRESSION_IMITATION_LOSS="${WEATHER_V8_REGRESSION_IMITATION_LOSS:-mse}" \
ONE_WAY="${WEATHER_V8_ONE_WAY:-1}" \
WARMUP_EPOCHS="${WEATHER_V8_WARMUP_EPOCHS:-10}" \
IMITATION_DECAY_START_EPOCH="${WEATHER_V8_DECAY_START_EPOCH:-20}" \
IMITATION_DECAY_END_EPOCH="${WEATHER_V8_DECAY_END_EPOCH:-45}" \
IMITATION_DECAY_MIN_SCALE="${WEATHER_V8_DECAY_MIN_SCALE:-0.0}" \
SSML_GUIDANCE_MODE="${WEATHER_V8_SSML_GUIDANCE_MODE:-reweight_only}" \
SSML_GATE_SCORE_MODE="${WEATHER_V8_SSML_GATE_SCORE_MODE:-peer_better_student_error}" \
SSML_SCORE_TRANSFORM="${WEATHER_V8_SSML_SCORE_TRANSFORM:-log1p}" \
SSML_TOPK_SCOPE="${WEATHER_V8_SSML_TOPK_SCOPE:-positive}" \
SSML_SUPERVISED_WEIGHT_MODE="${WEATHER_V8_SSML_SUPERVISED_WEIGHT_MODE:-binary}" \
run_logged_job \
  "worker2/weather_ssml_v8" \
  "$LOG_DIR/weather_ssml_v8_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_time_series_weather_ssml_v8.sh

echo "[worker2_weather_ssml_v8] job finished"
