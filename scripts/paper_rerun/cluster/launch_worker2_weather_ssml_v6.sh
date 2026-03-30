#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker2}"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
GPU="${TIME_SERIES_GPU:-0}" \
OUTPUT_ROOT="${WEATHER_V6_OUTPUT_ROOT:-results/time_series_weather_ssml_rescue_v6}" \
SEEDS="${WEATHER_V6_SEEDS:-0 1 2}" \
CASE_SPECS="${WEATHER_V6_CASE_SPECS:-a025_t10:0.25:0.0:0.02:0.01 a05_t10:0.5:0.0:0.02:0.01 a025_t15:0.25:0.0:0.02:0.015 a05_t15:0.5:0.0:0.02:0.015}" \
MODEL_PAIRS="${WEATHER_V6_MODEL_PAIRS:-transformer:dlinear dlinear:transformer}" \
REGRESSION_IMITATION_LOSS="${WEATHER_V6_REGRESSION_IMITATION_LOSS:-mse}" \
ONE_WAY="${WEATHER_V6_ONE_WAY:-1}" \
WARMUP_EPOCHS="${WEATHER_V6_WARMUP_EPOCHS:-10}" \
IMITATION_DECAY_START_EPOCH="${WEATHER_V6_DECAY_START_EPOCH:-20}" \
IMITATION_DECAY_END_EPOCH="${WEATHER_V6_DECAY_END_EPOCH:-45}" \
IMITATION_DECAY_MIN_SCALE="${WEATHER_V6_DECAY_MIN_SCALE:-0.0}" \
SSML_GUIDANCE_MODE="${WEATHER_V6_SSML_GUIDANCE_MODE:-reweight_only}" \
SSML_GATE_SCORE_MODE="${WEATHER_V6_SSML_GATE_SCORE_MODE:-peer_better_student_error}" \
SSML_SCORE_TRANSFORM="${WEATHER_V6_SSML_SCORE_TRANSFORM:-log1p}" \
run_logged_job \
  "worker2/weather_ssml_v6" \
  "$LOG_DIR/weather_ssml_v6_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_time_series_weather_ssml_v6.sh

echo "[worker2_weather_ssml_v6] job finished"
