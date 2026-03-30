#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker2}"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
GPU="${TIME_SERIES_GPU:-0}" \
OUTPUT_ROOT="${WEATHER_V9_OUTPUT_ROOT:-results/time_series_weather_ssml_rescue_v9}" \
SEEDS="${WEATHER_V9_SEEDS:-0 1 2}" \
CASE_SPECS="${WEATHER_V9_CASE_SPECS:-relg_a01_t03:0.1:0.0:0.05:0.03:0.90 relg_a02_t03:0.2:0.0:0.05:0.03:0.90 relg_a01_t05:0.1:0.0:0.05:0.05:0.90}" \
MODEL_PAIRS="${WEATHER_V9_MODEL_PAIRS:-transformer:dlinear}" \
REGRESSION_IMITATION_LOSS="${WEATHER_V9_REGRESSION_IMITATION_LOSS:-mse}" \
ONE_WAY="${WEATHER_V9_ONE_WAY:-1}" \
WARMUP_EPOCHS="${WEATHER_V9_WARMUP_EPOCHS:-5}" \
IMITATION_DECAY_START_EPOCH="${WEATHER_V9_DECAY_START_EPOCH:-15}" \
IMITATION_DECAY_END_EPOCH="${WEATHER_V9_DECAY_END_EPOCH:-35}" \
IMITATION_DECAY_MIN_SCALE="${WEATHER_V9_DECAY_MIN_SCALE:-0.0}" \
SSML_GUIDANCE_MODE="${WEATHER_V9_SSML_GUIDANCE_MODE:-reweight_only}" \
SSML_GATE_SCORE_MODE="${WEATHER_V9_SSML_GATE_SCORE_MODE:-peer_better_student_error_relgain}" \
SSML_SCORE_TRANSFORM="${WEATHER_V9_SSML_SCORE_TRANSFORM:-log1p}" \
SSML_TOPK_SCOPE="${WEATHER_V9_SSML_TOPK_SCOPE:-positive}" \
SSML_SUPERVISED_WEIGHT_MODE="${WEATHER_V9_SSML_SUPERVISED_WEIGHT_MODE:-binary}" \
run_logged_job \
  "worker2/weather_ssml_v9" \
  "$LOG_DIR/weather_ssml_v9_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_time_series_weather_ssml_v9.sh

echo "[worker2_weather_ssml_v9] job finished"
