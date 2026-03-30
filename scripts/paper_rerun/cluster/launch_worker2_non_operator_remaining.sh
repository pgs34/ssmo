#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker2}"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
GPU="${TIME_SERIES_GPU:-0}" \
OUTPUT_ROOT="${WEATHER_OUTPUT_ROOT:-results/time_series_weather_ssml_rescue_v3}" \
LAMBDAS="${WEATHER_LAMBDAS:-0.02 0.05}" \
MARGINS="${WEATHER_MARGINS:-0.02 0.05}" \
TOPKS="${WEATHER_TOPKS:-0.005 0.01}" \
ONE_WAYS="${WEATHER_ONE_WAYS:-1}" \
SSML_SUPERVISED_HOTSPOT_ALPHA="${WEATHER_HOTSPOT_ALPHA:-0.5}" \
SSML_GATE_SCORE_MODE="${WEATHER_SSML_GATE_SCORE_MODE:-peer_better_student_error}" \
SSML_SCORE_TRANSFORM="${WEATHER_SSML_SCORE_TRANSFORM:-log1p}" \
SSML_GUIDANCE_MODE="${WEATHER_SSML_GUIDANCE_MODE:-reweight_only}" \
run_logged_job \
  "worker2/weather_ssml_rescue" \
  "$LOG_DIR/weather_ssml_rescue_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_time_series_weather_ssml_rescue.sh

echo "[worker2_non_operator_remaining] job finished"
