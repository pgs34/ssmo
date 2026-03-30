#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker2}"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
GPU="${TIME_SERIES_GPU:-0}" \
OUTPUT_ROOT="${WEATHER_V12_OUTPUT_ROOT:-results/time_series_weather_ssml_rescue_v12}" \
SEEDS="${WEATHER_V12_SEEDS:-0 1 2}" \
MODEL_PAIRS="${WEATHER_V12_MODEL_PAIRS:-transformer:dlinear}" \
BATCH_SIZE="${WEATHER_V12_BATCH_SIZE:-96}" \
WARMUP_EPOCHS="${WEATHER_V12_WARMUP_EPOCHS:-1}" \
IMITATION_DECAY_START_EPOCH="${WEATHER_V12_DECAY_START_EPOCH:-3}" \
IMITATION_DECAY_END_EPOCH="${WEATHER_V12_DECAY_END_EPOCH:-15}" \
IMITATION_DECAY_MIN_SCALE="${WEATHER_V12_DECAY_MIN_SCALE:-0.2}" \
SSML_GATE_SCORE_MODE="${WEATHER_V12_SSML_GATE_SCORE_MODE:-peer_better_student_error_relgain}" \
SSML_SCORE_TRANSFORM="${WEATHER_V12_SSML_SCORE_TRANSFORM:-log1p}" \
SSML_TOPK_SCOPE="${WEATHER_V12_SSML_TOPK_SCOPE:-positive}" \
SSML_SUPERVISED_WEIGHT_MODE="${WEATHER_V12_SSML_SUPERVISED_WEIGHT_MODE:-binary}" \
CASE_SPECS="${WEATHER_V12_CASE_SPECS:-rw_dense_t30:reweight_only:0.05:0.0:0.0:0.30:1.0 rw_dense_t50:reweight_only:0.10:0.0:0.0:0.50:1.0 rw_dense_t100:reweight_only:0.10:0.0:0.0:1.00:1.0 hyb_dense_t30:hybrid:0.05:0.02:0.0:0.30:1.0}" \
run_logged_job \
  "worker2/weather_ssml_v12" \
  "$LOG_DIR/weather_ssml_v12_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_time_series_weather_ssml_v12.sh

echo "[worker2_weather_ssml_v12] job finished"
