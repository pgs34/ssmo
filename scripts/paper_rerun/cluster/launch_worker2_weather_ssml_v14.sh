#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker2}"
mkdir -p "$LOG_DIR"

if [[ -z "${WEATHER_V14_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  WEATHER_V14_INIT_CHECKPOINT_TEMPLATE='results/time_series_neural_ode_full_v11/baseline/time_series/{dataset}/{model}_independent_{regression_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${WEATHER_V14_PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  WEATHER_V14_PEER_INIT_CHECKPOINT_TEMPLATE='results/time_series_neural_ode_full_v11/baseline/time_series/{dataset}/{model}_independent_{regression_imitation_loss}_seed{seed}/model.pt'
fi

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
GPU="${TIME_SERIES_GPU:-0}" \
OUTPUT_ROOT="${WEATHER_V14_OUTPUT_ROOT:-results/time_series_weather_ssml_rescue_v14}" \
SEEDS="${WEATHER_V14_SEEDS:-0 1 2}" \
MODEL_PAIRS="${WEATHER_V14_MODEL_PAIRS:-neural_ode:dlinear}" \
BATCH_SIZE="${WEATHER_V14_BATCH_SIZE:-96}" \
REGRESSION_IMITATION_LOSS="${WEATHER_V14_REGRESSION_IMITATION_LOSS:-huber}" \
WARMUP_EPOCHS="${WEATHER_V14_WARMUP_EPOCHS:-1}" \
IMITATION_DECAY_START_EPOCH="${WEATHER_V14_DECAY_START_EPOCH:-3}" \
IMITATION_DECAY_END_EPOCH="${WEATHER_V14_DECAY_END_EPOCH:-20}" \
IMITATION_DECAY_MIN_SCALE="${WEATHER_V14_DECAY_MIN_SCALE:-0.25}" \
SSML_GUIDANCE_MODE="${WEATHER_V14_SSML_GUIDANCE_MODE:-hybrid}" \
SSML_GATE_SCORE_MODE="${WEATHER_V14_SSML_GATE_SCORE_MODE:-peer_better_student_error_relgain}" \
SSML_SCORE_TRANSFORM="${WEATHER_V14_SSML_SCORE_TRANSFORM:-log1p}" \
SSML_TOPK_SCOPE="${WEATHER_V14_SSML_TOPK_SCOPE:-total}" \
SSML_SUPERVISED_WEIGHT_MODE="${WEATHER_V14_SSML_SUPERVISED_WEIGHT_MODE:-binary}" \
SSML_STUDENT_ONLY="${WEATHER_V14_SSML_STUDENT_ONLY:-1}" \
SSML_FREEZE_PEER="${WEATHER_V14_SSML_FREEZE_PEER:-1}" \
SSML_WORSE_ONLY_UPDATE="${WEATHER_V14_SSML_WORSE_ONLY_UPDATE:-1}" \
INIT_CHECKPOINT_TEMPLATE="${WEATHER_V14_INIT_CHECKPOINT_TEMPLATE}" \
PEER_INIT_CHECKPOINT_TEMPLATE="${WEATHER_V14_PEER_INIT_CHECKPOINT_TEMPLATE}" \
CASE_SPECS="${WEATHER_V14_CASE_SPECS:-main_dense_t100:0.10:0.020:0.00:1.00:7:0.25 main_dense_t50:0.10:0.020:0.00:0.50:7:0.30 main_dense_t100_s9:0.15:0.015:0.00:1.00:9:0.35}" \
LOG_DIR="results/logs/time_series_weather_ssml_v14_worker2" \
run_logged_job \
  "worker2/weather_ssml_v14" \
  "$LOG_DIR/weather_ssml_v14_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_time_series_weather_ssml_v14.sh

echo "[worker2_weather_ssml_v14] job finished"
