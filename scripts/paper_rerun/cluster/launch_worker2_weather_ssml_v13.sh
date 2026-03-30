#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker2}"
mkdir -p "$LOG_DIR"

if [[ -z "${WEATHER_V13_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  WEATHER_V13_INIT_CHECKPOINT_TEMPLATE='results/time_series_neural_ode_full_v11/baseline/time_series/{dataset}/{model}_independent_{regression_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${WEATHER_V13_PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  WEATHER_V13_PEER_INIT_CHECKPOINT_TEMPLATE='results/time_series_neural_ode_full_v11/baseline/time_series/{dataset}/{model}_independent_{regression_imitation_loss}_seed{seed}/model.pt'
fi

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
GPU="${TIME_SERIES_GPU:-0}" \
OUTPUT_ROOT="${WEATHER_V13_OUTPUT_ROOT:-results/time_series_weather_ssml_rescue_v13}" \
SEEDS="${WEATHER_V13_SEEDS:-0 1 2}" \
MODEL_PAIRS="${WEATHER_V13_MODEL_PAIRS:-neural_ode:dlinear}" \
BATCH_SIZE="${WEATHER_V13_BATCH_SIZE:-96}" \
REGRESSION_IMITATION_LOSS="${WEATHER_V13_REGRESSION_IMITATION_LOSS:-huber}" \
WARMUP_EPOCHS="${WEATHER_V13_WARMUP_EPOCHS:-1}" \
IMITATION_DECAY_START_EPOCH="${WEATHER_V13_DECAY_START_EPOCH:-2}" \
IMITATION_DECAY_END_EPOCH="${WEATHER_V13_DECAY_END_EPOCH:-12}" \
IMITATION_DECAY_MIN_SCALE="${WEATHER_V13_DECAY_MIN_SCALE:-0.0}" \
SSML_GUIDANCE_MODE="${WEATHER_V13_SSML_GUIDANCE_MODE:-hybrid}" \
SSML_GATE_SCORE_MODE="${WEATHER_V13_SSML_GATE_SCORE_MODE:-peer_better_student_error_relgain}" \
SSML_SCORE_TRANSFORM="${WEATHER_V13_SSML_SCORE_TRANSFORM:-log1p}" \
SSML_TOPK_SCOPE="${WEATHER_V13_SSML_TOPK_SCOPE:-positive}" \
SSML_SUPERVISED_WEIGHT_MODE="${WEATHER_V13_SSML_SUPERVISED_WEIGHT_MODE:-binary}" \
INIT_CHECKPOINT_TEMPLATE="${WEATHER_V13_INIT_CHECKPOINT_TEMPLATE}" \
PEER_INIT_CHECKPOINT_TEMPLATE="${WEATHER_V13_PEER_INIT_CHECKPOINT_TEMPLATE}" \
CASE_SPECS="${WEATHER_V13_CASE_SPECS:-ema_res_k5_t15:hybrid:0.10:0.03:0.01:0.15:0.95:5:0.35:0.995 ema_res_k7_t25:hybrid:0.10:0.03:0.01:0.25:0.95:7:0.35:0.995 ema_rw_k7_t25:reweight_only:0.08:0.00:0.01:0.25:0.95:7:0.25:0.995}" \
run_logged_job \
  "worker2/weather_ssml_v13" \
  "$LOG_DIR/weather_ssml_v13_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_time_series_weather_ssml_v13.sh

echo "[worker2_weather_ssml_v13] job finished"
