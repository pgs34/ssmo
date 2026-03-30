#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker2}"
mkdir -p "$LOG_DIR"

if [[ -z "${WEATHER_V15_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  WEATHER_V15_INIT_CHECKPOINT_TEMPLATE='results/time_series_neural_ode_full_v11/baseline/time_series/{dataset}/{model}_independent_{regression_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${WEATHER_V15_PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  WEATHER_V15_PEER_INIT_CHECKPOINT_TEMPLATE='results/time_series_neural_ode_full_v11/baseline/time_series/{dataset}/{model}_independent_{regression_imitation_loss}_seed{seed}/model.pt'
fi

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
GPU="${TIME_SERIES_GPU:-0}" \
OUTPUT_ROOT="${WEATHER_V15_OUTPUT_ROOT:-results/time_series_weather_ssml_rescue_v15}" \
SEEDS="${WEATHER_V15_SEEDS:-0 1 2}" \
MODEL_PAIRS="${WEATHER_V15_MODEL_PAIRS:-neural_ode:dlinear}" \
BATCH_SIZE="${WEATHER_V15_BATCH_SIZE:-96}" \
REGRESSION_IMITATION_LOSS="${WEATHER_V15_REGRESSION_IMITATION_LOSS:-huber}" \
WARMUP_EPOCHS="${WEATHER_V15_WARMUP_EPOCHS:-1}" \
IMITATION_DECAY_START_EPOCH="${WEATHER_V15_DECAY_START_EPOCH:-2}" \
IMITATION_DECAY_END_EPOCH="${WEATHER_V15_DECAY_END_EPOCH:-18}" \
IMITATION_DECAY_MIN_SCALE="${WEATHER_V15_DECAY_MIN_SCALE:-0.50}" \
SSML_GUIDANCE_MODE="${WEATHER_V15_SSML_GUIDANCE_MODE:-hybrid}" \
SSML_GATE_SCORE_MODE="${WEATHER_V15_SSML_GATE_SCORE_MODE:-peer_better_student_error_relgain}" \
SSML_SCORE_TRANSFORM="${WEATHER_V15_SSML_SCORE_TRANSFORM:-log1p}" \
SSML_TOPK_SCOPE="${WEATHER_V15_SSML_TOPK_SCOPE:-positive}" \
SSML_SUPERVISED_WEIGHT_MODE="${WEATHER_V15_SSML_SUPERVISED_WEIGHT_MODE:-binary}" \
SSML_STUDENT_ONLY="${WEATHER_V15_SSML_STUDENT_ONLY:-1}" \
SSML_FREEZE_PEER="${WEATHER_V15_SSML_FREEZE_PEER:-1}" \
SSML_WORSE_ONLY_UPDATE="${WEATHER_V15_SSML_WORSE_ONLY_UPDATE:-1}" \
INIT_CHECKPOINT_TEMPLATE="${WEATHER_V15_INIT_CHECKPOINT_TEMPLATE}" \
PEER_INIT_CHECKPOINT_TEMPLATE="${WEATHER_V15_PEER_INIT_CHECKPOINT_TEMPLATE}" \
CASE_SPECS="${WEATHER_V15_CASE_SPECS:-anch_p25_aw1e4:0.05:0.005:0.00:0.25:5:0.15:0.0001 anch_p35_aw5e4:0.05:0.0075:0.00:0.35:5:0.20:0.0005 anch_p20_aw1e3:0.08:0.010:0.00:0.20:7:0.20:0.0010}" \
LOG_DIR="results/logs/time_series_weather_ssml_v15_worker2" \
run_logged_job \
  "worker2/weather_ssml_v15" \
  "$LOG_DIR/weather_ssml_v15_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_time_series_weather_ssml_v15.sh

echo "[worker2_weather_ssml_v15] job finished"
