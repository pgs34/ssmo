#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker2}"
mkdir -p "$LOG_DIR"

if [[ -z "${WEATHER_V17_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  WEATHER_V17_INIT_CHECKPOINT_TEMPLATE='results/time_series_ssml_topk_sweep_v1/weather/baseline_dml/time_series/{dataset}/{model}_independent_mse_seed{seed}/model.pt'
fi
if [[ -z "${WEATHER_V17_PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  WEATHER_V17_PEER_INIT_CHECKPOINT_TEMPLATE='results/time_series_ssml_topk_sweep_v1/weather/baseline_dml/time_series/{dataset}/{model}_independent_mse_seed{seed}/model.pt'
fi

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
GPU="${TIME_SERIES_GPU:-0}" \
OUTPUT_ROOT="${WEATHER_V17_OUTPUT_ROOT:-results/time_series_weather_ssml_rescue_v17}" \
SEEDS="${WEATHER_V17_SEEDS:-0 1 2}" \
MODEL_PAIRS="${WEATHER_V17_MODEL_PAIRS:-transformer:dlinear}" \
BATCH_SIZE="${WEATHER_V17_BATCH_SIZE:-96}" \
REGRESSION_IMITATION_LOSS="${WEATHER_V17_REGRESSION_IMITATION_LOSS:-huber}" \
WARMUP_EPOCHS="${WEATHER_V17_WARMUP_EPOCHS:-1}" \
IMITATION_DECAY_START_EPOCH="${WEATHER_V17_DECAY_START_EPOCH:-2}" \
IMITATION_DECAY_END_EPOCH="${WEATHER_V17_DECAY_END_EPOCH:-18}" \
IMITATION_DECAY_MIN_SCALE="${WEATHER_V17_DECAY_MIN_SCALE:-0.35}" \
SSML_GUIDANCE_MODE="${WEATHER_V17_SSML_GUIDANCE_MODE:-hybrid}" \
SSML_GATE_SCORE_MODE="${WEATHER_V17_SSML_GATE_SCORE_MODE:-peer_better_student_error_relgain}" \
SSML_SCORE_TRANSFORM="${WEATHER_V17_SSML_SCORE_TRANSFORM:-log1p}" \
SSML_TOPK_SCOPE="${WEATHER_V17_SSML_TOPK_SCOPE:-total}" \
SSML_SUPERVISED_WEIGHT_MODE="${WEATHER_V17_SSML_SUPERVISED_WEIGHT_MODE:-binary}" \
SSML_STUDENT_ONLY="${WEATHER_V17_SSML_STUDENT_ONLY:-1}" \
SSML_FREEZE_PEER="${WEATHER_V17_SSML_FREEZE_PEER:-1}" \
SSML_WORSE_ONLY_UPDATE="${WEATHER_V17_SSML_WORSE_ONLY_UPDATE:-1}" \
INIT_CHECKPOINT_TEMPLATE="${WEATHER_V17_INIT_CHECKPOINT_TEMPLATE}" \
PEER_INIT_CHECKPOINT_TEMPLATE="${WEATHER_V17_PEER_INIT_CHECKPOINT_TEMPLATE}" \
CASE_SPECS="${WEATHER_V17_CASE_SPECS:-cap_k5_t08:0.10:0.004:0.00:0.08:5:5:0.20:0.0005:0.12 cap_k7_t10:0.08:0.004:0.00:0.10:7:7:0.20:0.0005:0.15 cap_k9_t15:0.05:0.003:0.00:0.15:9:9:0.15:0.0001:0.18}" \
LOG_DIR="results/logs/time_series_weather_ssml_v17_worker2" \
run_logged_job \
  "worker2/weather_ssml_v17" \
  "$LOG_DIR/weather_ssml_v17_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_time_series_weather_ssml_v17.sh

echo "[worker2_weather_ssml_v17] job finished"
