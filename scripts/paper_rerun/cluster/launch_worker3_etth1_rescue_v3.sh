#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker3}"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
GPU="${TIME_SERIES_GPU:-0}" \
OUTPUT_ROOT="${ETTH1_RESCUE_OUTPUT_ROOT:-results/time_series_etth1_rescue_v3}" \
SEEDS="${ETTH1_RESCUE_SEEDS:-0}" \
CASE_SPECS="${ETTH1_RESCUE_CASE_SPECS:-a0:0.0:0.001:0.02:0.02 a0p5:0.5:0.001:0.02:0.02 sparse:0.0:0.002:0.05:0.01}" \
MODEL_PAIRS="${ETTH1_RESCUE_MODEL_PAIRS:-transformer:dlinear}" \
REGRESSION_IMITATION_LOSS="${ETTH1_RESCUE_REGRESSION_IMITATION_LOSS:-huber}" \
ONE_WAY="${ETTH1_RESCUE_ONE_WAY:-1}" \
WARMUP_EPOCHS="${ETTH1_RESCUE_WARMUP_EPOCHS:-5}" \
IMITATION_DECAY_START_EPOCH="${ETTH1_RESCUE_DECAY_START_EPOCH:-10}" \
IMITATION_DECAY_END_EPOCH="${ETTH1_RESCUE_DECAY_END_EPOCH:-50}" \
IMITATION_DECAY_MIN_SCALE="${ETTH1_RESCUE_DECAY_MIN_SCALE:-0.1}" \
SSML_GUIDANCE_MODE="${ETTH1_RESCUE_SSML_GUIDANCE_MODE:-reweight_only}" \
SSML_GATE_SCORE_MODE="${ETTH1_RESCUE_SSML_GATE_SCORE_MODE:-peer_better_student_error}" \
SSML_SCORE_TRANSFORM="${ETTH1_RESCUE_SSML_SCORE_TRANSFORM:-none}" \
run_logged_job \
  "worker3/etth1_rescue_v3" \
  "$LOG_DIR/etth1_rescue_v3_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_time_series_etth1_rescue_v3.sh

echo "[worker3_etth1_rescue_v3] job finished"
