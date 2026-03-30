#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker3}"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
GPU="${TIME_SERIES_GPU:-0}" \
OUTPUT_ROOT="${ETTH1_EARLY_BEST_OUTPUT_ROOT:-results/time_series_etth1_early_best_v1}" \
SEEDS="${ETTH1_EARLY_BEST_SEEDS:-0 1 2}" \
CASE_SPECS="${ETTH1_EARLY_BEST_CASE_SPECS:-a0p5_fastdecay:0.5:0.001:0.02:0.02 a025_fastdecay:0.25:0.001:0.02:0.02}" \
MODEL_PAIRS="${ETTH1_EARLY_BEST_MODEL_PAIRS:-transformer:dlinear}" \
REGRESSION_IMITATION_LOSS="${ETTH1_EARLY_BEST_REGRESSION_IMITATION_LOSS:-huber}" \
ONE_WAY="${ETTH1_EARLY_BEST_ONE_WAY:-1}" \
EPOCHS="${ETTH1_EARLY_BEST_EPOCHS:-15}" \
WARMUP_EPOCHS="${ETTH1_EARLY_BEST_WARMUP_EPOCHS:-1}" \
IMITATION_DECAY_START_EPOCH="${ETTH1_EARLY_BEST_DECAY_START_EPOCH:-3}" \
IMITATION_DECAY_END_EPOCH="${ETTH1_EARLY_BEST_DECAY_END_EPOCH:-8}" \
IMITATION_DECAY_MIN_SCALE="${ETTH1_EARLY_BEST_DECAY_MIN_SCALE:-0.0}" \
LIVE_PLOT_INTERVAL="${ETTH1_EARLY_BEST_LIVE_PLOT_INTERVAL:-5}" \
SSML_GUIDANCE_MODE="${ETTH1_EARLY_BEST_SSML_GUIDANCE_MODE:-reweight_only}" \
SSML_GATE_SCORE_MODE="${ETTH1_EARLY_BEST_SSML_GATE_SCORE_MODE:-peer_better_student_error}" \
SSML_SCORE_TRANSFORM="${ETTH1_EARLY_BEST_SSML_SCORE_TRANSFORM:-none}" \
run_logged_job \
  "worker3/etth1_early_best_v1" \
  "$LOG_DIR/etth1_early_best_v1_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_time_series_etth1_early_best_v1.sh

echo "[worker3_etth1_early_best_v1] job finished"
