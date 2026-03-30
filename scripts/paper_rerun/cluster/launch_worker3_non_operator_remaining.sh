#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker3}"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
GPU="${TIME_SERIES_GPU:-0}" \
OUTPUT_DIR="${ETTH1_OUTPUT_DIR:-results/time_series_etth1_ssml_confirm_v2}" \
LAMBDA_IMITATION="${ETTH1_LAMBDA_IMITATION:-0.005}" \
MARGIN="${ETTH1_MARGIN:-0.05}" \
SSML_TOPK_RATIO="${ETTH1_TOPK_RATIO:-0.01}" \
REGRESSION_IMITATION_LOSS="${ETTH1_REGRESSION_IMITATION_LOSS:-huber}" \
SSML_SUPERVISED_HOTSPOT_ALPHA="${ETTH1_HOTSPOT_ALPHA:-0.5}" \
SSML_GATE_SCORE_MODE="${ETTH1_SSML_GATE_SCORE_MODE:-peer_better_student_error}" \
SSML_SCORE_TRANSFORM="${ETTH1_SSML_SCORE_TRANSFORM:-none}" \
SSML_GUIDANCE_MODE="${ETTH1_SSML_GUIDANCE_MODE:-reweight_only}" \
run_logged_job \
  "worker3/etth1_ssml_confirm" \
  "$LOG_DIR/etth1_ssml_confirm_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_time_series_etth1_ssml_confirm.sh

echo "[worker3_non_operator_remaining] job finished"
