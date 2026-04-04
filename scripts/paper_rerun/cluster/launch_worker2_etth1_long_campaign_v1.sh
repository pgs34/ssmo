#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker2_etth1_long_campaign_v1}"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
GPU="${TIME_SERIES_GPU:-0}" \
OUTPUT_ROOT="${ETTH1_LONG_CAMPAIGN_OUTPUT_ROOT:-results/time_series_etth1_long_campaign_v1}" \
SEEDS="${ETTH1_LONG_CAMPAIGN_SEEDS:-0 1 2 3 4 5}" \
EPOCHS="${ETTH1_LONG_CAMPAIGN_EPOCHS:-120}" \
LIVE_PLOT_INTERVAL="${ETTH1_LONG_CAMPAIGN_LIVE_PLOT_INTERVAL:-10}" \
run_logged_job \
  "worker2/etth1_long_campaign_v1" \
  "$LOG_DIR/etth1_long_campaign_v1_gpu${TIME_SERIES_GPU:-0}.log" \
  bash scripts/paper_rerun/run_time_series_etth1_long_campaign_v1.sh

echo "[worker2_etth1_long_campaign_v1] job finished"
