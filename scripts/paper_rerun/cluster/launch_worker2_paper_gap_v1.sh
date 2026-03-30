#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/paper_gap_v1/worker2}"
PLOT_ROOT="${PLOT_ROOT:-results/plots/paper_gap_v1}"
PLOT_INTERVAL_SEC="${PLOT_INTERVAL_SEC:-300}"
ENABLE_PLOT_WATCHER="${ENABLE_PLOT_WATCHER:-1}"
mkdir -p "$LOG_DIR"

PLOT_WATCH_PID=""

if [[ "$ENABLE_PLOT_WATCHER" == "1" ]]; then
  LOGFILE="$LOG_DIR/time_series_canonical_plots.log" INTERVAL_SEC="$PLOT_INTERVAL_SEC" LABEL="time_series_canonical" \
    bash scripts/paper_rerun/watch_summary_plots.sh \
    "results/paper_gap_v1/time_series_canonical" \
    "$PLOT_ROOT/time_series_canonical" &
  PLOT_WATCH_PID="$!"
  echo "[worker2_paper_gap_v1] plot watcher pid=$PLOT_WATCH_PID"
fi

cleanup() {
  if [[ -n "${PLOT_WATCH_PID:-}" ]]; then
    kill "$PLOT_WATCH_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

run_logged_job \
  "worker2/time_series_canonical_v1" \
  "$LOG_DIR/time_series_canonical_gpu${TIME_SERIES_GPU:-0}.log" \
  env \
    CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_DIR="results/paper_gap_v1/time_series_canonical" \
    DATASETS="${DATASETS_TS:-etth1 electricity weather}" \
    METHODS="${METHODS_TS:-independent dml ssml}" \
    MODEL_PAIRS="${MODEL_PAIRS_TS:-transformer:dlinear}" \
    REQUIRE_DISTINCT_PEER="1" \
    EPOCHS="${EPOCHS_TS:-60}" \
    BATCH_SIZE="${BATCH_SIZE_TS:-64}" \
    NUM_WORKERS="${NUM_WORKERS_TS:-4}" \
    PRED_LENS="${PRED_LENS_TS:-24}" \
    REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS_TS:-mse}" \
    bash scripts/paper_rerun/run_core_time_series.sh

bash scripts/paper_rerun/refresh_summary_plots.sh \
  "results/paper_gap_v1/time_series_canonical" \
  "$PLOT_ROOT/time_series_canonical" \
  >>"$LOG_DIR/time_series_canonical_plots.log" 2>&1 || true

echo "[worker2_paper_gap_v1] job finished"
