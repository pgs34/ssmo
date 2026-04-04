#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_complementary_v4/worker2}"
PLOT_ROOT="${PLOT_ROOT:-results/plots/time_series_etth1_complementary_v4}"
PLOT_INTERVAL_SEC="${PLOT_INTERVAL_SEC:-300}"
ENABLE_PLOT_WATCHER="${ENABLE_PLOT_WATCHER:-1}"
mkdir -p "$LOG_DIR"

PLOT_WATCH_PID=""

if [[ "$ENABLE_PLOT_WATCHER" == "1" ]]; then
  LOGFILE="$LOG_DIR/plots.log" INTERVAL_SEC="$PLOT_INTERVAL_SEC" LABEL="time_series_etth1_complementary_v4" \
    bash scripts/paper_rerun/watch_summary_plots.sh \
    "results/time_series_etth1_complementary_v4" \
    "$PLOT_ROOT" &
  PLOT_WATCH_PID="$!"
  echo "[worker2_etth1_complementary_v4] plot watcher pid=$PLOT_WATCH_PID"
fi

cleanup() {
  if [[ -n "${PLOT_WATCH_PID:-}" ]]; then
    kill "$PLOT_WATCH_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

run_logged_job \
  "worker2/etth1_complementary_v4" \
  "$LOG_DIR/time_series_gpu${TIME_SERIES_GPU:-0}.log" \
  env \
    GPU="${TIME_SERIES_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_ROOT="results/time_series_etth1_complementary_v4" \
    bash scripts/paper_rerun/run_time_series_etth1_complementary_v4.sh

bash scripts/paper_rerun/refresh_summary_plots.sh \
  "results/time_series_etth1_complementary_v4" \
  "$PLOT_ROOT" \
  >>"$LOG_DIR/plots.log" 2>&1 || true

echo "[worker2_etth1_complementary_v4] job finished"
