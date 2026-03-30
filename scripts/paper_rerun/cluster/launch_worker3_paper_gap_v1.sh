#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/paper_gap_v1/worker3}"
PLOT_ROOT="${PLOT_ROOT:-results/plots/paper_gap_v1}"
PLOT_INTERVAL_SEC="${PLOT_INTERVAL_SEC:-300}"
ENABLE_PLOT_WATCHER="${ENABLE_PLOT_WATCHER:-1}"
mkdir -p "$LOG_DIR"

PLOT_WATCH_PID=""

if [[ "$ENABLE_PLOT_WATCHER" == "1" ]]; then
  LOGFILE="$LOG_DIR/operator_pairs_plots.log" INTERVAL_SEC="$PLOT_INTERVAL_SEC" LABEL="operator_pairs" \
    bash scripts/paper_rerun/watch_summary_plots.sh \
    "results/paper_gap_v1/operator_pairs" \
    "$PLOT_ROOT/operator_pairs" &
  PLOT_WATCH_PID="$!"
  echo "[worker3_paper_gap_v1] plot watcher pid=$PLOT_WATCH_PID"
fi

cleanup() {
  if [[ -n "${PLOT_WATCH_PID:-}" ]]; then
    kill "$PLOT_WATCH_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

run_logged_job \
  "worker3/operator_pairs_v1" \
  "$LOG_DIR/operator_pairs_gpu${OPERATOR_GPU:-0}.log" \
  env \
    CUDA_VISIBLE_DEVICES="${OPERATOR_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_DIR="results/paper_gap_v1/operator_pairs" \
    DATASETS="${DATASETS_OP:-burgers darcy navier_stokes}" \
    METHODS="${METHODS_OP:-independent dml ssml}" \
    MODEL_PAIRS="${MODEL_PAIRS_OP:-fno:deeponet fno:fno deeponet:deeponet}" \
    REQUIRE_DISTINCT_PEER="0" \
    EPOCHS="${EPOCHS_OP:-150}" \
    BATCH_SIZE="${BATCH_SIZE_OP:-16}" \
    NUM_WORKERS="${NUM_WORKERS_OP:-4}" \
    REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS_OP:-mse}" \
    DOWNLOAD="${DOWNLOAD_OP:-1}" \
    bash scripts/paper_rerun/run_core_operator.sh

bash scripts/paper_rerun/refresh_summary_plots.sh \
  "results/paper_gap_v1/operator_pairs" \
  "$PLOT_ROOT/operator_pairs" \
  >>"$LOG_DIR/operator_pairs_plots.log" 2>&1 || true

echo "[worker3_paper_gap_v1] job finished"
