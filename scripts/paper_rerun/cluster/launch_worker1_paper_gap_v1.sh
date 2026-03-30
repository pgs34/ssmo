#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/paper_gap_v1/worker1}"
PLOT_ROOT="${PLOT_ROOT:-results/plots/paper_gap_v1}"
PLOT_INTERVAL_SEC="${PLOT_INTERVAL_SEC:-300}"
ENABLE_PLOT_WATCHER="${ENABLE_PLOT_WATCHER:-1}"
mkdir -p "$LOG_DIR"

PLOT_WATCH_PIDS=()

start_plot_watch() {
  local label="$1"
  local input_dir="$2"
  local output_dir="$3"
  local logfile="$4"
  local pid
  if [[ "$ENABLE_PLOT_WATCHER" != "1" ]]; then
    return 0
  fi
  LOGFILE="$logfile" INTERVAL_SEC="$PLOT_INTERVAL_SEC" LABEL="$label" \
    bash scripts/paper_rerun/watch_summary_plots.sh "$input_dir" "$output_dir" &
  pid="$!"
  PLOT_WATCH_PIDS+=("$pid")
  echo "[worker1_paper_gap_v1] plot watcher $label pid=$pid"
}

cleanup() {
  local pid
  for pid in "${PLOT_WATCH_PIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
}

trap cleanup EXIT

start_plot_watch \
  "classification_homo_noise" \
  "results/paper_gap_v1/classification_homo_noise" \
  "$PLOT_ROOT/classification_homo_noise" \
  "$LOG_DIR/classification_homo_noise_plots.log"

start_plot_watch \
  "classification_hetero_noise" \
  "results/paper_gap_v1/classification_hetero_noise" \
  "$PLOT_ROOT/classification_hetero_noise" \
  "$LOG_DIR/classification_hetero_noise_plots.log"

run_logged_job \
  "worker1/classification_homo_noise_v1" \
  "$LOG_DIR/classification_homo_noise_gpu${CLASSIFICATION_GPU0:-0}.log" \
  env \
    CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU0:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_DIR="results/paper_gap_v1/classification_homo_noise" \
    DATASETS="${DATASETS_HOMO:-cifar100}" \
    METHODS="${METHODS_HOMO:-independent dml ssml}" \
    MODEL_PAIRS="${MODEL_PAIRS_HOMO:-resnet34_gelu:resnet34_gelu}" \
    REQUIRE_DISTINCT_PEER="0" \
    LABEL_NOISE_CONDITIONS="${LABEL_NOISE_CONDITIONS_HOMO:-none:0.0 symmetric:0.2 symmetric:0.4 asymmetric:0.2 asymmetric:0.4}" \
    EPOCHS="${EPOCHS_HOMO:-100}" \
    BATCH_SIZE="${BATCH_SIZE_HOMO:-128}" \
    NUM_WORKERS="${NUM_WORKERS_HOMO:-8}" \
    DOWNLOAD="${DOWNLOAD_HOMO:-1}" \
    bash scripts/paper_rerun/run_core_classification.sh \
  &

run_logged_job \
  "worker1/classification_hetero_noise_v1" \
  "$LOG_DIR/classification_hetero_noise_gpu${CLASSIFICATION_GPU1:-1}.log" \
  env \
    CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU1:-1}" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_DIR="results/paper_gap_v1/classification_hetero_noise" \
    DATASETS="${DATASETS_HETERO:-cifar100}" \
    METHODS="${METHODS_HETERO:-independent dml ssml}" \
    MODEL_PAIRS="${MODEL_PAIRS_HETERO:-resnet18:vit_b16}" \
    REQUIRE_DISTINCT_PEER="1" \
    LABEL_NOISE_CONDITIONS="${LABEL_NOISE_CONDITIONS_HETERO:-none:0.0 symmetric:0.2 symmetric:0.4 asymmetric:0.2 asymmetric:0.4}" \
    EPOCHS="${EPOCHS_HETERO:-100}" \
    BATCH_SIZE="${BATCH_SIZE_HETERO:-128}" \
    NUM_WORKERS="${NUM_WORKERS_HETERO:-8}" \
    DOWNLOAD="${DOWNLOAD_HETERO:-1}" \
    bash scripts/paper_rerun/run_core_classification.sh \
  &

wait

bash scripts/paper_rerun/refresh_summary_plots.sh \
  "results/paper_gap_v1/classification_homo_noise" \
  "$PLOT_ROOT/classification_homo_noise" \
  >>"$LOG_DIR/classification_homo_noise_plots.log" 2>&1 || true

bash scripts/paper_rerun/refresh_summary_plots.sh \
  "results/paper_gap_v1/classification_hetero_noise" \
  "$PLOT_ROOT/classification_hetero_noise" \
  >>"$LOG_DIR/classification_hetero_noise_plots.log" 2>&1 || true

echo "[worker1_paper_gap_v1] job finished"
