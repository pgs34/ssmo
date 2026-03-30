#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$(paper_results_root)/logs/worker1}"
mkdir -p "$LOG_DIR"

run_time_series_gpu0() {
  CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU0:-0}" \
  METHODS="${TIME_SERIES_METHODS:-independent dml ssml}" \
  DATASETS="${TIME_SERIES_DATASETS_GPU0:-etth1 electricity}" \
  BATCH_SIZE="${TIME_SERIES_BATCH_SIZE_GPU0:-32}" \
  NUM_WORKERS="${TIME_SERIES_NUM_WORKERS_GPU0:-2}" \
  OUTPUT_DIR="${TIME_SERIES_OUTPUT_DIR:-$(paper_results_root)/time_series}" \
  run_logged_job \
    "worker1/time_series_gpu${TIME_SERIES_GPU0:-0}" \
    "$LOG_DIR/time_series_gpu${TIME_SERIES_GPU0:-0}.log" \
    bash scripts/paper_rerun/run_core_time_series.sh
}

run_time_series_gpu1() {
  CUDA_VISIBLE_DEVICES="${TIME_SERIES_GPU1:-1}" \
  METHODS="${TIME_SERIES_METHODS:-independent dml ssml}" \
  DATASETS="${TIME_SERIES_DATASETS_GPU1:-weather}" \
  BATCH_SIZE="${TIME_SERIES_BATCH_SIZE_GPU1:-32}" \
  NUM_WORKERS="${TIME_SERIES_NUM_WORKERS_GPU1:-2}" \
  OUTPUT_DIR="${TIME_SERIES_OUTPUT_DIR:-$(paper_results_root)/time_series}" \
  run_logged_job \
    "worker1/time_series_gpu${TIME_SERIES_GPU1:-1}" \
    "$LOG_DIR/time_series_gpu${TIME_SERIES_GPU1:-1}.log" \
    bash scripts/paper_rerun/run_core_time_series.sh
}

run_time_series_gpu0 &
PID_TS0=$!
run_time_series_gpu1 &
PID_TS1=$!

echo "[worker1] started gpu${TIME_SERIES_GPU0:-0} pid=$PID_TS0"
echo "[worker1] started gpu${TIME_SERIES_GPU1:-1} pid=$PID_TS1"
echo "[worker1] results_root=$(paper_results_root)"

wait "$PID_TS0"
wait "$PID_TS1"
echo "[worker1] all jobs finished"
