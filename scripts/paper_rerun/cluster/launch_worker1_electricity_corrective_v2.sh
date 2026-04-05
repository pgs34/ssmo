#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker1_electricity_corrective_v2.lock"
flock -n 9 || {
  echo "[worker1_electricity_corrective_v2] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_electricity_corrective_v2/worker1}"
mkdir -p "$LOG_DIR"

run_gpu() {
  local gpu="$1"
  local suffix="$2"
  local case_specs="$3"
  run_logged_job \
    "worker1/electricity_corrective_v2_gpu${gpu}" \
    "$LOG_DIR/time_series_gpu${gpu}_${suffix}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      SEEDS="${SEEDS:-0 1 2}" \
      BATCH_SIZE="${BATCH_SIZE_2080TI:-64}" \
      NUM_WORKERS="${NUM_WORKERS_2080TI:-4}" \
      OUTPUT_ROOT="results/time_series_electricity_corrective_v2/${suffix}" \
      LOG_DIR="$LOG_DIR/${suffix}" \
      CASE_SPECS="$case_specs" \
      bash scripts/paper_rerun/run_time_series_electricity_corrective_v2.sh
}

GPU0_CASES="${GPU0_CASES:-late_b10_q80_m20:0.12:0.0010:64:0.00:6:6:10:12:24:0.00:0.80:0.0020:3:0.10:8:12:0.0001}"
GPU1_CASES="${GPU1_CASES:-late_b15_q75_m15_do10:0.14:0.0010:64:0.10:6:6:10:14:28:0.00:0.75:0.0015:3:0.15:8:12:0.0001}"

run_gpu "${TIME_SERIES_GPU0:-0}" "worker1_gpu0" "$GPU0_CASES" &
PID0=$!
run_gpu "${TIME_SERIES_GPU1:-1}" "worker1_gpu1" "$GPU1_CASES" &
PID1=$!

echo "[worker1_electricity_corrective_v2] started gpu${TIME_SERIES_GPU0:-0} pid=$PID0"
echo "[worker1_electricity_corrective_v2] started gpu${TIME_SERIES_GPU1:-1} pid=$PID1"

wait "$PID0"
wait "$PID1"
echo "[worker1_electricity_corrective_v2] all jobs finished"
