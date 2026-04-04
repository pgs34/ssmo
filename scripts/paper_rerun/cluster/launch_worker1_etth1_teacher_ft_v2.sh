#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker1_etth1_teacher_ft_v2.lock"
flock -n 9 || {
  echo "[worker1_etth1_teacher_ft_v2] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_teacher_ft_v2/worker1}"
mkdir -p "$LOG_DIR"

run_gpu() {
  local gpu="$1"
  local suffix="$2"
  local case_specs="$3"
  run_logged_job \
    "worker1/etth1_teacher_ft_v2_gpu${gpu}" \
    "$LOG_DIR/time_series_gpu${gpu}_${suffix}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      SEEDS="${SEEDS:-0 1 2}" \
      BATCH_SIZE="${BATCH_SIZE_2080TI:-384}" \
      NUM_WORKERS="${NUM_WORKERS_2080TI:-4}" \
      OUTPUT_ROOT="results/time_series_etth1_teacher_ft_v2/${suffix}" \
      LOG_DIR="$LOG_DIR/${suffix}" \
      CASE_SPECS="$case_specs" \
      bash scripts/paper_rerun/run_time_series_etth1_teacher_ft_v2.sh
}

GPU0_CASES="${GPU0_CASES:-trs16_res03_lr25e4:0.00025:0.012:0.00002:96:0.00:-0.35:12:26:0.12:0.18:0.15:13:0.04:1.6:0.3:5:22:42:0.08}"
GPU1_CASES="${GPU1_CASES:-trs10_res02_lr15e4:0.00015:0.008:0.00001:64:0.00:-0.15:10:24:0.08:0.12:0.12:13:0.06:1.0:0.2:4:18:36:0.05}"

run_gpu "${TIME_SERIES_GPU0:-0}" "worker1_gpu0" "$GPU0_CASES" &
PID0=$!
run_gpu "${TIME_SERIES_GPU1:-1}" "worker1_gpu1" "$GPU1_CASES" &
PID1=$!

echo "[worker1_etth1_teacher_ft_v2] started gpu${TIME_SERIES_GPU0:-0} pid=$PID0"
echo "[worker1_etth1_teacher_ft_v2] started gpu${TIME_SERIES_GPU1:-1} pid=$PID1"

wait "$PID0"
wait "$PID1"
echo "[worker1_etth1_teacher_ft_v2] all jobs finished"
