#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_etth1_teacher_ft_v2.lock"
flock -n 9 || {
  echo "[node0_etth1_teacher_ft_v2] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_teacher_ft_v2/node0}"
mkdir -p "$LOG_DIR"

run_gpu() {
  local gpu="$1"
  local suffix="$2"
  local case_specs="$3"
  run_logged_job \
    "node0/etth1_teacher_ft_v2_gpu${gpu}" \
    "$LOG_DIR/time_series_gpu${gpu}_${suffix}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      SEEDS="${SEEDS:-0 1 2}" \
      BATCH_SIZE="${BATCH_SIZE_4090:-768}" \
      NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
      OUTPUT_ROOT="results/time_series_etth1_teacher_ft_v2/${suffix}" \
      LOG_DIR="$LOG_DIR/${suffix}" \
      CASE_SPECS="$case_specs" \
      bash scripts/paper_rerun/run_time_series_etth1_teacher_ft_v2.sh
}

GPU0_CASES="${GPU0_CASES:-trs14_res02_lr2e4:0.0002:0.010:0.00001:64:0.00:-0.20:10:24:0.10:0.15:0.15:13:0.05:1.4:0.2:4:20:40:0.05}"
GPU1_CASES="${GPU1_CASES:-trs12_res00_lr2e4:0.0002:0.010:0.00001:64:0.00:-0.25:10:22:0.10:0.18:0.12:13:0.05:1.2:0.0:4:18:38:0.05}"

run_gpu "${TIME_SERIES_GPU0:-0}" "node0_gpu0" "$GPU0_CASES" &
PID0=$!
run_gpu "${TIME_SERIES_GPU1:-1}" "node0_gpu1" "$GPU1_CASES" &
PID1=$!

echo "[node0_etth1_teacher_ft_v2] started gpu${TIME_SERIES_GPU0:-0} pid=$PID0"
echo "[node0_etth1_teacher_ft_v2] started gpu${TIME_SERIES_GPU1:-1} pid=$PID1"

wait "$PID0"
wait "$PID1"
echo "[node0_etth1_teacher_ft_v2] all jobs finished"
