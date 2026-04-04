#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker1_etth1_teacher_ft_v3_wait.lock"
flock -n 9 || {
  echo "[worker1_etth1_teacher_ft_v3_wait] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_teacher_ft_v3/worker1}"
mkdir -p "$LOG_DIR"

POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-60}"
FREE_GPU_MAX_UTIL="${FREE_GPU_MAX_UTIL:-10}"
FREE_GPU_MAX_MEM_MIB="${FREE_GPU_MAX_MEM_MIB:-3072}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

gpu_is_available() {
  local target_gpu="$1"
  local idx util mem
  while IFS=',' read -r idx util mem; do
    idx="${idx//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    mem="${mem//[[:space:]]/}"
    if [[ "$idx" != "$target_gpu" ]]; then
      continue
    fi
    if (( util <= FREE_GPU_MAX_UTIL && mem <= FREE_GPU_MAX_MEM_MIB )); then
      return 0
    fi
    return 1
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
  return 1
}

run_gpu_when_available() {
  local gpu="$1"
  local suffix="$2"
  local case_specs="$3"
  while ! gpu_is_available "$gpu"; do
    echo "[$(timestamp)] [worker1_etth1_teacher_ft_v3_wait] waiting for gpu=$gpu util<=$FREE_GPU_MAX_UTIL mem<=$FREE_GPU_MAX_MEM_MIB MiB"
    sleep "$POLL_INTERVAL_SEC"
  done
  echo "[$(timestamp)] [worker1_etth1_teacher_ft_v3_wait] selected gpu=$gpu"
  run_logged_job \
    "worker1/etth1_teacher_ft_v3_gpu${gpu}" \
    "$LOG_DIR/time_series_gpu${gpu}_${suffix}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      SEEDS="${SEEDS:-0 1 2}" \
      BATCH_SIZE="${BATCH_SIZE_2080TI:-384}" \
      NUM_WORKERS="${NUM_WORKERS_2080TI:-4}" \
      OUTPUT_ROOT="results/time_series_etth1_teacher_ft_v3/${suffix}" \
      LOG_DIR="$LOG_DIR/${suffix}" \
      CASE_SPECS="$case_specs" \
      bash scripts/paper_rerun/run_time_series_etth1_teacher_ft_v3.sh
}

GPU0_CASES="${GPU0_CASES:-b08_q85_m25_t16_r01_lr25e4:0.00025:0.010:0.00003:96:0.00:-0.55:12:24:0.10:0.10:0.14:13:0.07:1.6:0.1:0.85:0.0025:3:0.08:4:20:34:0.00}"
GPU1_CASES="${GPU1_CASES:-b20_q70_m10_t10_r00_lr15e4:0.00015:0.007:0.00001:64:0.00:-0.35:10:20:0.08:0.05:0.08:13:0.10:1.0:0.0:0.70:0.0010:3:0.20:4:16:28:0.00}"

run_gpu_when_available "${TIME_SERIES_GPU0:-0}" "worker1_gpu0" "$GPU0_CASES" &
PID0=$!
run_gpu_when_available "${TIME_SERIES_GPU1:-1}" "worker1_gpu1" "$GPU1_CASES" &
PID1=$!

echo "[worker1_etth1_teacher_ft_v3_wait] started watchers pid0=$PID0 pid1=$PID1"

wait "$PID0"
wait "$PID1"
echo "[worker1_etth1_teacher_ft_v3_wait] all jobs finished"
