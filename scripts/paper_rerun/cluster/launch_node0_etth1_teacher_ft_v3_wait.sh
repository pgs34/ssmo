#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_etth1_teacher_ft_v3_wait.lock"
flock -n 9 || {
  echo "[node0_etth1_teacher_ft_v3_wait] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_teacher_ft_v3/node0}"
mkdir -p "$LOG_DIR"

POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-60}"
FREE_GPU_MAX_UTIL="${FREE_GPU_MAX_UTIL:-10}"
FREE_GPU_MAX_MEM_MIB="${FREE_GPU_MAX_MEM_MIB:-4096}"

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
    echo "[$(timestamp)] [node0_etth1_teacher_ft_v3_wait] waiting for gpu=$gpu util<=$FREE_GPU_MAX_UTIL mem<=$FREE_GPU_MAX_MEM_MIB MiB"
    sleep "$POLL_INTERVAL_SEC"
  done
  echo "[$(timestamp)] [node0_etth1_teacher_ft_v3_wait] selected gpu=$gpu"
  run_logged_job \
    "node0/etth1_teacher_ft_v3_gpu${gpu}" \
    "$LOG_DIR/time_series_gpu${gpu}_${suffix}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      SEEDS="${SEEDS:-0 1 2}" \
      BATCH_SIZE="${BATCH_SIZE_4090:-768}" \
      NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
      OUTPUT_ROOT="results/time_series_etth1_teacher_ft_v3/${suffix}" \
      LOG_DIR="$LOG_DIR/${suffix}" \
      CASE_SPECS="$case_specs" \
      bash scripts/paper_rerun/run_time_series_etth1_teacher_ft_v3.sh
}

GPU0_CASES="${GPU0_CASES:-b10_q80_m20_t14_r00_lr2e4:0.0002:0.009:0.00002:64:0.00:-0.40:10:22:0.10:0.10:0.12:13:0.08:1.4:0.0:0.80:0.0020:3:0.10:4:18:32:0.00}"
GPU1_CASES="${GPU1_CASES:-b15_q75_m15_t12_r00_lr2e4:0.0002:0.008:0.00001:64:0.00:-0.45:10:22:0.08:0.08:0.10:13:0.08:1.2:0.0:0.75:0.0015:3:0.15:4:16:30:0.00}"

run_gpu_when_available "${TIME_SERIES_GPU0:-0}" "node0_gpu0" "$GPU0_CASES" &
PID0=$!
run_gpu_when_available "${TIME_SERIES_GPU1:-1}" "node0_gpu1" "$GPU1_CASES" &
PID1=$!

echo "[node0_etth1_teacher_ft_v3_wait] started watchers pid0=$PID0 pid1=$PID1"

wait "$PID0"
wait "$PID1"
echo "[node0_etth1_teacher_ft_v3_wait] all jobs finished"
