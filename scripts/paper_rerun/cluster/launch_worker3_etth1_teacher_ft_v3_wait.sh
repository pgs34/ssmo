#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker3_etth1_teacher_ft_v3_wait.lock"
flock -n 9 || {
  echo "[worker3_etth1_teacher_ft_v3_wait] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_teacher_ft_v3/worker3}"
mkdir -p "$LOG_DIR"

POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-60}"
FREE_GPU_MAX_UTIL="${FREE_GPU_MAX_UTIL:-10}"
FREE_GPU_MAX_MEM_MIB="${FREE_GPU_MAX_MEM_MIB:-4096}"
TARGET_GPU="${TARGET_GPU:-0}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

gpu_is_available() {
  local idx util mem
  while IFS=',' read -r idx util mem; do
    idx="${idx//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    mem="${mem//[[:space:]]/}"
    if [[ "$idx" != "$TARGET_GPU" ]]; then
      continue
    fi
    if (( util <= FREE_GPU_MAX_UTIL && mem <= FREE_GPU_MAX_MEM_MIB )); then
      return 0
    fi
    return 1
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
  return 1
}

while ! gpu_is_available; do
  echo "[$(timestamp)] [worker3_etth1_teacher_ft_v3_wait] waiting for gpu=$TARGET_GPU util<=$FREE_GPU_MAX_UTIL mem<=$FREE_GPU_MAX_MEM_MIB MiB"
  sleep "$POLL_INTERVAL_SEC"
done

echo "[$(timestamp)] [worker3_etth1_teacher_ft_v3_wait] selected gpu=$TARGET_GPU"

run_logged_job \
  "worker3/etth1_teacher_ft_v3" \
  "$LOG_DIR/launcher.out" \
  env \
    GPU="$TARGET_GPU" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    BATCH_SIZE="${BATCH_SIZE:-768}" \
    NUM_WORKERS="${NUM_WORKERS:-4}" \
    OUTPUT_ROOT="results/time_series_etth1_teacher_ft_v3/worker3_gpu0" \
    LOG_DIR="$LOG_DIR/cases" \
    CASE_SPECS="${CASE_SPECS:-b15_q78_m15_t13_r02_lr2e4:0.0002:0.009:0.00002:64:0.00:-0.45:10:22:0.10:0.10:0.10:13:0.08:1.3:0.2:0.78:0.0015:3:0.15:4:18:32:0.00}" \
    bash scripts/paper_rerun/run_time_series_etth1_teacher_ft_v3.sh

echo "[worker3_etth1_teacher_ft_v3_wait] job finished"
