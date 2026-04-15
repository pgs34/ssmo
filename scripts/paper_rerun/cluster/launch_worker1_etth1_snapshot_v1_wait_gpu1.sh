#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker1_etth1_snapshot_v1_wait_gpu1.lock"
flock -n 9 || {
  echo "[worker1_etth1_snapshot_v1_wait_gpu1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_infinite_loop_v1/worker1_gpu1}"
mkdir -p "$LOG_DIR"

POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-60}"
FREE_GPU_MAX_UTIL="${FREE_GPU_MAX_UTIL:-10}"
FREE_GPU_MAX_MEM_MIB="${FREE_GPU_MAX_MEM_MIB:-4096}"
TARGET_GPU="${TARGET_GPU:-1}"

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
  echo "[$(timestamp)] [worker1_etth1_snapshot_v1_wait_gpu1] waiting for gpu=$TARGET_GPU util<=$FREE_GPU_MAX_UTIL mem<=$FREE_GPU_MAX_MEM_MIB MiB"
  sleep "$POLL_INTERVAL_SEC"
done

run_logged_job \
  "worker1/etth1_teacher_ft_snapshot_handoff_v1_gpu${TARGET_GPU}" \
  "$LOG_DIR/snapshot_handoff.log" \
  env \
    GPU="$TARGET_GPU" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    BATCH_SIZE="${BATCH_SIZE_2080TI:-320}" \
    NUM_WORKERS="${NUM_WORKERS_2080TI:-4}" \
    OUTPUT_ROOT="results/time_series_etth1_teacher_ft_snapshot_handoff_v1/worker1_gpu1" \
    LOG_DIR="results/logs/time_series_etth1_teacher_ft_snapshot_handoff_v1/worker1_gpu1/cases" \
    CASE_SPECS="${SNAPSHOT_CASE_SPECS:-tft_h26_t14_l012_lr2e4:0.00020:0.012:64:0.00:-0.20:12:26:26:0.14:0.18:0.15:13:0.05}" \
    bash scripts/paper_rerun/run_time_series_etth1_teacher_ft_snapshot_handoff_v1.sh

echo "[worker1_etth1_snapshot_v1_wait_gpu1] finished"
