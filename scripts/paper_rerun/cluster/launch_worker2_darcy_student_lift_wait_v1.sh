#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker2_darcy_student_lift_wait_v1.lock"
flock -n 9 || {
  echo "[worker2_darcy_student_lift_wait_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/operator_darcy_student_lift_v1/worker2}"
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
  echo "[$(timestamp)] [worker2_darcy_student_lift_wait_v1] waiting for gpu=$TARGET_GPU util<=$FREE_GPU_MAX_UTIL mem<=$FREE_GPU_MAX_MEM_MIB MiB"
  sleep "$POLL_INTERVAL_SEC"
done

echo "[$(timestamp)] [worker2_darcy_student_lift_wait_v1] selected gpu=$TARGET_GPU"

run_logged_job \
  "worker2/operator_darcy_student_lift_v1" \
  "$LOG_DIR/launcher.out" \
  env \
    GPU="$TARGET_GPU" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-results/operator_darcy_student_lift_v1}" \
    LOG_DIR="${INNER_LOG_DIR:-results/logs/operator_darcy_student_lift_v1}" \
    SEEDS="${SEEDS:-0 1 2}" \
    EPOCHS="${EPOCHS:-90}" \
    BATCH_SIZE="${BATCH_SIZE:-16}" \
    NUM_WORKERS="${NUM_WORKERS:-2}" \
    BASE_LR="${BASE_LR:-3e-4}" \
    HUBER_LR="${HUBER_LR:-2e-4}" \
    bash scripts/paper_rerun/run_operator_darcy_student_lift_v1.sh

echo "[worker2_darcy_student_lift_wait_v1] job finished"
