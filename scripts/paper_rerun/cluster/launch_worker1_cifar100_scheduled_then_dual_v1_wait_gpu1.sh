#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker1_cifar100_scheduled_then_dual_v1_wait_gpu1.lock"
flock -n 9 || {
  echo "[worker1_cifar100_scheduled_then_dual_v1_wait_gpu1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_infinite_loop_v1/worker1_gpu1}"
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
  echo "[$(timestamp)] [worker1_cifar100_scheduled_then_dual_v1_wait_gpu1] waiting for gpu=$TARGET_GPU util<=$FREE_GPU_MAX_UTIL mem<=$FREE_GPU_MAX_MEM_MIB MiB"
  sleep "$POLL_INTERVAL_SEC"
done

run_logged_job \
  "worker1/cifar100_scheduled_complement_v1_gpu${TARGET_GPU}" \
  "$LOG_DIR/scheduled_complement.log" \
  env \
    GPU="$TARGET_GPU" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    BATCH_SIZE="${BATCH_SIZE_2080TI:-128}" \
    NUM_WORKERS="${NUM_WORKERS_2080TI:-4}" \
    OUTPUT_ROOT="results/classification_cifar100_scheduled_complement_v1/worker1_gpu1" \
    LOG_DIR="results/logs/classification_cifar100_scheduled_complement_v1/worker1_gpu1/cases" \
    CASE_SPECS="${SCHEDULED_CASE_SPECS:-uh_sched_df10_x05_r30_60:useful_hard_sample_confident:0.24:0.04:0.012:0.020:0.35:0.020:4:6.0:0.0004:0.85:0.80:2:0.50:0.03:0.72:0.92:0.03:4:18:55:0.30:0.10:0.90:0.50:30:60}" \
    bash scripts/paper_rerun/run_classification_cifar100_scheduled_complement_v1.sh

run_logged_job \
  "worker1/cifar100_dual_peer_consensus_v1_gpu${TARGET_GPU}" \
  "$LOG_DIR/dual_peer_consensus.log" \
  env \
    GPU="$TARGET_GPU" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    BATCH_SIZE="${BATCH_SIZE_2080TI:-128}" \
    NUM_WORKERS="${NUM_WORKERS_2080TI:-4}" \
    OUTPUT_ROOT="results/classification_cifar100_dual_peer_consensus_v1/worker1_gpu1" \
    LOG_DIR="results/logs/classification_cifar100_dual_peer_consensus_v1/worker1_gpu1/cases" \
    CASE_SPECS="${DUAL_CASE_SPECS:-uh_cons_ag55:useful_hard_sample_confident:0.24:0.04:0.012:0.020:0.35:0.020:4:6.0:0.0004:0.85:0.80:2:0.50:0.03:0.72:0.92:0.03:4:18:55:0.30:0.55}" \
    bash scripts/paper_rerun/run_classification_cifar100_dual_peer_consensus_v1.sh

echo "[worker1_cifar100_scheduled_then_dual_v1_wait_gpu1] all jobs finished"
