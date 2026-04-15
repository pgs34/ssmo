#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker1_cifar100_strict128_dual_v2_wait_gpu1.lock"
flock -n 9 || {
  echo "[worker1_cifar100_strict128_dual_v2_wait_gpu1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_strict128_dual_v2/worker1_gpu1}"
mkdir -p "$LOG_DIR"

POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-60}"
FREE_GPU_MAX_UTIL="${FREE_GPU_MAX_UTIL:-10}"
FREE_GPU_MAX_MEM_MIB="${FREE_GPU_MAX_MEM_MIB:-512}"
TARGET_GPU="${TARGET_GPU:-1}"
CASE_SPEC="${CASE_SPEC:-pcu_cons_ag55:peer_confident_student_uncertain:0.20:0.05:0.012:0.020:0.38:0.020:5:7.0:0.0004:0.82:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:55:0.30:0.55}"

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
  echo "[$(timestamp)] [worker1_cifar100_strict128_dual_v2_wait_gpu1] waiting for gpu=$TARGET_GPU util<=$FREE_GPU_MAX_UTIL mem<=$FREE_GPU_MAX_MEM_MIB MiB"
  sleep "$POLL_INTERVAL_SEC"
done

run_logged_job \
  "worker1/cifar100_strict128_dual_v2_smoke_gpu${TARGET_GPU}" \
  "$LOG_DIR/smoke.log" \
  env \
    GPU="$TARGET_GPU" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="0" \
    EPOCHS="1" \
    BATCH_SIZE="${BATCH_SIZE_2080TI:-128}" \
    NUM_WORKERS="${NUM_WORKERS_2080TI:-4}" \
    OUTPUT_ROOT="results/classification_cifar100_strict128_dual_v2/smoke_worker1_gpu1" \
    LOG_DIR="results/logs/classification_cifar100_strict128_dual_v2/smoke_worker1_gpu1/cases" \
    PROTOCOL_ID="strict128_smoke" \
    HARDWARE_PROFILE="rtx2080ti" \
    CASE_SPECS="$CASE_SPEC" \
    bash scripts/paper_rerun/run_classification_cifar100_dual_peer_consensus_strict128_v2.sh

run_logged_job \
  "worker1/cifar100_strict128_dual_v2_gpu${TARGET_GPU}" \
  "$LOG_DIR/full.log" \
  env \
    GPU="$TARGET_GPU" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    EPOCHS="${EPOCHS:-100}" \
    BATCH_SIZE="${BATCH_SIZE_2080TI:-128}" \
    NUM_WORKERS="${NUM_WORKERS_2080TI:-4}" \
    OUTPUT_ROOT="results/classification_cifar100_dual_peer_consensus_strict128_v2/worker1_gpu1" \
    LOG_DIR="results/logs/classification_cifar100_dual_peer_consensus_strict128_v2/worker1_gpu1/cases" \
    PROTOCOL_ID="strict128" \
    HARDWARE_PROFILE="rtx2080ti" \
    CASE_SPECS="$CASE_SPEC" \
    bash scripts/paper_rerun/run_classification_cifar100_dual_peer_consensus_strict128_v2.sh

python scripts/paper_rerun/family_result_report.py \
  --run-root results/classification_cifar100_dual_peer_consensus_strict128_v2 \
  --metric-key best_val_acc \
  --expected-seeds 0,1,2 \
  --higher-is-better \
  --current-best 0.536567 \
  --strongest-baseline 0.545067 | tee "$LOG_DIR/family_report.json"

echo "[worker1_cifar100_strict128_dual_v2_wait_gpu1] finished"
