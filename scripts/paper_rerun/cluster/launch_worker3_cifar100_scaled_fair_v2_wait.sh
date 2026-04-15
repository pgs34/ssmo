#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker3_cifar100_scaled_fair_v2_wait.lock"
flock -n 9 || {
  echo "[worker3_cifar100_scaled_fair_v2_wait] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_scaled_fair_v2/worker3}"
CONFIG_DIR="${CONFIG_DIR:-results/classification_cifar100_scaled_fair_v2/config}"
mkdir -p "$LOG_DIR" "$CONFIG_DIR"

POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-60}"
FREE_GPU_MAX_UTIL="${FREE_GPU_MAX_UTIL:-10}"
FREE_GPU_MAX_MEM_MIB="${FREE_GPU_MAX_MEM_MIB:-512}"
TARGET_GPU="${TARGET_GPU:-0}"
CASE_SPEC="${CASE_SPEC:-oxtra42_thr42_gap1_pc18_aug125:0.42:0.02:0.018:0.000:0.42:0.01:18:6.0:0.0002:0.93:1.25:3:0.50:0.04:1:12:36:0.25}"

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

while [[ ! -f "$CONFIG_DIR/fair_batch.env" ]]; do
  echo "[$(timestamp)] [worker3_cifar100_scaled_fair_v2_wait] waiting for fair_batch.env"
  sleep 15
done

# shellcheck disable=SC1090
source "$CONFIG_DIR/fair_batch.env"
echo "[worker3_cifar100_scaled_fair_v2_wait] sourced FAIR_BATCH=$FAIR_BATCH"

while ! gpu_is_available; do
  echo "[$(timestamp)] [worker3_cifar100_scaled_fair_v2_wait] waiting for gpu=$TARGET_GPU util<=$FREE_GPU_MAX_UTIL mem<=$FREE_GPU_MAX_MEM_MIB MiB"
  sleep "$POLL_INTERVAL_SEC"
done

if run_logged_job \
  "worker3/cifar100_scaled_fair_v2_smoke_gpu${TARGET_GPU}" \
  "$LOG_DIR/smoke.log" \
  env \
    GPU="$TARGET_GPU" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="0" \
    EPOCHS="1" \
    BATCH_SIZE="$FAIR_BATCH" \
    NUM_WORKERS="${NUM_WORKERS_3090TI:-4}" \
    OUTPUT_ROOT="results/classification_cifar100_overbatch_reproduce_fair_v2/smoke_worker3_gpu0" \
    LOG_DIR="results/logs/classification_cifar100_overbatch_reproduce_fair_v2/smoke_worker3_gpu0/cases" \
    PROTOCOL_ID="${PROTOCOL_ID}_smoke" \
    HARDWARE_PROFILE="rtx3090ti" \
    CASE_SPECS="$CASE_SPEC" \
    bash scripts/paper_rerun/run_classification_cifar100_overbatch_reproduce_fair_v2.sh; then
  touch "$CONFIG_DIR/worker3_ready.flag"
else
  touch "$CONFIG_DIR/worker3_skip.flag"
  echo "[worker3_cifar100_scaled_fair_v2_wait] smoke failed for FAIR_BATCH=$FAIR_BATCH"
  exit 0
fi

run_logged_job \
  "worker3/cifar100_scaled_fair_v2_gpu${TARGET_GPU}" \
  "$LOG_DIR/full.log" \
  env \
    GPU="$TARGET_GPU" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    EPOCHS="${EPOCHS:-100}" \
    BATCH_SIZE="$FAIR_BATCH" \
    NUM_WORKERS="${NUM_WORKERS_3090TI:-4}" \
    OUTPUT_ROOT="results/classification_cifar100_overbatch_reproduce_fair_v2/worker3_gpu0" \
    LOG_DIR="results/logs/classification_cifar100_overbatch_reproduce_fair_v2/worker3_gpu0/cases" \
    PROTOCOL_ID="$PROTOCOL_ID" \
    HARDWARE_PROFILE="rtx3090ti" \
    CASE_SPECS="$CASE_SPEC" \
    bash scripts/paper_rerun/run_classification_cifar100_overbatch_reproduce_fair_v2.sh

python scripts/paper_rerun/family_result_report.py \
  --run-root results/classification_cifar100_overbatch_reproduce_fair_v2 \
  --metric-key best_val_acc \
  --expected-seeds 0,1,2 \
  --higher-is-better \
  --current-best 0.536567 \
  --strongest-baseline 0.545067 | tee "$LOG_DIR/family_report.json"

echo "[worker3_cifar100_scaled_fair_v2_wait] finished"
