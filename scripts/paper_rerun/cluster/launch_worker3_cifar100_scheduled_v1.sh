#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker3_cifar100_scheduled_v1.lock"
flock -n 9 || {
  echo "[worker3_cifar100_scheduled_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_infinite_loop_v1/worker3_gpu0}"
mkdir -p "$LOG_DIR"

GPU="${TARGET_GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"

run_logged_job \
  "worker3/cifar100_scheduled_complement_v1_gpu${GPU}" \
  "$LOG_DIR/scheduled_complement.log" \
  env \
    GPU="$GPU" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="$SEEDS" \
    BATCH_SIZE="${BATCH_SIZE_3090TI:-192}" \
    NUM_WORKERS="${NUM_WORKERS_3090TI:-4}" \
    OUTPUT_ROOT="results/classification_cifar100_scheduled_complement_v1/worker3_gpu0" \
    LOG_DIR="results/logs/classification_cifar100_scheduled_complement_v1/worker3_gpu0/cases" \
    CASE_SPECS="${SCHEDULED_CASE_SPECS:-uh_sched_df10_x05_r30_60:useful_hard_sample_confident:0.24:0.04:0.012:0.020:0.35:0.020:4:6.0:0.0004:0.85:0.80:2:0.50:0.03:0.72:0.92:0.03:4:18:55:0.30:0.10:0.90:0.50:30:60}" \
    bash scripts/paper_rerun/run_classification_cifar100_scheduled_complement_v1.sh

echo "[worker3_cifar100_scheduled_v1] finished"
