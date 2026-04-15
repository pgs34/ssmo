#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_cifar100_scheduled_v1_gpu0.lock"
flock -n 9 || {
  echo "[node0_cifar100_scheduled_v1_gpu0] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_infinite_loop_v1/node0_gpu0}"
mkdir -p "$LOG_DIR"

GPU="${TARGET_GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"

run_logged_job \
  "node0/cifar100_scheduled_complement_v1_gpu${GPU}" \
  "$LOG_DIR/scheduled_complement.log" \
  env \
    GPU="$GPU" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="$SEEDS" \
    BATCH_SIZE="${BATCH_SIZE_4090:-224}" \
    NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
    OUTPUT_ROOT="results/classification_cifar100_scheduled_complement_v1/node0_gpu0" \
    LOG_DIR="results/logs/classification_cifar100_scheduled_complement_v1/node0_gpu0/cases" \
    CASE_SPECS="${SCHEDULED_CASE_SPECS:-pcu_sched_df10_x05_r30_60:peer_confident_student_uncertain:0.20:0.05:0.012:0.020:0.38:0.020:5:7.0:0.0004:0.82:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:55:0.30:0.10:0.90:0.50:30:60}" \
    bash scripts/paper_rerun/run_classification_cifar100_scheduled_complement_v1.sh

echo "[node0_cifar100_scheduled_v1_gpu0] finished"
