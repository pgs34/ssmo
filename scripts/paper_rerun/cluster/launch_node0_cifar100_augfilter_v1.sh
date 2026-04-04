#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_augfilter_v1/node0}"
mkdir -p "$LOG_DIR"

run_gpu() {
  local gpu="$1"
  local suffix="$2"
  shift 2
  run_logged_job \
    "node0/cifar100_augfilter_v1_gpu${gpu}" \
    "$LOG_DIR/classification_gpu${gpu}_${suffix}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      OUTPUT_ROOT="results/classification_cifar100_augfilter_v1/${suffix}" \
      "$@" \
      bash scripts/paper_rerun/run_classification_cifar100_augfilter_v1.sh
}

run_gpu "${CLASSIFICATION_GPU0:-0}" "node0_gpu0" &
PID0=$!
run_gpu "${CLASSIFICATION_GPU1:-1}" "node0_gpu1" &
PID1=$!

echo "[node0_cifar100_augfilter_v1] started gpu${CLASSIFICATION_GPU0:-0} pid=$PID0"
echo "[node0_cifar100_augfilter_v1] started gpu${CLASSIFICATION_GPU1:-1} pid=$PID1"

wait "$PID0"
wait "$PID1"
echo "[node0_cifar100_augfilter_v1] all jobs finished"
