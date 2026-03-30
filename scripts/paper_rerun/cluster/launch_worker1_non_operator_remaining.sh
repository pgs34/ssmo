#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker1}"
mkdir -p "$LOG_DIR"

run_cifar10() {
  CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU0:-0}" \
  GPU="${CLASSIFICATION_GPU0:-0}" \
  DATASETS="cifar10" \
  OUTPUT_ROOT="${CLASSIFICATION_CIFAR10_OUTPUT_ROOT:-results/classification_ssml_topk_sweep_cifar10_v3}" \
  CIFAR10_TOPKS="${CLASSIFICATION_CIFAR10_TOPKS:-0.1 0.2 0.3}" \
  CIFAR100_TOPKS="" \
  SSML_SUPERVISED_HOTSPOT_ALPHA="${CLASSIFICATION_HOTSPOT_ALPHA:-1.0}" \
  run_logged_job \
    "worker1/classification_cifar10_gpu${CLASSIFICATION_GPU0:-0}" \
    "$LOG_DIR/classification_cifar10_gpu${CLASSIFICATION_GPU0:-0}.log" \
    bash scripts/paper_rerun/run_classification_ssml_topk_sweep.sh
}

run_cifar100() {
  CUDA_VISIBLE_DEVICES="${CLASSIFICATION_GPU1:-1}" \
  GPU="${CLASSIFICATION_GPU1:-1}" \
  DATASETS="cifar100" \
  OUTPUT_ROOT="${CLASSIFICATION_CIFAR100_OUTPUT_ROOT:-results/classification_ssml_topk_sweep_cifar100_v3}" \
  CIFAR10_TOPKS="" \
  CIFAR100_TOPKS="${CLASSIFICATION_CIFAR100_TOPKS:-0.01 0.02 0.05}" \
  SSML_SUPERVISED_HOTSPOT_ALPHA="${CLASSIFICATION_HOTSPOT_ALPHA:-1.0}" \
  run_logged_job \
    "worker1/classification_cifar100_gpu${CLASSIFICATION_GPU1:-1}" \
    "$LOG_DIR/classification_cifar100_gpu${CLASSIFICATION_GPU1:-1}.log" \
    bash scripts/paper_rerun/run_classification_ssml_topk_sweep.sh
}

run_cifar10 &
PID0=$!
run_cifar100 &
PID1=$!

echo "[worker1_non_operator_remaining] started classification cifar10 pid=$PID0 gpu=${CLASSIFICATION_GPU0:-0}"
echo "[worker1_non_operator_remaining] started classification cifar100 pid=$PID1 gpu=${CLASSIFICATION_GPU1:-1}"

wait "$PID0"
wait "$PID1"
echo "[worker1_non_operator_remaining] all jobs finished"
