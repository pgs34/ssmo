#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_teacher_safe_v2/worker3}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker3/cifar100_teacher_safe_v2" \
  "$LOG_DIR/classification_gpu${CLASSIFICATION_GPU:-0}.log" \
  env \
    GPU="${CLASSIFICATION_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_ROOT="results/classification_cifar100_teacher_safe_v2" \
    bash scripts/paper_rerun/run_classification_cifar100_teacher_safe_v2.sh

echo "[worker3_cifar100_teacher_safe_v2] job finished"
