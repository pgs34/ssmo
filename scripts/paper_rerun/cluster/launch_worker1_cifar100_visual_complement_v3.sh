#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_visual_complement_v3/worker1}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker1/cifar100_visual_complement_v3" \
  "$LOG_DIR/classification_gpu${CLASSIFICATION_GPU:-1}.log" \
  env \
    GPU="${CLASSIFICATION_GPU:-1}" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_ROOT="results/classification_cifar100_visual_complement_v3" \
    bash scripts/paper_rerun/run_classification_cifar100_visual_complement_v3.sh

echo "[worker1_cifar100_visual_complement_v3] job finished"
