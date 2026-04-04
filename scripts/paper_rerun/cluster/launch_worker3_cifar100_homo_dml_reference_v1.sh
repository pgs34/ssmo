#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_homo_dml_reference_v1/worker3}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker3/cifar100_homo_dml_reference_v1" \
  "$LOG_DIR/classification_gpu${CLASSIFICATION_GPU:-0}.log" \
  env \
    GPU="${CLASSIFICATION_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_ROOT="results/classification_cifar100_homo_dml_reference_v1" \
    bash scripts/paper_rerun/run_classification_cifar100_homo_dml_reference_v1.sh

echo "[worker3_cifar100_homo_dml_reference_v1] job finished"
