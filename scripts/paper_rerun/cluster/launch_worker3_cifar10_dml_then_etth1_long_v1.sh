#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker3_cifar10_dml_then_etth1_long_v1.lock"
flock -n 9 || {
  echo "[worker3_cifar10_dml_then_etth1_long_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/long_followups/worker3}"
mkdir -p "$LOG_DIR"

GPU="${GPU:-0}"

run_logged_job \
  "worker3/cifar10_homo_dml_long_v1" \
  "$LOG_DIR/cifar10_homo_dml_long_v1.log" \
  env \
    GPU="$GPU" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_ROOT="${CIFAR10_DML_OUTPUT_ROOT:-results/classification_cifar10_homo_dml_long_v1}" \
    SUMMARY_PLOT_ROOT="${CIFAR10_DML_PLOT_ROOT:-results/plots/classification_cifar10_homo_dml_long_v1}" \
    REFRESH_TOP_LEVEL="${CIFAR10_DML_REFRESH_TOP_LEVEL:-0}" \
    SEEDS="${CIFAR10_DML_SEEDS:-0 1 2}" \
    EPOCHS="${CIFAR10_DML_EPOCHS:-100}" \
    BATCH_SIZE="${CIFAR10_DML_BATCH_SIZE:-256}" \
    NUM_WORKERS="${CIFAR10_DML_NUM_WORKERS:-4}" \
    bash scripts/paper_rerun/run_classification_cifar10_homo_dml_long_v1.sh

run_logged_job \
  "worker3/etth1_correction_only_long_v6" \
  "$LOG_DIR/etth1_correction_only_long_v6.log" \
  env \
    GPU="$GPU" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_ROOT="${ETTH1_CORRLONG_OUTPUT_ROOT:-results/time_series_etth1_correction_only_long_v6}" \
    LOG_DIR="${ETTH1_CORRLONG_LOG_DIR:-results/logs/time_series_etth1_correction_only_long_v6}" \
    REFRESH_TOP_LEVEL="${ETTH1_CORRLONG_REFRESH_TOP_LEVEL:-0}" \
    SEEDS="${ETTH1_CORRLONG_SEEDS:-0 1 2}" \
    EPOCHS="${ETTH1_CORRLONG_EPOCHS:-90}" \
    BATCH_SIZE="${ETTH1_CORRLONG_BATCH_SIZE:-768}" \
    NUM_WORKERS="${ETTH1_CORRLONG_NUM_WORKERS:-4}" \
    bash scripts/paper_rerun/run_time_series_etth1_correction_only_long_v6.sh

echo "[worker3_cifar10_dml_then_etth1_long_v1] job finished"

