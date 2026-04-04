#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_cifar100_augfilter_seeded_v3_gpu0.lock"
flock -n 9 || {
  echo "[node0_cifar100_augfilter_seeded_v3_gpu0] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_augfilter_seeded_v3/node0}"
mkdir -p "$LOG_DIR"

CASE_SPECS="${CASE_SPECS:-uh_pb30_thr32_gap16_augmin70_augmax90_agap02:useful_hard_sample:0.30:0.06:0.012:0.016:0.32:0.016:4:6.0:0.0004:0.88:0.90:2:0.50:0.03:0.70:0.90:0.02 uhconf_pb18_thr37_gap24_augmin78_augmax93_agap03:useful_hard_sample_confident:0.18:0.04:0.010:0.024:0.37:0.024:5:6.5:0.0004:0.84:0.92:2:0.50:0.03:0.78:0.93:0.03}"

run_logged_job \
  "node0/cifar100_augfilter_seeded_v3_gpu0" \
  "$LOG_DIR/classification_gpu0_node0_gpu0.log" \
  env \
    GPU="${CLASSIFICATION_GPU0:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    BATCH_SIZE="${BATCH_SIZE_4090:-224}" \
    NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
    DOWNLOAD="${DOWNLOAD:-0}" \
    OUTPUT_ROOT="results/classification_cifar100_augfilter_seeded_v3/node0_gpu0" \
    LOG_DIR="$LOG_DIR/node0_gpu0" \
    CASE_SPECS="$CASE_SPECS" \
    bash scripts/paper_rerun/run_classification_cifar100_augfilter_seeded_v3.sh

echo "[node0_cifar100_augfilter_seeded_v3_gpu0] job finished"
