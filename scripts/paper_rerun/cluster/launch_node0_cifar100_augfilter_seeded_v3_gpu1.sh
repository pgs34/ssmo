#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_cifar100_augfilter_seeded_v3_gpu1.lock"
flock -n 9 || {
  echo "[node0_cifar100_augfilter_seeded_v3_gpu1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_augfilter_seeded_v3/node0}"
mkdir -p "$LOG_DIR"

CASE_SPECS="${CASE_SPECS:-pcu_pb22_thr37_gap18_augmin73_augmax91_agap02:peer_confident_student_uncertain:0.22:0.05:0.012:0.018:0.37:0.018:6:7.0:0.0004:0.84:0.90:2:0.50:0.02:0.73:0.91:0.02 pcu_pb16_thr42_gap22_augmin76_augmax92_agap04:peer_confident_student_uncertain:0.16:0.05:0.011:0.022:0.42:0.022:6:7.5:0.0005:0.80:0.92:2:0.50:0.02:0.76:0.92:0.04}"

run_logged_job \
  "node0/cifar100_augfilter_seeded_v3_gpu1" \
  "$LOG_DIR/classification_gpu1_node0_gpu1.log" \
  env \
    GPU="${CLASSIFICATION_GPU1:-1}" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    BATCH_SIZE="${BATCH_SIZE_4090:-224}" \
    NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
    DOWNLOAD="${DOWNLOAD:-0}" \
    OUTPUT_ROOT="results/classification_cifar100_augfilter_seeded_v3/node0_gpu1" \
    LOG_DIR="$LOG_DIR/node0_gpu1" \
    CASE_SPECS="$CASE_SPECS" \
    bash scripts/paper_rerun/run_classification_cifar100_augfilter_seeded_v3.sh

echo "[node0_cifar100_augfilter_seeded_v3_gpu1] job finished"
