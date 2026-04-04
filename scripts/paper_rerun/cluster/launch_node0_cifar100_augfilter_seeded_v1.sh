#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_cifar100_augfilter_seeded_v1.lock"
flock -n 9 || {
  echo "[node0_cifar100_augfilter_seeded_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_augfilter_seeded_v1/node0}"
mkdir -p "$LOG_DIR"

run_gpu() {
  local gpu="$1"
  local suffix="$2"
  local case_specs="$3"
  shift 3
  run_logged_job \
    "node0/cifar100_augfilter_seeded_v1_gpu${gpu}" \
    "$LOG_DIR/classification_gpu${gpu}_${suffix}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      SEEDS="${SEEDS:-0 1 2}" \
      BATCH_SIZE="${BATCH_SIZE_4090:-224}" \
      NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
      DOWNLOAD="${DOWNLOAD:-0}" \
      OUTPUT_ROOT="results/classification_cifar100_augfilter_seeded_v1/${suffix}" \
      LOG_DIR="$LOG_DIR/${suffix}" \
      CASE_SPECS="$case_specs" \
      "$@" \
      bash scripts/paper_rerun/run_classification_cifar100_augfilter_seeded_v1.sh
}

GPU0_CASES="${GPU0_CASES:-uhconf_pb24_thr35_gap20_augmin72_augmax92_agap03:useful_hard_sample_confident:0.24:0.04:0.012:0.020:0.35:0.020:4:6.0:0.0004:0.85:0.80:2:0.50:0.03:0.72:0.92:0.03 uhconf_pb24_thr35_gap20_augmin74_augmax90_agap03:useful_hard_sample_confident:0.24:0.04:0.012:0.020:0.35:0.020:4:6.0:0.0004:0.85:0.90:2:0.50:0.03:0.74:0.90:0.03}"
GPU1_CASES="${GPU1_CASES:-pcu_pb20_thr40_gap20_augmin75_augmax90_agap04:peer_confident_student_uncertain:0.20:0.05:0.012:0.020:0.40:0.020:5:7.0:0.0004:0.80:0.90:2:0.50:0.02:0.75:0.90:0.04 pcu_pb20_thr38_gap20_augmin72_augmax90_agap03:peer_confident_student_uncertain:0.20:0.05:0.012:0.020:0.38:0.020:5:7.0:0.0004:0.82:0.90:2:0.50:0.02:0.72:0.90:0.03}"

run_gpu "${CLASSIFICATION_GPU0:-0}" "node0_gpu0" "$GPU0_CASES" &
PID0=$!
run_gpu "${CLASSIFICATION_GPU1:-1}" "node0_gpu1" "$GPU1_CASES" &
PID1=$!

echo "[node0_cifar100_augfilter_seeded_v1] started gpu${CLASSIFICATION_GPU0:-0} pid=$PID0"
echo "[node0_cifar100_augfilter_seeded_v1] started gpu${CLASSIFICATION_GPU1:-1} pid=$PID1"

wait "$PID0"
wait "$PID1"
echo "[node0_cifar100_augfilter_seeded_v1] all jobs finished"
