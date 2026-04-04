#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_cifar100_augfilter_seeded_v3.lock"
flock -n 9 || {
  echo "[node0_cifar100_augfilter_seeded_v3] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_augfilter_seeded_v3/node0}"
mkdir -p "$LOG_DIR"

run_gpu() {
  local gpu="$1"
  local suffix="$2"
  local case_specs="$3"
  shift 3
  run_logged_job \
    "node0/cifar100_augfilter_seeded_v3_gpu${gpu}" \
    "$LOG_DIR/classification_gpu${gpu}_${suffix}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      SEEDS="${SEEDS:-0 1 2}" \
      BATCH_SIZE="${BATCH_SIZE_4090:-224}" \
      NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
      DOWNLOAD="${DOWNLOAD:-0}" \
      OUTPUT_ROOT="results/classification_cifar100_augfilter_seeded_v3/${suffix}" \
      LOG_DIR="$LOG_DIR/${suffix}" \
      CASE_SPECS="$case_specs" \
      "$@" \
      bash scripts/paper_rerun/run_classification_cifar100_augfilter_seeded_v3.sh
}

GPU0_CASES="${GPU0_CASES:-uh_pb30_thr32_gap16_augmin70_augmax90_agap02:useful_hard_sample:0.30:0.06:0.012:0.016:0.32:0.016:4:6.0:0.0004:0.88:0.90:2:0.50:0.03:0.70:0.90:0.02 uhconf_pb18_thr37_gap24_augmin78_augmax93_agap03:useful_hard_sample_confident:0.18:0.04:0.010:0.024:0.37:0.024:5:6.5:0.0004:0.84:0.92:2:0.50:0.03:0.78:0.93:0.03}"
GPU1_CASES="${GPU1_CASES:-pcu_pb22_thr37_gap18_augmin73_augmax91_agap02:peer_confident_student_uncertain:0.22:0.05:0.012:0.018:0.37:0.018:6:7.0:0.0004:0.84:0.90:2:0.50:0.02:0.73:0.91:0.02 pcu_pb16_thr42_gap22_augmin76_augmax92_agap04:peer_confident_student_uncertain:0.16:0.05:0.011:0.022:0.42:0.022:6:7.5:0.0005:0.80:0.92:2:0.50:0.02:0.76:0.92:0.04}"

run_gpu "${CLASSIFICATION_GPU0:-0}" "node0_gpu0" "$GPU0_CASES" &
PID0=$!
run_gpu "${CLASSIFICATION_GPU1:-1}" "node0_gpu1" "$GPU1_CASES" &
PID1=$!

echo "[node0_cifar100_augfilter_seeded_v3] started gpu${CLASSIFICATION_GPU0:-0} pid=$PID0"
echo "[node0_cifar100_augfilter_seeded_v3] started gpu${CLASSIFICATION_GPU1:-1} pid=$PID1"

wait "$PID0"
wait "$PID1"
echo "[node0_cifar100_augfilter_seeded_v3] all jobs finished"
