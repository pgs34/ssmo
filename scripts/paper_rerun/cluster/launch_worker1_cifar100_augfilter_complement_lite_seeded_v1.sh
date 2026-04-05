#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker1_cifar100_augfilter_complement_lite_seeded_v1.lock"
flock -n 9 || {
  echo "[worker1_cifar100_augfilter_complement_lite_seeded_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_augfilter_complement_lite_seeded_v1/worker1}"
mkdir -p "$LOG_DIR"

run_gpu() {
  local gpu="$1"
  local suffix="$2"
  local case_specs="$3"
  run_logged_job \
    "worker1/cifar100_augfilter_complement_lite_seeded_v1_gpu${gpu}" \
    "$LOG_DIR/classification_gpu${gpu}_${suffix}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      SEEDS="${SEEDS:-0 1 2}" \
      EPOCHS="${EPOCHS:-100}" \
      BATCH_SIZE="${BATCH_SIZE_2080TI:-128}" \
      NUM_WORKERS="${NUM_WORKERS_2080TI:-4}" \
      OUTPUT_ROOT="results/classification_cifar100_augfilter_complement_lite_seeded_v1/${suffix}" \
      LOG_DIR="$LOG_DIR/${suffix}" \
      CASE_SPECS="$case_specs" \
      bash scripts/paper_rerun/run_classification_cifar100_augfilter_complement_lite_seeded_v1.sh
}

GPU0_CASES="${GPU0_CASES:-pcu_lite_df10_m80_x05:peer_confident_student_uncertain:0.20:0.05:0.012:0.020:0.38:0.020:5:7.0:0.0004:0.82:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:55:0.30:0.10:0.80:0.50 pcu_lite_df15_m85_x08:peer_confident_student_uncertain:0.22:0.05:0.011:0.020:0.39:0.022:5:7.0:0.0004:0.84:0.85:2:0.50:0.02:0.74:0.91:0.03:6:22:65:0.25:0.15:0.85:0.80}"
GPU1_CASES="${GPU1_CASES:-uh_lite_df10_m80_x05:useful_hard_sample_confident:0.24:0.04:0.012:0.020:0.35:0.020:4:6.0:0.0004:0.85:0.80:2:0.50:0.03:0.72:0.92:0.03:4:18:55:0.30:0.10:0.80:0.50 uh_lite_df15_m85_x08:useful_hard_sample_confident:0.22:0.04:0.011:0.020:0.36:0.020:4:6.0:0.0004:0.84:0.82:2:0.50:0.03:0.74:0.92:0.03:6:22:65:0.25:0.15:0.85:0.80}"

run_gpu "${CLASSIFICATION_GPU0:-0}" "worker1_gpu0" "$GPU0_CASES" &
PID0=$!
run_gpu "${CLASSIFICATION_GPU1:-1}" "worker1_gpu1" "$GPU1_CASES" &
PID1=$!

echo "[worker1_cifar100_augfilter_complement_lite_seeded_v1] started gpu${CLASSIFICATION_GPU0:-0} pid=$PID0"
echo "[worker1_cifar100_augfilter_complement_lite_seeded_v1] started gpu${CLASSIFICATION_GPU1:-1} pid=$PID1"

wait "$PID0"
wait "$PID1"
echo "[worker1_cifar100_augfilter_complement_lite_seeded_v1] all jobs finished"
