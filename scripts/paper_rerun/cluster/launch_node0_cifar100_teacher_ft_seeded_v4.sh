#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_cifar100_teacher_ft_seeded_v4.lock"
flock -n 9 || {
  echo "[node0_cifar100_teacher_ft_seeded_v4] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_teacher_ft_seeded_v4/node0}"
mkdir -p "$LOG_DIR"

run_gpu() {
  local gpu="$1"
  local suffix="$2"
  local case_specs="$3"
  run_logged_job \
    "node0/cifar100_teacher_ft_seeded_v4_gpu${gpu}" \
    "$LOG_DIR/classification_gpu${gpu}_${suffix}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      SEEDS="${SEEDS:-0 1 2}" \
      BATCH_SIZE="${BATCH_SIZE_4090:-128}" \
      NUM_WORKERS="${NUM_WORKERS_4090:-6}" \
      DOWNLOAD="${DOWNLOAD:-0}" \
      OUTPUT_ROOT="results/classification_cifar100_teacher_ft_seeded_v4/${suffix}" \
      LOG_DIR="$LOG_DIR/${suffix}" \
      CASE_SPECS="$case_specs" \
      bash scripts/paper_rerun/run_classification_cifar100_teacher_ft_seeded_v4.sh
}

GPU0_CASES="${GPU0_CASES:-pcu_late_pb16_thr40_gap24_aug78_92_ag03:peer_confident_student_uncertain:0.16:0.05:0.010:0.024:0.40:0.024:4:7.0:0.0008:0.84:0.50:2:0.02:0.78:0.92:0.03:10:35:100:0.30 pcu_late_pb14_thr42_gap26_aug80_92_ag04:peer_confident_student_uncertain:0.14:0.05:0.010:0.026:0.42:0.026:4:7.0:0.0008:0.82:0.50:2:0.02:0.80:0.92:0.04:12:40:105:0.25}"
GPU1_CASES="${GPU1_CASES:-uh_late_pb18_thr36_gap20_aug76_94_ag02:useful_hard_sample_confident:0.18:0.04:0.012:0.020:0.36:0.020:4:6.0:0.0006:0.86:0.45:2:0.03:0.76:0.94:0.02:8:30:95:0.30 uh_late_pb16_thr38_gap22_aug78_94_ag03:useful_hard_sample_confident:0.16:0.04:0.011:0.022:0.38:0.022:4:6.0:0.0006:0.84:0.45:2:0.03:0.78:0.94:0.03:10:35:100:0.25}"

run_gpu "${CLASSIFICATION_GPU0:-0}" "node0_gpu0" "$GPU0_CASES" &
PID0=$!
run_gpu "${CLASSIFICATION_GPU1:-1}" "node0_gpu1" "$GPU1_CASES" &
PID1=$!

echo "[node0_cifar100_teacher_ft_seeded_v4] started gpu${CLASSIFICATION_GPU0:-0} pid=$PID0"
echo "[node0_cifar100_teacher_ft_seeded_v4] started gpu${CLASSIFICATION_GPU1:-1} pid=$PID1"

wait "$PID0"
wait "$PID1"
echo "[node0_cifar100_teacher_ft_seeded_v4] all jobs finished"
