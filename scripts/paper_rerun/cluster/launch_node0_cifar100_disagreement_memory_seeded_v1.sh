#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_cifar100_disagreement_memory_seeded_v1.lock"
flock -n 9 || {
  echo "[node0_cifar100_disagreement_memory_seeded_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_disagreement_memory_seeded_v1/node0}"
mkdir -p "$LOG_DIR"

run_gpu() {
  local gpu="$1"
  local suffix="$2"
  local case_specs="$3"
  run_logged_job \
    "node0/cifar100_disagreement_memory_seeded_v1_gpu${gpu}" \
    "$LOG_DIR/classification_gpu${gpu}_${suffix}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      SEEDS="${SEEDS:-0 1 2}" \
      BATCH_SIZE="${BATCH_SIZE_4090:-128}" \
      NUM_WORKERS="${NUM_WORKERS_4090:-6}" \
      DOWNLOAD="${DOWNLOAD:-0}" \
      OUTPUT_ROOT="results/classification_cifar100_disagreement_memory_seeded_v1/${suffix}" \
      LOG_DIR="$LOG_DIR/${suffix}" \
      CASE_SPECS="$case_specs" \
      bash scripts/paper_rerun/run_classification_cifar100_disagreement_memory_seeded_v1.sh
}

GPU0_CASES="${GPU0_CASES:-pcu_mem_df35_m90_x10:peer_confident_student_uncertain:0.18:0.05:0.010:0.020:0.38:0.022:4:7.0:0.0008:0.84:0.50:2:0.02:0.76:0.92:0.03:10:35:100:0.30:0.35:0.90:1.0 pcu_mem_df45_m95_x15:peer_confident_student_uncertain:0.16:0.05:0.009:0.020:0.40:0.024:4:7.0:0.0008:0.82:0.50:2:0.02:0.78:0.92:0.04:12:40:105:0.25:0.45:0.95:1.5}"
GPU1_CASES="${GPU1_CASES:-uh_mem_df35_m90_x10:useful_hard_sample_confident:0.20:0.04:0.012:0.018:0.34:0.020:4:6.0:0.0006:0.86:0.45:2:0.03:0.74:0.94:0.02:8:30:95:0.30:0.35:0.90:1.0 uh_mem_df50_m95_x20:useful_hard_sample_confident:0.18:0.04:0.011:0.020:0.36:0.022:5:6.0:0.0006:0.84:0.45:2:0.03:0.76:0.94:0.03:10:35:100:0.25:0.50:0.95:2.0}"

run_gpu "${CLASSIFICATION_GPU0:-0}" "node0_gpu0" "$GPU0_CASES" &
PID0=$!
run_gpu "${CLASSIFICATION_GPU1:-1}" "node0_gpu1" "$GPU1_CASES" &
PID1=$!

echo "[node0_cifar100_disagreement_memory_seeded_v1] started gpu${CLASSIFICATION_GPU0:-0} pid=$PID0"
echo "[node0_cifar100_disagreement_memory_seeded_v1] started gpu${CLASSIFICATION_GPU1:-1} pid=$PID1"

wait "$PID0"
wait "$PID1"
echo "[node0_cifar100_disagreement_memory_seeded_v1] all jobs finished"
