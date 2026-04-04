#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker1_operator_student_lifts_v2.lock"
flock -n 9 || {
  echo "[worker1_operator_student_lifts_v2] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/operator_student_lifts_v2/worker1}"
mkdir -p "$LOG_DIR"

run_logged_job \
  "worker1/operator_burgers_student_lift_v2_gpu0" \
  "$LOG_DIR/burgers_gpu0.log" \
  env \
    GPU="${BURGERS_GPU:-0}" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_ROOT="${BURGERS_OUTPUT_ROOT:-results/operator_burgers_student_lift_v2}" \
    LOG_DIR="${BURGERS_LOG_DIR:-results/logs/operator_burgers_student_lift_v2}" \
    SEEDS="${SEEDS:-0 1 2}" \
    EPOCHS="${BURGERS_EPOCHS:-140}" \
    BATCH_SIZE="${BURGERS_BATCH_SIZE:-16}" \
    NUM_WORKERS="${BURGERS_NUM_WORKERS:-2}" \
    bash scripts/paper_rerun/run_operator_burgers_student_lift_v2.sh &
PID0=$!

run_logged_job \
  "worker1/operator_darcy_student_lift_v2_gpu1" \
  "$LOG_DIR/darcy_gpu1.log" \
  env \
    GPU="${DARCY_GPU:-1}" \
    DEVICE="${DEVICE:-cuda}" \
    OUTPUT_ROOT="${DARCY_OUTPUT_ROOT:-results/operator_darcy_student_lift_v2}" \
    LOG_DIR="${DARCY_LOG_DIR:-results/logs/operator_darcy_student_lift_v2}" \
    SEEDS="${SEEDS:-0 1 2}" \
    EPOCHS="${DARCY_EPOCHS:-110}" \
    BATCH_SIZE="${DARCY_BATCH_SIZE:-16}" \
    NUM_WORKERS="${DARCY_NUM_WORKERS:-2}" \
    bash scripts/paper_rerun/run_operator_darcy_student_lift_v2.sh &
PID1=$!

echo "[worker1_operator_student_lifts_v2] started burgers gpu${BURGERS_GPU:-0} pid=$PID0"
echo "[worker1_operator_student_lifts_v2] started darcy gpu${DARCY_GPU:-1} pid=$PID1"

wait "$PID0"
wait "$PID1"
echo "[worker1_operator_student_lifts_v2] all jobs finished"
