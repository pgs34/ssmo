#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker1_electricity_handoff_router_seeded_v1.lock"
flock -n 9 || {
  echo "[worker1_electricity_handoff_router_seeded_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/time_series_electricity_handoff_router_seeded_v1/worker1}"
mkdir -p "$LOG_DIR"

run_gpu() {
  local gpu="$1"
  local suffix="$2"
  local case_specs="$3"
  run_logged_job \
    "worker1/electricity_handoff_router_seeded_v1_gpu${gpu}" \
    "$LOG_DIR/time_series_gpu${gpu}_${suffix}.log" \
    env \
      GPU="$gpu" \
      DEVICE="${DEVICE:-cuda}" \
      SEEDS="${SEEDS:-0 1 2}" \
      EPOCHS="${EPOCHS:-70}" \
      BATCH_SIZE="${BATCH_SIZE_2080TI:-64}" \
      NUM_WORKERS="${NUM_WORKERS_2080TI:-4}" \
      OUTPUT_ROOT="results/time_series_electricity_handoff_router_seeded_v1/${suffix}" \
      LOG_DIR="$LOG_DIR/${suffix}" \
      CASE_SPECS="$case_specs" \
      bash scripts/paper_rerun/run_time_series_electricity_handoff_router_seeded_v1.sh
}

GPU0_CASES="${GPU0_CASES:-elec_handoff_q75_b18:0.10:0.0008:64:0.00:10:10:18:18:24:0.05:0.75:0.0005:3:0.18:12:18:0.0001:22:0.70:1.00:0.00 elec_handoff_q70_b20:0.12:0.0010:64:0.00:8:8:16:16:22:0.05:0.70:0.0003:3:0.20:12:18:0.0001:20:0.65:1.05:0.00}"
GPU1_CASES="${GPU1_CASES:-elec_handoff_q80_b15:0.08:0.0008:48:0.00:12:12:20:20:26:0.08:0.80:0.0008:5:0.15:12:20:0.0001:24:0.75:0.95:0.00}"

run_gpu "${TIME_SERIES_GPU0:-0}" "worker1_gpu0" "$GPU0_CASES" &
PID0=$!
run_gpu "${TIME_SERIES_GPU1:-1}" "worker1_gpu1" "$GPU1_CASES" &
PID1=$!

echo "[worker1_electricity_handoff_router_seeded_v1] started gpu${TIME_SERIES_GPU0:-0} pid=$PID0"
echo "[worker1_electricity_handoff_router_seeded_v1] started gpu${TIME_SERIES_GPU1:-1} pid=$PID1"

wait "$PID0"
wait "$PID1"
echo "[worker1_electricity_handoff_router_seeded_v1] all jobs finished"
