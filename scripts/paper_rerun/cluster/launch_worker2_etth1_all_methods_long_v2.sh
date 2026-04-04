#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/non_operator_remaining/worker2_etth1_all_methods_long_v2}"
mkdir -p "$LOG_DIR"

POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-60}"
FREE_GPU_MAX_UTIL="${FREE_GPU_MAX_UTIL:-10}"
FREE_GPU_MAX_MEM_MIB="${FREE_GPU_MAX_MEM_MIB:-4096}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

gpu_is_available() {
  local target_gpu="$1"
  local idx util mem
  while IFS=',' read -r idx util mem; do
    idx="${idx//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    mem="${mem//[[:space:]]/}"
    if [[ "$idx" != "$target_gpu" ]]; then
      continue
    fi
    if (( util <= FREE_GPU_MAX_UTIL && mem <= FREE_GPU_MAX_MEM_MIB )); then
      return 0
    fi
    return 1
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
  return 1
}

wait_for_gpu() {
  local requested_gpu="${TIME_SERIES_GPU:-}"
  local idx util mem

  while true; do
    if [[ -n "$requested_gpu" ]]; then
      if gpu_is_available "$requested_gpu"; then
        printf '%s\n' "$requested_gpu"
        return 0
      fi
      echo "[$(timestamp)] [worker2_etth1_all_methods_long_v2] waiting for gpu=$requested_gpu util<=$FREE_GPU_MAX_UTIL mem<=$FREE_GPU_MAX_MEM_MIB MiB" >&2
      sleep "$POLL_INTERVAL_SEC"
      continue
    fi

    while IFS=',' read -r idx util mem; do
      idx="${idx//[[:space:]]/}"
      util="${util//[[:space:]]/}"
      mem="${mem//[[:space:]]/}"
      if (( util <= FREE_GPU_MAX_UTIL && mem <= FREE_GPU_MAX_MEM_MIB )); then
        printf '%s\n' "$idx"
        return 0
      fi
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)

    echo "[$(timestamp)] [worker2_etth1_all_methods_long_v2] waiting for any gpu util<=$FREE_GPU_MAX_UTIL mem<=$FREE_GPU_MAX_MEM_MIB MiB" >&2
    sleep "$POLL_INTERVAL_SEC"
  done
}

SELECTED_GPU="$(wait_for_gpu)"
echo "[$(timestamp)] [worker2_etth1_all_methods_long_v2] selected gpu=$SELECTED_GPU"

run_logged_job \
  "worker2/etth1_all_methods_long_v2" \
  "$LOG_DIR/etth1_all_methods_long_v2_gpu${SELECTED_GPU}.log" \
  env \
    CUDA_VISIBLE_DEVICES="$SELECTED_GPU" \
    GPU="$SELECTED_GPU" \
    OUTPUT_ROOT="${ETTH1_ALL_METHODS_LONG_V2_OUTPUT_ROOT:-results/time_series_etth1_all_methods_long_v2}" \
    SUMMARY_PLOT_ROOT="${ETTH1_ALL_METHODS_LONG_V2_PLOT_ROOT:-results/plots/time_series_etth1_all_methods_long_v2}" \
    SEEDS="${ETTH1_ALL_METHODS_LONG_V2_SEEDS:-0 1 2 3 4 5}" \
    EPOCHS="${ETTH1_ALL_METHODS_LONG_V2_EPOCHS:-200}" \
    LIVE_PLOT_INTERVAL="${ETTH1_ALL_METHODS_LONG_V2_LIVE_PLOT_INTERVAL:-10}" \
    bash scripts/paper_rerun/run_time_series_etth1_all_methods_long_v2.sh

echo "[$(timestamp)] [worker2_etth1_all_methods_long_v2] job finished"
