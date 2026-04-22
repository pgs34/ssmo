#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_IDS_STR="${GPU_IDS:-0 1}"
read -r -a GPU_IDS <<< "$GPU_IDS_STR"

if [[ ${#GPU_IDS[@]} -lt 1 ]]; then
  echo "[run_all_parallel] no gpu ids provided" >&2
  exit 1
fi

ALL_EXPERIMENTS=(
  weather
  electricity
  etth1
  burgers
  darcy
  cifar10
  cifar100_cifarstem
)

DEFAULT_GPU0_EXPERIMENTS="weather etth1 cifar100_cifarstem"
DEFAULT_GPU1_EXPERIMENTS="electricity burgers darcy cifar10"

GPU0_EXPERIMENTS_STR="${GPU0_EXPERIMENTS:-$DEFAULT_GPU0_EXPERIMENTS}"
GPU1_EXPERIMENTS_STR="${GPU1_EXPERIMENTS:-$DEFAULT_GPU1_EXPERIMENTS}"

read -r -a GPU0_EXPERIMENTS <<< "$GPU0_EXPERIMENTS_STR"
read -r -a GPU1_EXPERIMENTS <<< "$GPU1_EXPERIMENTS_STR"

JOBS_PER_GPU="${JOBS_PER_GPU:-all}"
JOBS_PER_GPU0="${JOBS_PER_GPU0:-$JOBS_PER_GPU}"
JOBS_PER_GPU1="${JOBS_PER_GPU1:-$JOBS_PER_GPU}"

run_one_experiment() {
  local gpu="$1"
  local experiment="$2"
  local status=0

  echo "[run_all_parallel][gpu${gpu}] start $experiment"
  if CUDA_VISIBLE_DEVICES="$gpu" GPU="$gpu" bash "$SCRIPT_DIR/run_experiment.sh" "$experiment"; then
    echo "[run_all_parallel][gpu${gpu}] done $experiment"
  else
    status=$?
    echo "[run_all_parallel][gpu${gpu}] fail $experiment status=$status" >&2
    return "$status"
  fi
}

run_gpu_queue() {
  local gpu="$1"
  local max_jobs="$2"
  shift 2
  local experiments=("$@")
  local queue_status=0
  local finished_pid=""
  local finished_status=0
  local idx
  local pid

  local -a running_pids=()
  local -a running_names=()

  cleanup_queue() {
    local active_pid
    for active_pid in "${running_pids[@]}"; do
      kill "$active_pid" 2>/dev/null || true
    done
    wait || true
  }

  trap cleanup_queue INT TERM

  if [[ ${#experiments[@]} -eq 0 ]]; then
    echo "[run_all_parallel][gpu${gpu}] no experiments assigned"
    return 0
  fi

  if [[ "$max_jobs" == "all" || "$max_jobs" == "max" || "$max_jobs" == "0" ]]; then
    max_jobs="${#experiments[@]}"
  fi

  if (( max_jobs < 1 )); then
    echo "[run_all_parallel][gpu${gpu}] invalid max_jobs=$max_jobs" >&2
    return 1
  fi

  launch_job() {
    local experiment="$1"
    run_one_experiment "$gpu" "$experiment" &
    pid=$!
    running_pids+=("$pid")
    running_names+=("$experiment")
    echo "[run_all_parallel][gpu${gpu}] launch $experiment pid=$pid active=${#running_pids[@]}/$max_jobs"
  }

  reap_one() {
    finished_pid=""
    if wait -n -p finished_pid "${running_pids[@]}"; then
      finished_status=0
    else
      finished_status=$?
      queue_status=1
    fi

    for idx in "${!running_pids[@]}"; do
      if [[ "${running_pids[$idx]}" == "$finished_pid" ]]; then
        echo "[run_all_parallel][gpu${gpu}] slot_free ${running_names[$idx]} pid=$finished_pid status=$finished_status"
        unset 'running_pids[idx]'
        unset 'running_names[idx]'
        running_pids=("${running_pids[@]}")
        running_names=("${running_names[@]}")
        break
      fi
    done
  }

  for experiment in "${experiments[@]}"; do
    while (( ${#running_pids[@]} >= max_jobs )); do
      reap_one
    done
    launch_job "$experiment"
  done

  while (( ${#running_pids[@]} > 0 )); do
    reap_one
  done

  return "$queue_status"
}

QUEUE_STATUS=0
QUEUE_PIDS=()

cleanup_main() {
  local queue_pid
  for queue_pid in "${QUEUE_PIDS[@]}"; do
    kill "$queue_pid" 2>/dev/null || true
  done
  wait || true
}

trap cleanup_main INT TERM

echo "[run_all_parallel] gpu0=${GPU_IDS[0]} jobs=${JOBS_PER_GPU0} queue=${GPU0_EXPERIMENTS[*]}"
if [[ ${#GPU_IDS[@]} -ge 2 ]]; then
  echo "[run_all_parallel] gpu1=${GPU_IDS[1]} jobs=${JOBS_PER_GPU1} queue=${GPU1_EXPERIMENTS[*]}"
fi

run_gpu_queue "${GPU_IDS[0]}" "$JOBS_PER_GPU0" "${GPU0_EXPERIMENTS[@]}" &
QUEUE_PIDS+=("$!")

if [[ ${#GPU_IDS[@]} -ge 2 ]]; then
  run_gpu_queue "${GPU_IDS[1]}" "$JOBS_PER_GPU1" "${GPU1_EXPERIMENTS[@]}" &
  QUEUE_PIDS+=("$!")
fi

for queue_pid in "${QUEUE_PIDS[@]}"; do
  if ! wait "$queue_pid"; then
    QUEUE_STATUS=1
  fi
done

exit "$QUEUE_STATUS"
