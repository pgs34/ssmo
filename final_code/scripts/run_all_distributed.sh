#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$ROOT_DIR/results"
DISPATCH_ROOT="$RESULTS_DIR/_dispatch"
LOG_ROOT="$ROOT_DIR/logs/distributed"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MANIFEST_DIR="$DISPATCH_ROOT/$RUN_ID"
PLAN_FILE="$MANIFEST_DIR/plan.tsv"
EVENTS_FILE="$MANIFEST_DIR/events.tsv"
SUMMARY_FILE="$MANIFEST_DIR/summary.txt"
LATEST_FILE="$DISPATCH_ROOT/latest_run_id.txt"

LOCAL_GPU1_EXPERIMENTS_STR="${LOCAL_GPU1_EXPERIMENTS:-cifar100_cifarstem}"
WORKER2_GPU0_EXPERIMENTS_STR="${WORKER2_GPU0_EXPERIMENTS:-etth1 electricity}"
WORKER1_GPU0_EXPERIMENTS_STR="${WORKER1_GPU0_EXPERIMENTS:-weather darcy}"
WORKER1_GPU1_EXPERIMENTS_STR="${WORKER1_GPU1_EXPERIMENTS:-burgers cifar10}"

LOCAL_GPU1_ENV="${LOCAL_GPU1_ENV:-CIFAR100_STAGE_MAX_PARALLEL_RUNS=1 CLASSIFICATION_MAX_PARALLEL_RUNS=1}"
WORKER2_GPU0_ENV="${WORKER2_GPU0_ENV:-ETTH1_STAGE_MAX_PARALLEL_RUNS=2 TIME_SERIES_MAX_PARALLEL_RUNS=2}"
WORKER1_GPU0_ENV="${WORKER1_GPU0_ENV:-TIME_SERIES_MAX_PARALLEL_RUNS=2 OPERATOR_MAX_PARALLEL_RUNS=2}"
WORKER1_GPU1_ENV="${WORKER1_GPU1_ENV:-BURGERS_STAGE_MAX_PARALLEL_RUNS=2 OPERATOR_MAX_PARALLEL_RUNS=2 CIFAR10_STAGE_MAX_PARALLEL_RUNS=2 CLASSIFICATION_MAX_PARALLEL_RUNS=2}"
FORWARDED_ENV_NAMES_STR="${FORWARDED_ENV_NAMES:-DRY_RUN SEEDS EPOCHS DATASETS METHODS MODEL_PAIRS INDEPENDENT_MODELS REQUIRE_DISTINCT_PEER BATCH_SIZE NUM_WORKERS DEVICE DOWNLOAD TRAIN_SUBSET_SIZE VAL_SUBSET_SIZE MAX_PARALLEL_RUNS TIME_SERIES_MAX_PARALLEL_RUNS OPERATOR_MAX_PARALLEL_RUNS CLASSIFICATION_MAX_PARALLEL_RUNS ETTH1_STAGE_MAX_PARALLEL_RUNS BURGERS_STAGE_MAX_PARALLEL_RUNS CIFAR10_STAGE_MAX_PARALLEL_RUNS CIFAR100_STAGE_MAX_PARALLEL_RUNS}"

read -r -a LOCAL_GPU1_EXPERIMENTS <<< "$LOCAL_GPU1_EXPERIMENTS_STR"
read -r -a WORKER2_GPU0_EXPERIMENTS <<< "$WORKER2_GPU0_EXPERIMENTS_STR"
read -r -a WORKER1_GPU0_EXPERIMENTS <<< "$WORKER1_GPU0_EXPERIMENTS_STR"
read -r -a WORKER1_GPU1_EXPERIMENTS <<< "$WORKER1_GPU1_EXPERIMENTS_STR"
read -r -a FORWARDED_ENV_NAMES <<< "$FORWARDED_ENV_NAMES_STR"

QUEUE_PIDS=()
QUEUE_STATUS=0

mkdir -p "$MANIFEST_DIR" "$LOG_ROOT/$RUN_ID"
printf '%s\n' "$RUN_ID" > "$LATEST_FILE"
printf 'order\ttarget\thost\tgpu\texperiment\tenv\n' > "$PLAN_FILE"
printf 'ts\tevent\ttarget\thost\tgpu\texperiment\tstatus\tlog\n' > "$EVENTS_FILE"

sanitize_field() {
  printf '%s' "$1" | tr '\t\r\n' '   '
}

append_event() {
  local ts="$1"
  local event="$2"
  local target="$3"
  local host="$4"
  local gpu="$5"
  local experiment="$6"
  local status="$7"
  local log_file="$8"

  {
    flock -x 9
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(sanitize_field "$ts")" \
      "$(sanitize_field "$event")" \
      "$(sanitize_field "$target")" \
      "$(sanitize_field "$host")" \
      "$(sanitize_field "$gpu")" \
      "$(sanitize_field "$experiment")" \
      "$(sanitize_field "$status")" \
      "$(sanitize_field "$log_file")" >&9
  } 9>>"$EVENTS_FILE"
}

append_plan() {
  local order="$1"
  local target="$2"
  local host="$3"
  local gpu="$4"
  local experiment="$5"
  local env_string="$6"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(sanitize_field "$order")" \
    "$(sanitize_field "$target")" \
    "$(sanitize_field "$host")" \
    "$(sanitize_field "$gpu")" \
    "$(sanitize_field "$experiment")" \
    "$(sanitize_field "$env_string")" >> "$PLAN_FILE"
}

build_run_command() {
  local gpu="$1"
  local target="$2"
  local host="$3"
  local experiment="$4"
  local env_string="$5"
  local var_name
  local -a env_parts=()
  local -a extra_env=()

  for var_name in "${FORWARDED_ENV_NAMES[@]}"; do
    if [[ ${!var_name+x} == x ]]; then
      env_parts+=("$var_name=${!var_name}")
    fi
  done

  if [[ -n "$env_string" ]]; then
    read -r -a extra_env <<< "$env_string"
    env_parts+=("${extra_env[@]}")
  fi

  env_parts+=(
    "CUDA_VISIBLE_DEVICES=$gpu"
    "GPU=$gpu"
    "RUN_DISTRIBUTED_RUN_ID=$RUN_ID"
    "RUN_DISTRIBUTED_TARGET=$target"
    "RUN_DISTRIBUTED_HOST=$host"
    "RUN_DISTRIBUTED_GPU=$gpu"
  )

  printf 'cd %q && ' "$ROOT_DIR"
  printf '%q ' env "${env_parts[@]}" bash "$SCRIPT_DIR/run_experiment.sh" "$experiment"
}

run_one_experiment() {
  local target="$1"
  local host="$2"
  local gpu="$3"
  local env_string="$4"
  local experiment="$5"
  local scheduler_log="$LOG_ROOT/$RUN_ID/${target}__${experiment}.log"
  local remote_command
  local status=0

  remote_command="$(build_run_command "$gpu" "$target" "$host" "$experiment" "$env_string")"

  echo "[run_all_distributed][$target] start $experiment"
  echo "[run_all_distributed][$target] log=$scheduler_log"
  append_event "$(date --iso-8601=seconds)" "start" "$target" "$host" "$gpu" "$experiment" "running" "$scheduler_log"

  if [[ "$host" == "localhost" ]]; then
    if bash -lc "$remote_command" 2>&1 | tee "$scheduler_log"; then
      echo "[run_all_distributed][$target] done $experiment"
      append_event "$(date --iso-8601=seconds)" "finish" "$target" "$host" "$gpu" "$experiment" "ok" "$scheduler_log"
    else
      status=$?
      echo "[run_all_distributed][$target] fail $experiment status=$status" >&2
      append_event "$(date --iso-8601=seconds)" "finish" "$target" "$host" "$gpu" "$experiment" "fail:$status" "$scheduler_log"
      return "$status"
    fi
    return 0
  fi

  if ssh -o BatchMode=yes "$host" "bash -lc $(printf '%q' "$remote_command")" 2>&1 | tee "$scheduler_log"; then
    echo "[run_all_distributed][$target] done $experiment"
    append_event "$(date --iso-8601=seconds)" "finish" "$target" "$host" "$gpu" "$experiment" "ok" "$scheduler_log"
  else
    status=$?
    echo "[run_all_distributed][$target] fail $experiment status=$status" >&2
    append_event "$(date --iso-8601=seconds)" "finish" "$target" "$host" "$gpu" "$experiment" "fail:$status" "$scheduler_log"
    return "$status"
  fi
}

run_target_queue() {
  local target="$1"
  local host="$2"
  local gpu="$3"
  local env_string="$4"
  shift 4
  local experiments=("$@")
  local experiment

  if [[ ${#experiments[@]} -eq 0 ]]; then
    echo "[run_all_distributed][$target] no experiments assigned"
    return 0
  fi

  for experiment in "${experiments[@]}"; do
    run_one_experiment "$target" "$host" "$gpu" "$env_string" "$experiment"
  done
}

cleanup_main() {
  local pid
  for pid in "${QUEUE_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
}

trap cleanup_main INT TERM

order=1
for experiment in "${LOCAL_GPU1_EXPERIMENTS[@]}"; do
  append_plan "$order" "local_gpu1" "localhost" "1" "$experiment" "$LOCAL_GPU1_ENV"
  order=$((order + 1))
done
for experiment in "${WORKER2_GPU0_EXPERIMENTS[@]}"; do
  append_plan "$order" "worker2_gpu0" "worker2" "0" "$experiment" "$WORKER2_GPU0_ENV"
  order=$((order + 1))
done
for experiment in "${WORKER1_GPU0_EXPERIMENTS[@]}"; do
  append_plan "$order" "worker1_gpu0" "worker1" "0" "$experiment" "$WORKER1_GPU0_ENV"
  order=$((order + 1))
done
for experiment in "${WORKER1_GPU1_EXPERIMENTS[@]}"; do
  append_plan "$order" "worker1_gpu1" "worker1" "1" "$experiment" "$WORKER1_GPU1_ENV"
  order=$((order + 1))
done

{
  echo "run_id=$RUN_ID"
  echo "results_dispatch_dir=$MANIFEST_DIR"
  echo
  echo "local_gpu1: host=localhost gpu=1 env=$LOCAL_GPU1_ENV queue=${LOCAL_GPU1_EXPERIMENTS[*]}"
  echo "worker2_gpu0: host=worker2 gpu=0 env=$WORKER2_GPU0_ENV queue=${WORKER2_GPU0_EXPERIMENTS[*]}"
  echo "worker1_gpu0: host=worker1 gpu=0 env=$WORKER1_GPU0_ENV queue=${WORKER1_GPU0_EXPERIMENTS[*]}"
  echo "worker1_gpu1: host=worker1 gpu=1 env=$WORKER1_GPU1_ENV queue=${WORKER1_GPU1_EXPERIMENTS[*]}"
} > "$SUMMARY_FILE"

echo "[run_all_distributed] run_id=$RUN_ID"
echo "[run_all_distributed] manifest_dir=$MANIFEST_DIR"
echo "[run_all_distributed] local_gpu1 queue=${LOCAL_GPU1_EXPERIMENTS[*]}"
echo "[run_all_distributed] worker2_gpu0 queue=${WORKER2_GPU0_EXPERIMENTS[*]}"
echo "[run_all_distributed] worker1_gpu0 queue=${WORKER1_GPU0_EXPERIMENTS[*]}"
echo "[run_all_distributed] worker1_gpu1 queue=${WORKER1_GPU1_EXPERIMENTS[*]}"

run_target_queue "local_gpu1" "localhost" "1" "$LOCAL_GPU1_ENV" "${LOCAL_GPU1_EXPERIMENTS[@]}" &
QUEUE_PIDS+=("$!")

run_target_queue "worker2_gpu0" "worker2" "0" "$WORKER2_GPU0_ENV" "${WORKER2_GPU0_EXPERIMENTS[@]}" &
QUEUE_PIDS+=("$!")

run_target_queue "worker1_gpu0" "worker1" "0" "$WORKER1_GPU0_ENV" "${WORKER1_GPU0_EXPERIMENTS[@]}" &
QUEUE_PIDS+=("$!")

run_target_queue "worker1_gpu1" "worker1" "1" "$WORKER1_GPU1_ENV" "${WORKER1_GPU1_EXPERIMENTS[@]}" &
QUEUE_PIDS+=("$!")

for pid in "${QUEUE_PIDS[@]}"; do
  if ! wait "$pid"; then
    QUEUE_STATUS=1
  fi
done

append_event "$(date --iso-8601=seconds)" "run_complete" "scheduler" "localhost" "-" "-" "$QUEUE_STATUS" "$MANIFEST_DIR"
exit "$QUEUE_STATUS"
