#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_cifar100_scaled_fair_v2.lock"
flock -n 9 || {
  echo "[node0_cifar100_scaled_fair_v2] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_scaled_fair_v2/node0}"
CONFIG_DIR="${CONFIG_DIR:-results/classification_cifar100_scaled_fair_v2/config}"
mkdir -p "$LOG_DIR" "$CONFIG_DIR"
rm -f "$CONFIG_DIR/fair_batch.env" "$CONFIG_DIR/worker3_ready.flag" "$CONFIG_DIR/worker3_skip.flag"

POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-60}"
FREE_GPU_MAX_UTIL="${FREE_GPU_MAX_UTIL:-10}"
FREE_GPU_MAX_MEM_MIB="${FREE_GPU_MAX_MEM_MIB:-512}"

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
  local gpu="$1"
  while ! gpu_is_available "$gpu"; do
    echo "[$(timestamp)] [node0_cifar100_scaled_fair_v2] waiting for gpu=$gpu util<=$FREE_GPU_MAX_UTIL mem<=$FREE_GPU_MAX_MEM_MIB MiB"
    sleep "$POLL_INTERVAL_SEC"
  done
}

run_smoke_reference() {
  local batch="$1"
  run_logged_job \
    "node0/cifar100_scaled_fair_v2_smoke_bs${batch}" \
    "$LOG_DIR/smoke_dml_bs${batch}.log" \
    env \
      GPU="0" \
      DEVICE="${DEVICE:-cuda}" \
      REFERENCE_MODE="dml" \
      SEEDS="0" \
      EPOCHS="1" \
      BATCH_SIZE="$batch" \
      NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
      OUTPUT_ROOT="results/classification_cifar100_scaled_fair_v2/smoke_dml_bs${batch}" \
      LOG_DIR="results/logs/classification_cifar100_scaled_fair_v2/smoke_dml_bs${batch}" \
      PROTOCOL_ID="scaled_fair_bs${batch}_smoke" \
      HARDWARE_PROFILE="rtx4090" \
      bash scripts/paper_rerun/run_classification_cifar100_scaled_fair_reference_v2.sh
}

wait_for_gpu 0
wait_for_gpu 1

FAIR_BATCH="${FAIR_BATCH:-3072}"
if ! run_smoke_reference "$FAIR_BATCH"; then
  FAIR_BATCH=1536
  run_smoke_reference "$FAIR_BATCH"
fi

printf 'FAIR_BATCH=%s\nPROTOCOL_ID=scaled_fair_bs%s\n' "$FAIR_BATCH" "$FAIR_BATCH" > "$CONFIG_DIR/fair_batch.env"
echo "[node0_cifar100_scaled_fair_v2] selected FAIR_BATCH=$FAIR_BATCH"

run_logged_job \
  "node0/cifar100_scaled_fair_v2_independent" \
  "$LOG_DIR/independent.log" \
  env \
    GPU="0" \
    DEVICE="${DEVICE:-cuda}" \
    REFERENCE_MODE="independent" \
    SEEDS="${SEEDS:-0 1 2}" \
    EPOCHS="${EPOCHS:-100}" \
    BATCH_SIZE="$FAIR_BATCH" \
    NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
    OUTPUT_ROOT="results/classification_cifar100_scaled_fair_reference_v2/node0_gpu0_independent" \
    LOG_DIR="results/logs/classification_cifar100_scaled_fair_reference_v2/node0_gpu0_independent" \
    PROTOCOL_ID="scaled_fair_bs${FAIR_BATCH}" \
    HARDWARE_PROFILE="rtx4090" \
    bash scripts/paper_rerun/run_classification_cifar100_scaled_fair_reference_v2.sh &
PID_INDEP=$!

run_logged_job \
  "node0/cifar100_scaled_fair_v2_dml" \
  "$LOG_DIR/dml.log" \
  env \
    GPU="1" \
    DEVICE="${DEVICE:-cuda}" \
    REFERENCE_MODE="dml" \
    SEEDS="${SEEDS:-0 1 2}" \
    EPOCHS="${EPOCHS:-100}" \
    BATCH_SIZE="$FAIR_BATCH" \
    NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
    OUTPUT_ROOT="results/classification_cifar100_scaled_fair_reference_v2/node0_gpu1_dml" \
    LOG_DIR="results/logs/classification_cifar100_scaled_fair_reference_v2/node0_gpu1_dml" \
    PROTOCOL_ID="scaled_fair_bs${FAIR_BATCH}" \
    HARDWARE_PROFILE="rtx4090" \
    bash scripts/paper_rerun/run_classification_cifar100_scaled_fair_reference_v2.sh &
PID_DML=$!

echo "[node0_cifar100_scaled_fair_v2] started reference runs indep=$PID_INDEP dml=$PID_DML"
wait "$PID_INDEP"
wait "$PID_DML"
echo "[node0_cifar100_scaled_fair_v2] references finished"

for _ in $(seq 1 20); do
  if [[ -f "$CONFIG_DIR/worker3_ready.flag" || -f "$CONFIG_DIR/worker3_skip.flag" ]]; then
    break
  fi
  sleep 15
done

WORKER3_READY=0
if [[ -f "$CONFIG_DIR/worker3_ready.flag" ]]; then
  WORKER3_READY=1
fi

run_logged_job \
  "node0/cifar100_scaled_fair_v2_ssml_gpu0" \
  "$LOG_DIR/ssml_gpu0.log" \
  env \
    GPU="0" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    EPOCHS="${EPOCHS:-100}" \
    BATCH_SIZE="$FAIR_BATCH" \
    NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
    OUTPUT_ROOT="results/classification_cifar100_overbatch_reproduce_fair_v2/node0_gpu0" \
    LOG_DIR="results/logs/classification_cifar100_overbatch_reproduce_fair_v2/node0_gpu0/cases" \
    PROTOCOL_ID="scaled_fair_bs${FAIR_BATCH}" \
    HARDWARE_PROFILE="rtx4090" \
    CASE_SPECS="${GPU0_CASE_SPECS:-oxtra44_thr40_gap1_pc16_aug100:0.44:0.025:0.012:0.000:0.40:0.01:16:5.5:0.0002:0.92:1.00:4:0.50:0.04:1:10:35:0.25}" \
    bash scripts/paper_rerun/run_classification_cifar100_overbatch_reproduce_fair_v2.sh &
PID_SSML0=$!

GPU1_CASES="${GPU1_CASE_SPECS:-oxtra38_thr45_gap1_pc12_aug100:0.38:0.015:0.015:0.000:0.45:0.01:12:6.0:0.0003:0.95:1.00:2:0.50:0.03:1:15:45:0.35}"
if [[ "$WORKER3_READY" == "0" ]]; then
  GPU1_CASES="${GPU1_CASES} ${GPU1_EXTRA_CASE_SPECS:-oxtra42_thr42_gap1_pc18_aug125:0.42:0.02:0.018:0.000:0.42:0.01:18:6.0:0.0002:0.93:1.25:3:0.50:0.04:1:12:36:0.25}"
fi

run_logged_job \
  "node0/cifar100_scaled_fair_v2_ssml_gpu1" \
  "$LOG_DIR/ssml_gpu1.log" \
  env \
    GPU="1" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="${SEEDS:-0 1 2}" \
    EPOCHS="${EPOCHS:-100}" \
    BATCH_SIZE="$FAIR_BATCH" \
    NUM_WORKERS="${NUM_WORKERS_4090:-4}" \
    OUTPUT_ROOT="results/classification_cifar100_overbatch_reproduce_fair_v2/node0_gpu1" \
    LOG_DIR="results/logs/classification_cifar100_overbatch_reproduce_fair_v2/node0_gpu1/cases" \
    PROTOCOL_ID="scaled_fair_bs${FAIR_BATCH}" \
    HARDWARE_PROFILE="rtx4090" \
    CASE_SPECS="$GPU1_CASES" \
    bash scripts/paper_rerun/run_classification_cifar100_overbatch_reproduce_fair_v2.sh &
PID_SSML1=$!

echo "[node0_cifar100_scaled_fair_v2] started ssml runs gpu0=$PID_SSML0 gpu1=$PID_SSML1 worker3_ready=$WORKER3_READY"
wait "$PID_SSML0"
wait "$PID_SSML1"

python scripts/paper_rerun/family_result_report.py \
  --run-root results/classification_cifar100_overbatch_reproduce_fair_v2 \
  --metric-key best_val_acc \
  --expected-seeds 0,1,2 \
  --higher-is-better \
  --current-best 0.536567 \
  --strongest-baseline 0.545067 | tee "$LOG_DIR/family_report.json"

echo "[node0_cifar100_scaled_fair_v2] finished"
