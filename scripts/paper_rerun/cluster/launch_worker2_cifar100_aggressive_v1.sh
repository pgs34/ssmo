#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_worker2_cifar100_aggressive_v1.lock"
flock -n 9 || {
  echo "[worker2_cifar100_aggressive_v1] launcher already running"
  exit 0
}

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_aggressive_v1/worker2}"
mkdir -p "$LOG_DIR"

GPU="${TARGET_GPU:-0}"
SEED="${SEED_SHARD:-2}"

run_logged_job \
  "worker2/cifar100_aggressive_v1_pool_gpu${GPU}_seed${SEED}" \
  "$LOG_DIR/pool.log" \
  env \
    GPU="$GPU" \
    DEVICE="${DEVICE:-cuda}" \
    SEEDS="$SEED" \
    EPOCHS="${POOL_EPOCHS:-100}" \
    BATCH_SIZE="${POOL_BATCH_SIZE_3090TI:-128}" \
    NUM_WORKERS="${NUM_WORKERS_3090TI:-4}" \
    DOWNLOAD="${DOWNLOAD:-0}" \
    INDEPENDENT_MODELS="resnet34_gelu" \
    PROTOCOL_ID="bestckpt_pool_v2" \
    HARDWARE_PROFILE="rtx3090ti" \
    LOG_DIR="$LOG_DIR/pool" \
    bash scripts/paper_rerun/run_classification_cifar100_bestckpt_pool_v2.sh

for group in independent dml; do
  run_logged_job \
    "worker2/cifar100_aggressive_v1_strict_${group}_gpu${GPU}_seed${SEED}" \
    "$LOG_DIR/strict_${group}.log" \
    env \
      GPU="$GPU" \
      DEVICE="${DEVICE:-cuda}" \
      RUN_GROUP="$group" \
      SEEDS="$SEED" \
      EPOCHS="${STRICT_EPOCHS:-100}" \
      BATCH_SIZE="${STRICT_BATCH_SIZE_3090TI:-128}" \
      NUM_WORKERS="${NUM_WORKERS_3090TI:-4}" \
      DOWNLOAD="${DOWNLOAD:-0}" \
      PROTOCOL_ID="strict128_aggressive_v1" \
      HARDWARE_PROFILE="rtx3090ti" \
      LOG_DIR="$LOG_DIR/strict_${group}" \
      bash scripts/paper_rerun/run_classification_cifar100_strict128_aggressive_v1.sh
done

for case in \
  "pcu_ramp_wide:peer_confident_student_uncertain:0.28:0.14:0.05:0.012:0.000:0.32:0.40:0.00:0.02:5:6.0:0.0004:0.85:0.9:2:0.5:0.02:0.00:1.0:0.00:4:12:45:0.35:0.00:0.00:0.00:-1:-1:0.00" \
  "uh_sched_mem:useful_hard_sample_confident:0.24:0.12:0.04:0.012:0.000:0.33:0.38:0.00:0.02:4:6.0:0.0004:0.88:0.8:2:0.5:0.02:0.00:1.0:0.00:5:18:55:0.30:0.10:0.90:0.50:30:60:0.00" \
  "pcu_dual55:peer_confident_student_uncertain:0.28:0.14:0.05:0.012:0.000:0.32:0.40:0.00:0.02:5:6.0:0.0004:0.85:0.9:2:0.5:0.02:0.00:1.0:0.00:4:12:45:0.35:0.00:0.00:0.00:-1:-1:0.55"
do
  label="${case%%:*}"
  run_logged_job \
    "worker2/cifar100_aggressive_v1_${label}_gpu${GPU}_seed${SEED}" \
    "$LOG_DIR/${label}.log" \
    env \
      GPU="$GPU" \
      DEVICE="${DEVICE:-cuda}" \
      RUN_GROUP="ssml" \
      SEEDS="$SEED" \
      EPOCHS="${STRICT_EPOCHS:-100}" \
      BATCH_SIZE="${STRICT_BATCH_SIZE_3090TI:-128}" \
      NUM_WORKERS="${NUM_WORKERS_3090TI:-4}" \
      DOWNLOAD="${DOWNLOAD:-0}" \
      PROTOCOL_ID="strict128_aggressive_v1" \
      HARDWARE_PROFILE="rtx3090ti" \
      SSML_CASE_SPECS="$case" \
      LOG_DIR="$LOG_DIR/$label" \
      bash scripts/paper_rerun/run_classification_cifar100_strict128_aggressive_v1.sh
done

for group in independent dml; do
  label="scaled3072_${group}"
  override_key="scaled3072_${group}_v2"
  run_logged_job \
    "worker2/cifar100_aggressive_v1_${label}_gpu${GPU}_seed${SEED}" \
    "$LOG_DIR/${label}.log" \
    env \
      GPU="$GPU" \
      DEVICE="${DEVICE:-cuda}" \
      RUN_GROUP="$group" \
      SEEDS="$SEED" \
      EPOCHS="${SCALED_EPOCHS:-100}" \
      BATCH_SIZE="${SCALED_BATCH_SIZE_3072:-3072}" \
      NUM_WORKERS="${NUM_WORKERS_3090TI:-4}" \
      DOWNLOAD="${DOWNLOAD:-0}" \
      PROTOCOL_ID="scaled_fair_bs3072_aggressive_v1" \
      HARDWARE_PROFILE="rtx3090ti" \
      LR="0.08" \
      INDEPENDENT_LABEL="$override_key" \
      DML_LABEL="$override_key" \
      LOG_DIR="$LOG_DIR/${label}" \
      bash scripts/paper_rerun/run_classification_cifar100_scaled_fair_aggressive_v1.sh
done

run_logged_job \
  "worker2/cifar100_aggressive_v1_oxtra42_gpu${GPU}_seed${SEED}" \
  "$LOG_DIR/oxtra42.log" \
  env \
    GPU="$GPU" \
    DEVICE="${DEVICE:-cuda}" \
    RUN_GROUP="ssml" \
    SEEDS="$SEED" \
    EPOCHS="${SCALED_EPOCHS:-100}" \
    BATCH_SIZE="${SCALED_BATCH_SIZE_3072:-3072}" \
    NUM_WORKERS="${NUM_WORKERS_3090TI:-4}" \
    DOWNLOAD="${DOWNLOAD:-0}" \
    PROTOCOL_ID="scaled_fair_bs3072_aggressive_v1" \
    HARDWARE_PROFILE="rtx3090ti" \
    LR="0.08" \
    SSML_CASE_SPECS="oxtra42_trainer_v2:0.42:0.020:0.018:0.000:0.42:0.01:18:6.0:0.0002:0.93:1.25:3:0.50:0.04:12:18:36:0.25" \
    LOG_DIR="$LOG_DIR/oxtra42" \
    bash scripts/paper_rerun/run_classification_cifar100_scaled_fair_aggressive_v1.sh

for group in independent dml; do
  label="scaled1536_${group}"
  override_key="scaled1536_${group}_v2"
  run_logged_job \
    "worker2/cifar100_aggressive_v1_${label}_gpu${GPU}_seed${SEED}" \
    "$LOG_DIR/${label}.log" \
    env \
      GPU="$GPU" \
      DEVICE="${DEVICE:-cuda}" \
      RUN_GROUP="$group" \
      SEEDS="$SEED" \
      EPOCHS="${SCALED_EPOCHS:-100}" \
      BATCH_SIZE="${SCALED_BATCH_SIZE_1536:-1536}" \
      NUM_WORKERS="${NUM_WORKERS_3090TI:-4}" \
      DOWNLOAD="${DOWNLOAD:-0}" \
      PROTOCOL_ID="scaled_fair_bs1536_aggressive_v1" \
      HARDWARE_PROFILE="rtx3090ti" \
      LR="0.05" \
      INDEPENDENT_LABEL="$override_key" \
      DML_LABEL="$override_key" \
      LOG_DIR="$LOG_DIR/${label}" \
      bash scripts/paper_rerun/run_classification_cifar100_scaled_fair_aggressive_v1.sh
done

run_logged_job \
  "worker2/cifar100_aggressive_v1_oxtra38_gpu${GPU}_seed${SEED}" \
  "$LOG_DIR/oxtra38.log" \
  env \
    GPU="$GPU" \
    DEVICE="${DEVICE:-cuda}" \
    RUN_GROUP="ssml" \
    SEEDS="$SEED" \
    EPOCHS="${SCALED_EPOCHS:-100}" \
    BATCH_SIZE="${SCALED_BATCH_SIZE_1536:-1536}" \
    NUM_WORKERS="${NUM_WORKERS_3090TI:-4}" \
    DOWNLOAD="${DOWNLOAD:-0}" \
    PROTOCOL_ID="scaled_fair_bs1536_aggressive_v1" \
    HARDWARE_PROFILE="rtx3090ti" \
    LR="0.05" \
    SSML_CASE_SPECS="oxtra38_trainer_v2:0.38:0.015:0.015:0.000:0.45:0.01:12:6.0:0.0003:0.95:1.00:2:0.50:0.03:15:20:45:0.35" \
    LOG_DIR="$LOG_DIR/oxtra38" \
    bash scripts/paper_rerun/run_classification_cifar100_scaled_fair_aggressive_v1.sh

echo "[worker2_cifar100_aggressive_v1] finished"
