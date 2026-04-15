#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_node0_cifar100_cifarstem_followup_v1_gpu1_seed2.lock"
flock -n 9 || {
  echo "[node0_cifar100_cifarstem_followup_v1_gpu1_seed2] launcher already running"
  exit 0
}

LOG_ROOT="${LOG_ROOT:-results/logs/classification_cifar100_cifarstem_followup_v1/gpu1_seed2_queue}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_cifar100_cifarstem_followup_v1}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
if [[ -z "${BEST_CKPT_TEMPLATE:-}" ]]; then
  BEST_CKPT_TEMPLATE="results/classification_cifar100_bestckpt_pool_cifarstem_v1/classification/classification/cifar100/{model}_independent_${CLASSIFICATION_IMITATION_LOSS}_seed{seed}/best_model.pt"
fi
GPU="${GPU:-1}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-rtx4090}"
SEED="${SEED:-2}"
EPOCHS="${EPOCHS:-100}"
INDEPENDENT_BATCH_SIZE="${INDEPENDENT_BATCH_SIZE:-1536}"
PAIRWISE_BATCH_SIZE="${PAIRWISE_BATCH_SIZE:-768}"
PAIRWISE_LR="${PAIRWISE_LR:-0.025}"
PAIRWISE_WEIGHT_DECAY="${PAIRWISE_WEIGHT_DECAY:-5e-4}"
PAIRWISE_SCHEDULER_WARMUP_EPOCHS="${PAIRWISE_SCHEDULER_WARMUP_EPOCHS:-8}"
PAIRWISE_SCHEDULER_MIN_SCALE="${PAIRWISE_SCHEDULER_MIN_SCALE:-0.20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD="${DOWNLOAD:-0}"
CONTROL_GROUPS="${CONTROL_GROUPS:-independent dml}"

mkdir -p "$LOG_ROOT"

summary_path_for_label() {
  local label="$1"
  local seed="$2"
  local run_dir="$OUTPUT_ROOT/$label/classification/cifar100"

  if [[ "$label" == "cifarstem_independent_v1" ]]; then
    printf '%s\n' "$run_dir/resnet34_cifar_gelu_independent_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/summary.json"
    return 0
  fi
  if [[ "$label" == "cifarstem_dml_v1" ]]; then
    printf '%s\n' "$run_dir/resnet34_cifar_gelu_dml_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/summary.json"
    return 0
  fi
  printf '%s\n' "$run_dir/resnet34_cifar_gelu_ssml_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/summary.json"
}

run_control() {
  local group="$1"
  local label="cifarstem_${group}_v1"
  local log_dir="$LOG_ROOT/${group}_seed${SEED}"
  local summary_path
  summary_path="$(summary_path_for_label "$label" "$SEED")"
  if [[ -f "$summary_path" ]]; then
    echo "[node0_cifar100_cifarstem_followup_v1_gpu1_seed2] skip_completed label=$label seed=$SEED summary=$summary_path"
    return 0
  fi
  run_logged_job \
    "node0/cifar100_cifarstem_followup_v1_gpu${GPU}_${group}_seed${SEED}" \
    "$log_dir.log" \
    env \
      GPU="$GPU" \
      DEVICE="${DEVICE:-cuda}" \
      RUN_GROUP="$group" \
      SEEDS="$SEED" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$INDEPENDENT_BATCH_SIZE" \
      INDEPENDENT_BATCH_SIZE="$INDEPENDENT_BATCH_SIZE" \
      DML_BATCH_SIZE="$PAIRWISE_BATCH_SIZE" \
      SSML_BATCH_SIZE="$PAIRWISE_BATCH_SIZE" \
      DML_LR="$PAIRWISE_LR" \
      SSML_LR="$PAIRWISE_LR" \
      DML_WEIGHT_DECAY="$PAIRWISE_WEIGHT_DECAY" \
      SSML_WEIGHT_DECAY="$PAIRWISE_WEIGHT_DECAY" \
      DML_SCHEDULER_WARMUP_EPOCHS="$PAIRWISE_SCHEDULER_WARMUP_EPOCHS" \
      SSML_SCHEDULER_WARMUP_EPOCHS="$PAIRWISE_SCHEDULER_WARMUP_EPOCHS" \
      DML_SCHEDULER_MIN_SCALE="$PAIRWISE_SCHEDULER_MIN_SCALE" \
      SSML_SCHEDULER_MIN_SCALE="$PAIRWISE_SCHEDULER_MIN_SCALE" \
      NUM_WORKERS="$NUM_WORKERS" \
      DOWNLOAD="$DOWNLOAD" \
      HARDWARE_PROFILE="$HARDWARE_PROFILE" \
      LOG_DIR="$log_dir" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      PROTOCOL_ID="cifarstem_followup_v1" \
      CIFARSTEM_INDEPENDENT_LABEL="cifarstem_independent_v1" \
      CIFARSTEM_DML_LABEL="cifarstem_dml_v1" \
      BEST_CKPT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      bash scripts/paper_rerun/run_classification_cifar100_cifarstem_followup_v1.sh
}

run_ssml_spec() {
  local label="$1"
  local spec="$2"
  local log_dir="$LOG_ROOT/${label}_seed${SEED}"
  local summary_path
  summary_path="$(summary_path_for_label "$label" "$SEED")"
  if [[ -f "$summary_path" ]]; then
    echo "[node0_cifar100_cifarstem_followup_v1_gpu1_seed2] skip_completed label=$label seed=$SEED summary=$summary_path"
    return 0
  fi
  run_logged_job \
    "node0/cifar100_cifarstem_followup_v1_gpu${GPU}_${label}_seed${SEED}" \
    "$log_dir.log" \
    env \
      GPU="$GPU" \
      DEVICE="${DEVICE:-cuda}" \
      RUN_GROUP="ssml" \
      SEEDS="$SEED" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$INDEPENDENT_BATCH_SIZE" \
      INDEPENDENT_BATCH_SIZE="$INDEPENDENT_BATCH_SIZE" \
      DML_BATCH_SIZE="$PAIRWISE_BATCH_SIZE" \
      SSML_BATCH_SIZE="$PAIRWISE_BATCH_SIZE" \
      DML_LR="$PAIRWISE_LR" \
      SSML_LR="$PAIRWISE_LR" \
      DML_WEIGHT_DECAY="$PAIRWISE_WEIGHT_DECAY" \
      SSML_WEIGHT_DECAY="$PAIRWISE_WEIGHT_DECAY" \
      DML_SCHEDULER_WARMUP_EPOCHS="$PAIRWISE_SCHEDULER_WARMUP_EPOCHS" \
      SSML_SCHEDULER_WARMUP_EPOCHS="$PAIRWISE_SCHEDULER_WARMUP_EPOCHS" \
      DML_SCHEDULER_MIN_SCALE="$PAIRWISE_SCHEDULER_MIN_SCALE" \
      SSML_SCHEDULER_MIN_SCALE="$PAIRWISE_SCHEDULER_MIN_SCALE" \
      NUM_WORKERS="$NUM_WORKERS" \
      DOWNLOAD="$DOWNLOAD" \
      HARDWARE_PROFILE="$HARDWARE_PROFILE" \
      SSML_CASE_SPECS="$spec" \
      LOG_DIR="$log_dir" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      PROTOCOL_ID="cifarstem_followup_v1" \
      CIFARSTEM_INDEPENDENT_LABEL="cifarstem_independent_v1" \
      CIFARSTEM_DML_LABEL="cifarstem_dml_v1" \
      BEST_CKPT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      bash scripts/paper_rerun/run_classification_cifar100_cifarstem_followup_v1.sh
}

for control_group in $CONTROL_GROUPS; do
  run_control "$control_group"
done

run_ssml_spec \
  "pcu_cifarstem_sched_v1" \
  "pcu_cifarstem_sched_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.012:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00"
run_ssml_spec \
  "pcu_cifarstem_sched_l10_v1" \
  "pcu_cifarstem_sched_l10_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.010:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00"
run_ssml_spec \
  "pcu_cifarstem_sched_l09_v1" \
  "pcu_cifarstem_sched_l09_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.009:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00"
run_ssml_spec \
  "pcu_cifarstem_sched_l08_t7_v1" \
  "pcu_cifarstem_sched_l08_t7_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.008:0.000:0.35:0.40:0.01:0.03:5:7.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00"
run_ssml_spec \
  "pcu_cifarstem_dense_v1" \
  "pcu_cifarstem_dense_v1:peer_confident_student_uncertain:0.28:0.12:0.05:0.012:0.000:0.35:0.40:0.01:0.03:6:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.15:0.90:1.00:25:70:0.00"
run_ssml_spec \
  "oxtra42_cifarstem_v1" \
  "oxtra42_cifarstem_v1:useful_hard_sample_confident:0.42:0.42:0.020:0.018:0.000:0.42:0.42:0.01:0.01:18:6.0:0.0002:0.93:1.25:3:0.50:0.04:0.00:1.00:0.00:12:18:36:0.25:0.00:0.00:0.00:-1:-1:0.00"
run_ssml_spec \
  "oxtra35_cifarstem_relax_v1" \
  "oxtra35_cifarstem_relax_v1:useful_hard_sample_confident:0.35:0.35:0.020:0.015:0.000:0.40:0.40:0.01:0.01:12:6.0:0.0002:0.90:1.10:3:0.50:0.04:0.00:1.00:0.00:8:18:45:0.25:0.00:0.00:0.00:-1:-1:0.00"

echo "[node0_cifar100_cifarstem_followup_v1_gpu1_seed2] all jobs finished"
