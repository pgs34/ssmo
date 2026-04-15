#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

QUEUE_NAME="${QUEUE_NAME:-cifarstem_followup_backfill_queue}"
TARGET_GPU="${TARGET_GPU:-0}"
TARGET_SEED="${TARGET_SEED:-0}"
JOB_ITEMS="${JOB_ITEMS:-}"
LOG_ROOT="${LOG_ROOT:-results/logs/classification_cifar100_cifarstem_followup_v1/autofinish}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_cifar100_cifarstem_followup_v1}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
PROTOCOL_ID="${PROTOCOL_ID:-cifarstem_followup_v1}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-rtx3090ti}"
CIFARSTEM_INDEPENDENT_LABEL="${CIFARSTEM_INDEPENDENT_LABEL:-cifarstem_independent_v1}"
CIFARSTEM_DML_LABEL="${CIFARSTEM_DML_LABEL:-cifarstem_dml_v1}"
ALL_PROBE_CASE_SPECS="${ALL_PROBE_CASE_SPECS:-pcu_cifarstem_sched_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.012:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00 pcu_cifarstem_sched_l10_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.010:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00 pcu_cifarstem_dense_v1:peer_confident_student_uncertain:0.28:0.12:0.05:0.012:0.000:0.35:0.40:0.01:0.03:6:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.15:0.90:1.00:25:70:0.00 oxtra42_cifarstem_v1:useful_hard_sample_confident:0.42:0.42:0.020:0.018:0.000:0.42:0.42:0.01:0.01:18:6.0:0.0002:0.93:1.25:3:0.50:0.04:0.00:1.00:0.00:12:18:36:0.25:0.00:0.00:0.00:-1:-1:0.00 pcu_cifarstem_sched_l09_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.009:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00 pcu_cifarstem_sched_l08_t7_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.008:0.000:0.35:0.40:0.01:0.03:5:7.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00 oxtra35_cifarstem_relax_v1:useful_hard_sample_confident:0.35:0.35:0.020:0.015:0.000:0.40:0.40:0.01:0.01:12:6.0:0.0002:0.90:1.10:3:0.50:0.04:0.00:1.00:0.00:8:18:45:0.25:0.00:0.00:0.00:-1:-1:0.00}"

FOLLOWUP_EPOCHS="${FOLLOWUP_EPOCHS:-100}"
INDEPENDENT_BATCH_SIZE="${INDEPENDENT_BATCH_SIZE:-1536}"
DUAL_BATCH_SIZE="${DUAL_BATCH_SIZE:-768}"
NUM_WORKERS="${NUM_WORKERS:-4}"
INDEPENDENT_LR="${INDEPENDENT_LR:-0.05}"
INDEPENDENT_WARMUP="${INDEPENDENT_WARMUP:-5}"
INDEPENDENT_MIN_SCALE="${INDEPENDENT_MIN_SCALE:-0.10}"
DML_LR="${DML_LR:-0.025}"
DML_WARMUP="${DML_WARMUP:-8}"
DML_MIN_SCALE="${DML_MIN_SCALE:-0.20}"
SSML_LR="${SSML_LR:-0.025}"
SSML_WARMUP="${SSML_WARMUP:-8}"
SSML_MIN_SCALE="${SSML_MIN_SCALE:-0.20}"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT" "$LOG_ROOT"
exec 9>"$LOCK_ROOT/${QUEUE_NAME}.lock"
flock -n 9 || {
  echo "[run_worker_cifar100_cifarstem_followup_queue_v1] queue already running: $QUEUE_NAME"
  exit 0
}

summary_path_for_label() {
  local label="$1"
  local seed="$2"
  local run_dir="$OUTPUT_ROOT/$label/classification/cifar100"

  if [[ "$label" == "$CIFARSTEM_INDEPENDENT_LABEL" ]]; then
    printf '%s\n' "$run_dir/resnet34_cifar_gelu_independent_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/summary.json"
    return 0
  fi
  if [[ "$label" == "$CIFARSTEM_DML_LABEL" ]]; then
    printf '%s\n' "$run_dir/resnet34_cifar_gelu_dml_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/summary.json"
    return 0
  fi
  printf '%s\n' "$run_dir/resnet34_cifar_gelu_ssml_${CLASSIFICATION_IMITATION_LOSS}_seed${seed}/summary.json"
}

find_case_spec() {
  local target_label="$1"
  local spec
  for spec in $ALL_PROBE_CASE_SPECS; do
    if [[ "${spec%%:*}" == "$target_label" ]]; then
      printf '%s\n' "$spec"
      return 0
    fi
  done
  return 1
}

run_group_job() {
  local run_group="$1"
  local label="$2"
  shift 2

  local summary_path
  summary_path="$(summary_path_for_label "$label" "$TARGET_SEED")"
  if [[ -f "$summary_path" ]]; then
    echo "[run_worker_cifar100_cifarstem_followup_queue_v1] skip_completed queue=$QUEUE_NAME label=$label seed=$TARGET_SEED summary=$summary_path"
    return 0
  fi

  echo "[run_worker_cifar100_cifarstem_followup_queue_v1] start queue=$QUEUE_NAME label=$label seed=$TARGET_SEED group=$run_group"
  env \
    GPU="$TARGET_GPU" \
    DEVICE="cuda" \
    RUN_GROUP="$run_group" \
    SEEDS="$TARGET_SEED" \
    EPOCHS="$FOLLOWUP_EPOCHS" \
    BATCH_SIZE="$DUAL_BATCH_SIZE" \
    INDEPENDENT_BATCH_SIZE="$INDEPENDENT_BATCH_SIZE" \
    DML_BATCH_SIZE="$DUAL_BATCH_SIZE" \
    SSML_BATCH_SIZE="$DUAL_BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    DOWNLOAD="0" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    PROTOCOL_ID="$PROTOCOL_ID" \
    HARDWARE_PROFILE="$HARDWARE_PROFILE" \
    CIFARSTEM_INDEPENDENT_LABEL="$CIFARSTEM_INDEPENDENT_LABEL" \
    CIFARSTEM_DML_LABEL="$CIFARSTEM_DML_LABEL" \
    LOG_DIR="$LOG_ROOT/${label}_seed${TARGET_SEED}" \
    CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
    "$@" \
    bash scripts/paper_rerun/run_classification_cifar100_cifarstem_followup_v1.sh
}

for item in $JOB_ITEMS; do
  IFS=':' read -r run_group label <<< "$item"
  case "$run_group" in
    independent)
      run_group_job \
        "independent" \
        "$label" \
        INDEPENDENT_LR="$INDEPENDENT_LR" \
        INDEPENDENT_SCHEDULER_WARMUP_EPOCHS="$INDEPENDENT_WARMUP" \
        INDEPENDENT_SCHEDULER_MIN_SCALE="$INDEPENDENT_MIN_SCALE"
      ;;
    dml)
      run_group_job \
        "dml" \
        "$label" \
        DML_LR="$DML_LR" \
        DML_SCHEDULER_WARMUP_EPOCHS="$DML_WARMUP" \
        DML_SCHEDULER_MIN_SCALE="$DML_MIN_SCALE"
      ;;
    ssml)
      case_spec="$(find_case_spec "$label")" || {
        echo "[run_worker_cifar100_cifarstem_followup_queue_v1] missing_case_spec label=$label" >&2
        exit 1
      }
      run_group_job \
        "ssml" \
        "$label" \
        SSML_LR="$SSML_LR" \
        SSML_SCHEDULER_WARMUP_EPOCHS="$SSML_WARMUP" \
        SSML_SCHEDULER_MIN_SCALE="$SSML_MIN_SCALE" \
        SSML_CASE_SPECS="$case_spec"
      ;;
    *)
      echo "[run_worker_cifar100_cifarstem_followup_queue_v1] unknown run_group=$run_group item=$item" >&2
      exit 1
      ;;
  esac
done

echo "[run_worker_cifar100_cifarstem_followup_queue_v1] done queue=$QUEUE_NAME"
