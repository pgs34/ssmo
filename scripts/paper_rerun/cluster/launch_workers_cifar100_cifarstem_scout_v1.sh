#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOCK_ROOT="results/.locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/launch_workers_cifar100_cifarstem_scout_v1.lock"
flock -n 9 || {
  echo "[workers_cifar100_cifarstem_scout_v1] launcher already running"
  exit 0
}

LOG_ROOT="${LOG_ROOT:-results/logs/classification_cifar100_cifarstem_scout_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_cifar100_cifarstem_scout_v1}"
SCOUT_EPOCHS="${SCOUT_EPOCHS:-20}"
SCOUT_SEED="${SCOUT_SEED:-2}"

PAIR_BATCH_3090TI="${PAIR_BATCH_3090TI:-768}"
PAIR_BATCH_2080TI="${PAIR_BATCH_2080TI:-192}"
NUM_WORKERS_3090TI="${NUM_WORKERS_3090TI:-4}"
NUM_WORKERS_2080TI="${NUM_WORKERS_2080TI:-2}"
SSML_LR_3090TI="${SSML_LR_3090TI:-0.025}"
SSML_LR_2080TI="${SSML_LR_2080TI:-0.00625}"
SSML_WARMUP_3090TI="${SSML_WARMUP_3090TI:-8}"
SSML_WARMUP_2080TI="${SSML_WARMUP_2080TI:-8}"
SSML_MIN_SCALE_3090TI="${SSML_MIN_SCALE_3090TI:-0.20}"
SSML_MIN_SCALE_2080TI="${SSML_MIN_SCALE_2080TI:-0.20}"

mkdir -p "$LOG_ROOT"

PCU_BASE_SPEC="pcu_cifarstem_sched_scout_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.012:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00"
PCU_L10_SPEC="pcu_cifarstem_sched_l10_scout_v1:peer_confident_student_uncertain:0.24:0.10:0.05:0.010:0.000:0.35:0.40:0.01:0.03:5:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.12:0.90:0.80:30:65:0.00"
PCU_DENSE_SPEC="pcu_cifarstem_dense_scout_v1:peer_confident_student_uncertain:0.28:0.12:0.05:0.012:0.000:0.35:0.40:0.01:0.03:6:6.0:0.0004:0.85:0.90:2:0.50:0.02:0.72:0.90:0.03:4:18:60:0.35:0.15:0.90:1.00:25:70:0.00"
OXTRA42_SPEC="oxtra42_cifarstem_scout_v1:useful_hard_sample_confident:0.42:0.42:0.020:0.018:0.000:0.42:0.42:0.01:0.01:18:6.0:0.0002:0.93:1.25:3:0.50:0.04:0.00:1.00:0.00:12:18:36:0.25:0.00:0.00:0.00:-1:-1:0.00"

launch_remote_scout() {
  local host="$1"
  local gpu="$2"
  local label="$3"
  local spec="$4"
  local batch_size="$5"
  local num_workers="$6"
  local lr="$7"
  local warmup="$8"
  local min_scale="$9"
  local hardware_profile="${10}"

  local remote_log_dir="$ROOT_DIR/$LOG_ROOT/$host/$label"
  local remote_stdout="$remote_log_dir/launcher.out"

  echo "[workers_cifar100_cifarstem_scout_v1] launch host=$host gpu=$gpu label=$label batch_size=$batch_size lr=$lr warmup=$warmup"

  ssh "$host" "mkdir -p '$remote_log_dir' '$ROOT_DIR/$OUTPUT_ROOT' && cd '$ROOT_DIR' && nohup env \
GPU='$gpu' \
DEVICE='cuda' \
RUN_GROUP='ssml' \
SEEDS='$SCOUT_SEED' \
EPOCHS='$SCOUT_EPOCHS' \
BATCH_SIZE='$batch_size' \
SSML_BATCH_SIZE='$batch_size' \
DML_BATCH_SIZE='$batch_size' \
NUM_WORKERS='$num_workers' \
DOWNLOAD='0' \
OUTPUT_ROOT='$OUTPUT_ROOT' \
PROTOCOL_ID='cifarstem_scout_v1' \
HARDWARE_PROFILE='$hardware_profile' \
SSML_LR='$lr' \
SSML_SCHEDULER_WARMUP_EPOCHS='$warmup' \
SSML_SCHEDULER_MIN_SCALE='$min_scale' \
LOG_DIR='$LOG_ROOT/$host/$label' \
SSML_CASE_SPECS='$spec' \
bash scripts/paper_rerun/run_classification_cifar100_cifarstem_followup_v1.sh \
> '$remote_stdout' 2>&1 < /dev/null & echo \$!"
}

launch_remote_scout "worker2" "0" "pcu_base" "$PCU_BASE_SPEC" "$PAIR_BATCH_3090TI" "$NUM_WORKERS_3090TI" "$SSML_LR_3090TI" "$SSML_WARMUP_3090TI" "$SSML_MIN_SCALE_3090TI" "rtx3090ti"
launch_remote_scout "worker3" "0" "pcu_l10" "$PCU_L10_SPEC" "$PAIR_BATCH_3090TI" "$NUM_WORKERS_3090TI" "$SSML_LR_3090TI" "$SSML_WARMUP_3090TI" "$SSML_MIN_SCALE_3090TI" "rtx3090ti"
launch_remote_scout "worker1" "0" "pcu_dense" "$PCU_DENSE_SPEC" "$PAIR_BATCH_2080TI" "$NUM_WORKERS_2080TI" "$SSML_LR_2080TI" "$SSML_WARMUP_2080TI" "$SSML_MIN_SCALE_2080TI" "rtx2080ti"
launch_remote_scout "worker1" "1" "oxtra42" "$OXTRA42_SPEC" "$PAIR_BATCH_2080TI" "$NUM_WORKERS_2080TI" "$SSML_LR_2080TI" "$SSML_WARMUP_2080TI" "$SSML_MIN_SCALE_2080TI" "rtx2080ti"

echo "[workers_cifar100_cifarstem_scout_v1] dispatch complete"
