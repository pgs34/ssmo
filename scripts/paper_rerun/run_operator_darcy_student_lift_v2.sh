#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/operator_darcy_student_lift_v2}"
LOG_DIR="${LOG_DIR:-results/logs/operator_darcy_student_lift_v2}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-110}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BASE_LR="${BASE_LR:-2e-4}"
HUBER_LR="${HUBER_LR:-1.5e-4}"
BASE_DARCY_ROOT="${BASE_DARCY_ROOT:-results/operator_ssml_tuned_v1/operator/darcy}"
DEFAULT_INIT_CHECKPOINT_TEMPLATE="${BASE_DARCY_ROOT}/deeponet_independent_mse_seed"'{'seed'}'"/model.pt"
DEFAULT_PEER_INIT_CHECKPOINT_TEMPLATE="${BASE_DARCY_ROOT}/fno_independent_mse_seed"'{'seed'}'"/model.pt"
INIT_CHECKPOINT_TEMPLATE="${INIT_CHECKPOINT_TEMPLATE:-$DEFAULT_INIT_CHECKPOINT_TEMPLATE}"
PEER_INIT_CHECKPOINT_TEMPLATE="${PEER_INIT_CHECKPOINT_TEMPLATE:-$DEFAULT_PEER_INIT_CHECKPOINT_TEMPLATE}"

mkdir -p "$LOG_DIR"

run_case() {
  local label="$1"
  local methods="$2"
  local imitation_loss="$3"
  local lambda_imitation="$4"
  local warmup="$5"
  local decay_start="$6"
  local decay_end="$7"
  local decay_min_scale="$8"
  local granularity="$9"
  local lr="${10}"

  run_locked_job \
    "operator_darcy_student_lift_v2/$label" \
    "operator_darcy_student_lift_v2/$label" \
    "$LOG_DIR/${label}.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      OUTPUT_DIR="$OUTPUT_ROOT/$label" \
      METHODS="$methods" \
      DATASETS="darcy" \
      MODEL_PAIRS="deeponet:fno" \
      INDEPENDENT_MODELS="deeponet" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      LR="$lr" \
      NUM_WORKERS="$NUM_WORKERS" \
      REGRESSION_IMITATION_LOSS="$imitation_loss" \
      LAMBDA_IMITATION="$lambda_imitation" \
      MARGIN="0.0" \
      WARMUP_EPOCHS="$warmup" \
      IMITATION_DECAY_START_EPOCH="$decay_start" \
      IMITATION_DECAY_END_EPOCH="$decay_end" \
      IMITATION_DECAY_MIN_SCALE="$decay_min_scale" \
      HETERO_SSML_ONE_WAY="1" \
      SSML_STUDENT_ONLY="1" \
      SSML_FREEZE_PEER="1" \
      OPERATOR_WEIGHT_GRANULARITY="$granularity" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      DOWNLOAD="0" \
      bash scripts/paper_rerun/run_core_operator.sh
}

echo "[operator_darcy_student_lift_v2] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS"
echo "[operator_darcy_student_lift_v2] init_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[operator_darcy_student_lift_v2] peer_init_template=$PEER_INIT_CHECKPOINT_TEMPLATE"

run_case "ctrl_ft_lr2e4" "independent" "mse" "0.0" "0" "0" "0" "1.0" "sample" "$BASE_LR"
run_case "sample_ft_l003_w15_d45_90_lr2e4" "ssml" "mse" "0.03" "15" "45" "90" "0.10" "sample" "$BASE_LR"
run_case "elem_ft_l002_w20_d50_95_lr2e4" "ssml" "mse" "0.02" "20" "50" "95" "0.10" "element" "$BASE_LR"
run_case "elem_ft_huber_l0015_w25_d55_95_lr15e4" "ssml" "huber" "0.015" "25" "55" "95" "0.10" "element" "$HUBER_LR"

echo "[operator_darcy_student_lift_v2] done"
