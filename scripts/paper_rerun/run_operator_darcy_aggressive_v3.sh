#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/operator_darcy_aggressive_v3}"
LOG_DIR="${LOG_DIR:-results/logs/operator_darcy_aggressive_v3}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-110}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LIVE_PLOT_INTERVAL="${LIVE_PLOT_INTERVAL:-10}"
CASE_FILTER="${CASE_FILTER:-}"
BASE_DARCY_ROOT="${BASE_DARCY_ROOT:-results/operator_ssml_tuned_v1/operator/darcy}"
DEFAULT_INIT_CHECKPOINT_TEMPLATE="${BASE_DARCY_ROOT}/deeponet_independent_mse_seed"'{'seed'}'"/model.pt"
DEFAULT_PEER_INIT_CHECKPOINT_TEMPLATE="${BASE_DARCY_ROOT}/fno_independent_mse_seed"'{'seed'}'"/model.pt"
INIT_CHECKPOINT_TEMPLATE="${INIT_CHECKPOINT_TEMPLATE:-$DEFAULT_INIT_CHECKPOINT_TEMPLATE}"
PEER_INIT_CHECKPOINT_TEMPLATE="${PEER_INIT_CHECKPOINT_TEMPLATE:-$DEFAULT_PEER_INIT_CHECKPOINT_TEMPLATE}"

mkdir -p "$LOG_DIR"

should_run_case() {
  local label="$1"
  if [[ -z "$CASE_FILTER" ]]; then
    return 0
  fi
  [[ " $CASE_FILTER " == *" $label "* ]]
}

run_case() {
  local label="$1"
  local methods="$2"
  local lr="$3"
  local lr_scheduler="$4"
  local scheduler_warmup_epochs="$5"
  local scheduler_min_scale="$6"
  local grad_clip="$7"
  local lambda_imitation="$8"
  local warmup_epochs="$9"
  local decay_start="${10}"
  local decay_end="${11}"
  local decay_min_scale="${12}"
  local hint_mode="${13}"
  local granularity="${14}"
  local relay_stage_epochs="${15}"

  if ! should_run_case "$label"; then
    echo "[operator_darcy_aggressive_v3] skip case=$label filter=$CASE_FILTER"
    return 0
  fi

  run_locked_job \
    "operator_darcy_aggressive_v3/$label" \
    "operator_darcy_aggressive_v3/$label" \
    "$LOG_DIR/${label}.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      OUTPUT_DIR="$OUTPUT_ROOT/$label" \
      METHODS="$methods" \
      DATASETS="darcy" \
      MODEL_PAIRS="deeponet:fno" \
      INDEPENDENT_MODELS="deeponet" \
      REQUIRE_DISTINCT_PEER="1" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      LR="$lr" \
      WEIGHT_DECAY="0.0" \
      LR_SCHEDULER="$lr_scheduler" \
      SCHEDULER_WARMUP_EPOCHS="$scheduler_warmup_epochs" \
      SCHEDULER_MIN_SCALE="$scheduler_min_scale" \
      GRAD_CLIP="$grad_clip" \
      REGRESSION_IMITATION_LOSS="mse" \
      LAMBDA_IMITATION="$lambda_imitation" \
      MARGIN="0.0" \
      WARMUP_EPOCHS="$warmup_epochs" \
      IMITATION_DECAY_START_EPOCH="$decay_start" \
      IMITATION_DECAY_END_EPOCH="$decay_end" \
      IMITATION_DECAY_MIN_SCALE="$decay_min_scale" \
      HETERO_SSML_ONE_WAY="1" \
      SSML_STUDENT_ONLY="1" \
      SSML_FREEZE_PEER="1" \
      OPERATOR_WEIGHT_GRANULARITY="$granularity" \
      RELAY_HINT_MODE="$hint_mode" \
      RELAY_STAGE_EPOCHS="$relay_stage_epochs" \
      RELAY_TAPER_SCHEDULE="linear" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      SAVE_BEST_CHECKPOINT="1" \
      LIVE_PLOT_INTERVAL="$LIVE_PLOT_INTERVAL" \
      DOWNLOAD="0" \
      bash scripts/paper_rerun/run_core_operator.sh
}

echo "[operator_darcy_aggressive_v3] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "[operator_darcy_aggressive_v3] output_root=$OUTPUT_ROOT"
echo "[operator_darcy_aggressive_v3] init_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[operator_darcy_aggressive_v3] peer_init_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[operator_darcy_aggressive_v3] case_filter=$CASE_FILTER"

run_case "ctrl_const_lr2e4_clip1" "independent" "2e-4" "none" "0" "0.0" "1.0" "0.0" "0" "-1" "-1" "1.0" "full" "sample" ""
run_case "ctrl_cos_lr15e4_w10_min05_clip1" "independent" "1.5e-4" "cosine" "10" "0.05" "1.0" "0.0" "0" "-1" "-1" "1.0" "full" "sample" ""
run_case "const_full_l015_w20_d50_95_sample_lr2e4" "ssml" "2e-4" "none" "0" "0.0" "1.0" "0.015" "20" "50" "95" "0.10" "full" "sample" ""
run_case "const_coarse_l020_w20_d50_95_element_lr2e4" "ssml" "2e-4" "none" "0" "0.0" "1.0" "0.020" "20" "50" "95" "0.10" "coarse" "element" ""
run_case "cos_relay_full_l015_s15_35_40_sample_lr15e4" "ssml" "1.5e-4" "cosine" "10" "0.05" "1.0" "0.015" "0" "-1" "-1" "1.0" "full" "sample" "15,35,40"
run_case "cos_relay_coarse_l020_s15_35_40_element_lr15e4" "ssml" "1.5e-4" "cosine" "10" "0.05" "1.0" "0.020" "0" "-1" "-1" "1.0" "coarse" "element" "15,35,40"

echo "[operator_darcy_aggressive_v3] done"
