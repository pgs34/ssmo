#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/operator_darcy_relay_v1}"
LOG_DIR="${LOG_DIR:-results/logs/operator_darcy_relay_v1}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-110}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BASE_DARCY_ROOT="${BASE_DARCY_ROOT:-results/operator_ssml_tuned_v1/operator/darcy}"
DEFAULT_INIT_CHECKPOINT_TEMPLATE="${BASE_DARCY_ROOT}/deeponet_independent_mse_seed"'{'seed'}'"/model.pt"
DEFAULT_PEER_INIT_CHECKPOINT_TEMPLATE="${BASE_DARCY_ROOT}/fno_independent_mse_seed"'{'seed'}'"/model.pt"
INIT_CHECKPOINT_TEMPLATE="${INIT_CHECKPOINT_TEMPLATE:-$DEFAULT_INIT_CHECKPOINT_TEMPLATE}"
PEER_INIT_CHECKPOINT_TEMPLATE="${PEER_INIT_CHECKPOINT_TEMPLATE:-$DEFAULT_PEER_INIT_CHECKPOINT_TEMPLATE}"
CASE_SPECS="${CASE_SPECS:-darcy_relay_coarse:0.00020:0.025:20,35,30:coarse:linear darcy_relay_hotspot:0.00015:0.020:25,30,30:hotspot:cosine}"

mkdir -p "$LOG_DIR"

run_case() {
  local label="$1"
  shift
  run_locked_job \
    "operator_darcy_relay_v1/$label" \
    "operator_darcy_relay_v1/$label" \
    "$LOG_DIR/${label}.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      OUTPUT_DIR="$OUTPUT_ROOT/$label" \
      METHODS="ssml" \
      DATASETS="darcy" \
      MODEL_PAIRS="deeponet:fno" \
      INDEPENDENT_MODELS="deeponet" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      REGRESSION_IMITATION_LOSS="mse" \
      MARGIN="0.0" \
      WARMUP_EPOCHS="0" \
      HETERO_SSML_ONE_WAY="1" \
      SSML_STUDENT_ONLY="1" \
      SSML_FREEZE_PEER="1" \
      OPERATOR_WEIGHT_GRANULARITY="element" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      DOWNLOAD="0" \
      LIVE_PLOT_INTERVAL="10" \
      "$@" \
      bash scripts/paper_rerun/run_core_operator.sh
}

echo "[operator_darcy_relay_v1] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS"
echo "[operator_darcy_relay_v1] init_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[operator_darcy_relay_v1] peer_init_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[operator_darcy_relay_v1] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label lr lambda relay_stage_epochs relay_hint_mode relay_taper_schedule <<< "$spec"
  run_case \
    "$label" \
    LR="$lr" \
    LAMBDA_IMITATION="$lambda" \
    RELAY_STAGE_EPOCHS="$relay_stage_epochs" \
    RELAY_HINT_MODE="$relay_hint_mode" \
    RELAY_TAPER_SCHEDULE="$relay_taper_schedule"
done

echo "[operator_darcy_relay_v1] done"
