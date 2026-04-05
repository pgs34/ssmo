#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/operator_burgers_relay_v1}"
LOG_DIR="${LOG_DIR:-results/logs/operator_burgers_relay_v1}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-140}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BASE_BURGERS_ROOT="${BASE_BURGERS_ROOT:-results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
BURGERS_FILE="${BURGERS_FILE:-$DATA_ROOT/burgers_data_R10.mat}"
DEFAULT_INIT_CHECKPOINT_TEMPLATE="${BASE_BURGERS_ROOT}/deeponet_independent_mse_seed"'{'seed'}'"/model.pt"
DEFAULT_PEER_INIT_CHECKPOINT_TEMPLATE="${BASE_BURGERS_ROOT}/fno_independent_mse_seed"'{'seed'}'"/model.pt"
INIT_CHECKPOINT_TEMPLATE="${INIT_CHECKPOINT_TEMPLATE:-$DEFAULT_INIT_CHECKPOINT_TEMPLATE}"
PEER_INIT_CHECKPOINT_TEMPLATE="${PEER_INIT_CHECKPOINT_TEMPLATE:-$DEFAULT_PEER_INIT_CHECKPOINT_TEMPLATE}"
CASE_SPECS="${CASE_SPECS:-burgers_relay_coarse:0.00030:0.030:25,45,35:coarse:linear burgers_relay_hotspot:0.00025:0.025:30,40,35:hotspot:cosine}"

mkdir -p "$LOG_DIR"

ensure_burgers_data() {
  if [[ -f "$BURGERS_FILE" ]]; then
    echo "[operator_burgers_relay_v1] burgers data present: $BURGERS_FILE"
    return 0
  fi
  echo "[operator_burgers_relay_v1] missing burgers data: $BURGERS_FILE" >&2
  return 1
}

run_case() {
  local label="$1"
  shift
  run_locked_job \
    "operator_burgers_relay_v1/$label" \
    "operator_burgers_relay_v1/$label" \
    "$LOG_DIR/${label}.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      OUTPUT_DIR="$OUTPUT_ROOT/$label" \
      METHODS="ssml" \
      DATASETS="burgers" \
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

echo "[operator_burgers_relay_v1] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS"
echo "[operator_burgers_relay_v1] init_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[operator_burgers_relay_v1] peer_init_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[operator_burgers_relay_v1] case_specs=$CASE_SPECS"

ensure_burgers_data

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

echo "[operator_burgers_relay_v1] done"
