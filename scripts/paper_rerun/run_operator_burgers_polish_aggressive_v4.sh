#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/operator_burgers_polish_aggressive_v4}"
LOG_DIR="${LOG_DIR:-results/logs/operator_burgers_polish_aggressive_v4}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-180}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LIVE_PLOT_INTERVAL="${LIVE_PLOT_INTERVAL:-10}"
CASE_FILTER="${CASE_FILTER:-}"
BASE_BURGERS_ROOT="${BASE_BURGERS_ROOT:-results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers}"
DEFAULT_INIT_CHECKPOINT_TEMPLATE="${BASE_BURGERS_ROOT}/fno_independent_mse_seed"'{'seed'}'"/model.pt"
DEFAULT_PEER_INIT_CHECKPOINT_TEMPLATE="${BASE_BURGERS_ROOT}/deeponet_independent_mse_seed"'{'seed'}'"/model.pt"
INIT_CHECKPOINT_TEMPLATE="${INIT_CHECKPOINT_TEMPLATE:-$DEFAULT_INIT_CHECKPOINT_TEMPLATE}"
PEER_INIT_CHECKPOINT_TEMPLATE="${PEER_INIT_CHECKPOINT_TEMPLATE:-$DEFAULT_PEER_INIT_CHECKPOINT_TEMPLATE}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
BURGERS_FILE="${BURGERS_FILE:-$DATA_ROOT/burgers_data_R10.mat}"
BURGERS_ZIP_FILE="${BURGERS_ZIP_FILE:-$DATA_ROOT/burgers_data_R10.mat.zip}"
BURGERS_GDOWN_ID="${BURGERS_GDOWN_ID:-}"
BURGERS_GDOWN_URL="${BURGERS_GDOWN_URL:-}"

mkdir -p "$LOG_DIR"

ensure_burgers_data() {
  mkdir -p "$DATA_ROOT"
  if [[ -f "$BURGERS_FILE" ]]; then
    echo "[operator_burgers_polish_aggressive_v4] burgers data present: $BURGERS_FILE"
    return 0
  fi

  if [[ -f "$BURGERS_ZIP_FILE" ]]; then
    echo "[operator_burgers_polish_aggressive_v4] restoring burgers data from local archive: $BURGERS_ZIP_FILE"
    python3 - "$BURGERS_ZIP_FILE" <<'PY'
from pathlib import Path
import sys
import zipfile
zip_path = Path(sys.argv[1])
out_dir = zip_path.parent
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(out_dir)
PY
  fi

  if [[ -f "$BURGERS_FILE" ]]; then
    echo "[operator_burgers_polish_aggressive_v4] burgers data restored: $BURGERS_FILE"
    return 0
  fi

  if [[ -n "$BURGERS_GDOWN_ID" ]]; then
    echo "[operator_burgers_polish_aggressive_v4] downloading burgers data via gdown id=$BURGERS_GDOWN_ID"
    gdown --id "$BURGERS_GDOWN_ID" -O "$BURGERS_FILE"
  elif [[ -n "$BURGERS_GDOWN_URL" ]]; then
    echo "[operator_burgers_polish_aggressive_v4] downloading burgers data via gdown url"
    gdown "$BURGERS_GDOWN_URL" -O "$BURGERS_FILE"
  fi

  if [[ ! -f "$BURGERS_FILE" ]]; then
    echo "[operator_burgers_polish_aggressive_v4] missing $BURGERS_FILE" >&2
    return 1
  fi
}

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
    echo "[operator_burgers_polish_aggressive_v4] skip case=$label filter=$CASE_FILTER"
    return 0
  fi

  run_locked_job \
    "operator_burgers_polish_aggressive_v4/$label" \
    "operator_burgers_polish_aggressive_v4/$label" \
    "$LOG_DIR/${label}.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      OUTPUT_DIR="$OUTPUT_ROOT/$label" \
      METHODS="$methods" \
      DATASETS="burgers" \
      MODEL_PAIRS="fno:deeponet" \
      INDEPENDENT_MODELS="fno" \
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

echo "[operator_burgers_polish_aggressive_v4] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "[operator_burgers_polish_aggressive_v4] output_root=$OUTPUT_ROOT"
echo "[operator_burgers_polish_aggressive_v4] init_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[operator_burgers_polish_aggressive_v4] peer_init_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[operator_burgers_polish_aggressive_v4] case_filter=$CASE_FILTER"

ensure_burgers_data

run_case "ctrl_cos_lr4e4_w10_min02_clip1" "independent" "4e-4" "cosine" "10" "0.02" "1.0" "0.0" "0" "-1" "-1" "1.0" "full" "sample" ""
run_case "const_full_l0012_w30_d110_170_sample_lr5e4" "ssml" "5e-4" "none" "0" "0.0" "1.0" "0.012" "30" "110" "170" "0.02" "full" "sample" ""
run_case "cos_full_l0010_w20_d90_150_sample_lr4e4" "ssml" "4e-4" "cosine" "10" "0.02" "1.0" "0.010" "20" "90" "150" "0.02" "full" "sample" ""
run_case "cos_relay_full_l0012_s20_70_40_sample_lr4e4" "ssml" "4e-4" "cosine" "10" "0.02" "1.0" "0.012" "0" "-1" "-1" "1.0" "full" "sample" "20,70,40"
run_case "cos_relay_coarse_l0008_s20_70_50_element_lr4e4" "ssml" "4e-4" "cosine" "10" "0.02" "1.0" "0.008" "0" "-1" "-1" "1.0" "coarse" "element" "20,70,50"

echo "[operator_burgers_polish_aggressive_v4] done"
