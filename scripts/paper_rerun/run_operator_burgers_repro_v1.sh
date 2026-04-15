#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

GPU="${GPU:-1}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/operator_burgers_repro_v1}"
LOG_DIR="${LOG_DIR:-results/logs/operator_burgers_repro_v1}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-180}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LIVE_PLOT_INTERVAL="${LIVE_PLOT_INTERVAL:-10}"
BASE_BURGERS_ROOT="${BASE_BURGERS_ROOT:-results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers}"
DEFAULT_INIT_CHECKPOINT_TEMPLATE="${BASE_BURGERS_ROOT}/fno_independent_mse_seed"'{'seed'}'"/model.pt"
DEFAULT_PEER_INIT_CHECKPOINT_TEMPLATE="${BASE_BURGERS_ROOT}/deeponet_independent_mse_seed"'{'seed'}'"/model.pt"
INIT_CHECKPOINT_TEMPLATE="${INIT_CHECKPOINT_TEMPLATE:-$DEFAULT_INIT_CHECKPOINT_TEMPLATE}"
PEER_INIT_CHECKPOINT_TEMPLATE="${PEER_INIT_CHECKPOINT_TEMPLATE:-$DEFAULT_PEER_INIT_CHECKPOINT_TEMPLATE}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
BURGERS_FILE="${BURGERS_FILE:-$DATA_ROOT/burgers_data_R10.mat}"
BURGERS_ZIP_FILE="${BURGERS_ZIP_FILE:-$DATA_ROOT/burgers_data_R10.mat.zip}"

CASE_LABEL="cos_relay_full_l0012_s20_70_40_sample_lr4e4"

mkdir -p "$LOG_DIR"

ensure_burgers_data() {
  mkdir -p "$DATA_ROOT"
  if [[ -f "$BURGERS_FILE" ]]; then
    echo "[operator_burgers_repro_v1] burgers data present: $BURGERS_FILE"
    return 0
  fi

  if [[ -f "$BURGERS_ZIP_FILE" ]]; then
    echo "[operator_burgers_repro_v1] restoring burgers data from local archive: $BURGERS_ZIP_FILE"
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

  if [[ ! -f "$BURGERS_FILE" ]]; then
    echo "[operator_burgers_repro_v1] missing $BURGERS_FILE" >&2
    return 1
  fi
}

echo "[operator_burgers_repro_v1] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "[operator_burgers_repro_v1] output_root=$OUTPUT_ROOT"
echo "[operator_burgers_repro_v1] case_label=$CASE_LABEL"
echo "[operator_burgers_repro_v1] init_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[operator_burgers_repro_v1] peer_init_template=$PEER_INIT_CHECKPOINT_TEMPLATE"

ensure_burgers_data

run_locked_job \
  "operator_burgers_repro_v1/$CASE_LABEL" \
  "operator_burgers_repro_v1/$CASE_LABEL" \
  "$LOG_DIR/${CASE_LABEL}.log" \
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    DEVICE="$DEVICE" \
    OUTPUT_DIR="$OUTPUT_ROOT/$CASE_LABEL" \
    METHODS="ssml" \
    DATASETS="burgers" \
    MODEL_PAIRS="fno:deeponet" \
    INDEPENDENT_MODELS="fno" \
    REQUIRE_DISTINCT_PEER="1" \
    SEEDS="$SEEDS" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    LR="4e-4" \
    WEIGHT_DECAY="0.0" \
    LR_SCHEDULER="cosine" \
    SCHEDULER_WARMUP_EPOCHS="10" \
    SCHEDULER_MIN_SCALE="0.02" \
    GRAD_CLIP="1.0" \
    REGRESSION_IMITATION_LOSS="mse" \
    LAMBDA_IMITATION="0.012" \
    MARGIN="0.0" \
    WARMUP_EPOCHS="0" \
    IMITATION_DECAY_START_EPOCH="-1" \
    IMITATION_DECAY_END_EPOCH="-1" \
    IMITATION_DECAY_MIN_SCALE="1.0" \
    SSML_STUDENT_ONLY="1" \
    SSML_FREEZE_PEER="1" \
    OPERATOR_WEIGHT_GRANULARITY="sample" \
    RELAY_HINT_MODE="full" \
    RELAY_STAGE_EPOCHS="20,70,40" \
    RELAY_TAPER_SCHEDULE="linear" \
    INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
    PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
    SAVE_BEST_CHECKPOINT="1" \
    LIVE_PLOT_INTERVAL="$LIVE_PLOT_INTERVAL" \
    DOWNLOAD="0" \
    bash scripts/paper_rerun/run_core_operator.sh

echo "[operator_burgers_repro_v1] done"
