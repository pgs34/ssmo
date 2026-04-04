#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/operator_burgers_student_lift_v2}"
LOG_DIR="${LOG_DIR:-results/logs/operator_burgers_student_lift_v2}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-140}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BASE_LR="${BASE_LR:-4e-4}"
HUBER_LR="${HUBER_LR:-3e-4}"
BASE_BURGERS_ROOT="${BASE_BURGERS_ROOT:-results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
BURGERS_FILE="${BURGERS_FILE:-$DATA_ROOT/burgers_data_R10.mat}"
BURGERS_ZIP_FILE="${BURGERS_ZIP_FILE:-$DATA_ROOT/burgers_data_R10.mat.zip}"
DEFAULT_INIT_CHECKPOINT_TEMPLATE="${BASE_BURGERS_ROOT}/deeponet_independent_mse_seed"'{'seed'}'"/model.pt"
DEFAULT_PEER_INIT_CHECKPOINT_TEMPLATE="${BASE_BURGERS_ROOT}/fno_independent_mse_seed"'{'seed'}'"/model.pt"
INIT_CHECKPOINT_TEMPLATE="${INIT_CHECKPOINT_TEMPLATE:-$DEFAULT_INIT_CHECKPOINT_TEMPLATE}"
PEER_INIT_CHECKPOINT_TEMPLATE="${PEER_INIT_CHECKPOINT_TEMPLATE:-$DEFAULT_PEER_INIT_CHECKPOINT_TEMPLATE}"

mkdir -p "$LOG_DIR"

ensure_burgers_data() {
  mkdir -p "$DATA_ROOT"
  if [[ -f "$BURGERS_FILE" ]]; then
    echo "[operator_burgers_student_lift_v2] burgers data present: $BURGERS_FILE"
    return 0
  fi
  if [[ -f "$BURGERS_ZIP_FILE" ]]; then
    echo "[operator_burgers_student_lift_v2] restoring burgers data from local archive: $BURGERS_ZIP_FILE"
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
    echo "[operator_burgers_student_lift_v2] missing $BURGERS_FILE" >&2
    return 1
  fi
}

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
    "operator_burgers_student_lift_v2/$label" \
    "operator_burgers_student_lift_v2/$label" \
    "$LOG_DIR/${label}.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      OUTPUT_DIR="$OUTPUT_ROOT/$label" \
      METHODS="$methods" \
      DATASETS="burgers" \
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

echo "[operator_burgers_student_lift_v2] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS"
echo "[operator_burgers_student_lift_v2] init_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[operator_burgers_student_lift_v2] peer_init_template=$PEER_INIT_CHECKPOINT_TEMPLATE"

ensure_burgers_data

run_case "ctrl_ft_lr4e4" "independent" "mse" "0.0" "0" "0" "0" "1.0" "sample" "$BASE_LR"
run_case "sample_ft_l004_w20_d60_120_lr4e4" "ssml" "mse" "0.04" "20" "60" "120" "0.10" "sample" "$BASE_LR"
run_case "elem_ft_l003_w25_d70_130_lr4e4" "ssml" "mse" "0.03" "25" "70" "130" "0.10" "element" "$BASE_LR"
run_case "elem_ft_huber_l002_w30_d80_140_lr3e4" "ssml" "huber" "0.02" "30" "80" "140" "0.10" "element" "$HUBER_LR"

echo "[operator_burgers_student_lift_v2] done"
