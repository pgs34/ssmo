#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/operator_burgers_followup_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-180}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/operator_burgers_followup_v1}"
SUMMARY_PLOT_ROOT="${SUMMARY_PLOT_ROOT:-results/plots/operator_burgers_followup_v1}"
REFRESH_TOP_LEVEL="${REFRESH_TOP_LEVEL:-0}"
CASE_SPECS="${CASE_SPECS:-burgers_l005_m0_w12_d60_120_ow1:0.05:0.00:12:60:120:0.10:1 burgers_l010_m0_w10_d50_110_ow1:0.10:0.00:10:50:110:0.10:1 burgers_l015_m0_w8_d40_100_ow1:0.15:0.00:8:40:100:0.10:1}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data}"
BURGERS_FILE="${BURGERS_FILE:-$DATA_ROOT/burgers_data_R10.mat}"
BURGERS_ZIP_FILE="${BURGERS_ZIP_FILE:-$DATA_ROOT/burgers_data_R10.mat.zip}"
BURGERS_GDOWN_ID="${BURGERS_GDOWN_ID:-}"
BURGERS_GDOWN_URL="${BURGERS_GDOWN_URL:-}"

ensure_burgers_data() {
  mkdir -p "$DATA_ROOT"
  if [[ -f "$BURGERS_FILE" ]]; then
    echo "[operator_burgers_followup_v1] burgers data present: $BURGERS_FILE"
    return 0
  fi

  if [[ -f "$BURGERS_ZIP_FILE" ]]; then
    echo "[operator_burgers_followup_v1] restoring burgers data from local archive: $BURGERS_ZIP_FILE"
    python - "$BURGERS_ZIP_FILE" <<'PY'
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
    echo "[operator_burgers_followup_v1] burgers data restored: $BURGERS_FILE"
    return 0
  fi

  if [[ -n "$BURGERS_GDOWN_ID" ]]; then
    echo "[operator_burgers_followup_v1] downloading burgers data via gdown id=$BURGERS_GDOWN_ID"
    gdown --id "$BURGERS_GDOWN_ID" -O "$BURGERS_FILE"
  elif [[ -n "$BURGERS_GDOWN_URL" ]]; then
    echo "[operator_burgers_followup_v1] downloading burgers data via gdown url"
    gdown "$BURGERS_GDOWN_URL" -O "$BURGERS_FILE"
  fi

  if [[ ! -f "$BURGERS_FILE" ]]; then
    echo "[operator_burgers_followup_v1] missing $BURGERS_FILE and no usable download source was configured" >&2
    return 1
  fi

  echo "[operator_burgers_followup_v1] burgers data ready: $BURGERS_FILE"
}

run_case() {
  local label="$1"
  shift
  run_logged_job \
    "operator_burgers_followup_v1/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      OUTPUT_DIR="$OUTPUT_ROOT/$label" \
      DATASETS="burgers" \
      METHODS="independent dml ssml" \
      MODEL_PAIRS="fno:deeponet" \
      INDEPENDENT_MODELS="fno deeponet" \
      REQUIRE_DISTINCT_PEER="1" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      REGRESSION_IMITATION_LOSS="mse" \
      "$@" \
      bash scripts/paper_rerun/run_core_operator.sh
}

echo "[operator_burgers_followup_v1] output_root=$OUTPUT_ROOT"
echo "[operator_burgers_followup_v1] summary_plot_root=$SUMMARY_PLOT_ROOT"
echo "[operator_burgers_followup_v1] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS"
echo "[operator_burgers_followup_v1] case_specs=$CASE_SPECS"

ensure_burgers_data

for spec in $CASE_SPECS; do
  IFS=':' read -r label lambda margin warmup decay_start decay_end decay_min_scale hetero_one_way <<< "$spec"
  run_case \
    "$label" \
    LAMBDA_IMITATION="$lambda" \
    MARGIN="$margin" \
    WARMUP_EPOCHS="$warmup" \
    IMITATION_DECAY_START_EPOCH="$decay_start" \
    IMITATION_DECAY_END_EPOCH="$decay_end" \
    IMITATION_DECAY_MIN_SCALE="$decay_min_scale" \
    HETERO_SSML_ONE_WAY="$hetero_one_way"
done

echo "[operator_burgers_followup_v1] refreshing summary plots"
bash scripts/paper_rerun/refresh_summary_plots.sh "$OUTPUT_ROOT" "$SUMMARY_PLOT_ROOT"

if [[ "$REFRESH_TOP_LEVEL" == "1" ]]; then
  echo "[operator_burgers_followup_v1] refreshing top-level plots"
  bash scripts/paper_rerun/refresh_top_level_best_plots.sh
fi

echo "[operator_burgers_followup_v1] done"
