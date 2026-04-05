#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/operator_burgers_fno_polish_fair_v2}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-180}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/operator_burgers_fno_polish_fair_v2}"
CASE_SPECS="${CASE_SPECS:-fno_polish_coarse_l002_w24_d90_170:0.020:0.00:24:90:170:0.05:coarse:element fno_polish_hotspot_l003_w20_d80_160:0.030:0.00:20:80:160:0.05:hotspot:element fno_polish_full_l0015_w30_d100_180:0.015:0.00:30:100:180:0.05:full:sample}"

if [[ -z "${INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  INIT_CHECKPOINT_TEMPLATE='results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno_independent_mse_seed{seed}/model.pt'
fi
if [[ -z "${PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  PEER_INIT_CHECKPOINT_TEMPLATE='results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/deeponet_independent_mse_seed{seed}/model.pt'
fi

run_case() {
  local label="$1"
  shift
  run_locked_job \
    "operator_burgers_fno_polish_fair_v2/$label" \
    "operator_burgers_fno_polish_fair_v2/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      OUTPUT_DIR="$OUTPUT_ROOT/$label" \
      DATASETS="burgers" \
      METHODS="ssml" \
      MODEL_PAIRS="fno:deeponet" \
      REQUIRE_DISTINCT_PEER="1" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      REGRESSION_IMITATION_LOSS="mse" \
      SSML_STUDENT_ONLY="1" \
      SSML_FREEZE_PEER="1" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      "$@" \
      bash scripts/paper_rerun/run_core_operator.sh
}

echo "[operator_burgers_fno_polish_fair_v2] output_root=$OUTPUT_ROOT"
echo "[operator_burgers_fno_polish_fair_v2] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "[operator_burgers_fno_polish_fair_v2] init_checkpoint_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[operator_burgers_fno_polish_fair_v2] peer_init_checkpoint_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[operator_burgers_fno_polish_fair_v2] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label lambda margin warmup decay_start decay_end decay_min_scale hint_mode granularity <<< "$spec"
  run_case \
    "$label" \
    LAMBDA_IMITATION="$lambda" \
    MARGIN="$margin" \
    WARMUP_EPOCHS="$warmup" \
    IMITATION_DECAY_START_EPOCH="$decay_start" \
    IMITATION_DECAY_END_EPOCH="$decay_end" \
    IMITATION_DECAY_MIN_SCALE="$decay_min_scale" \
    RELAY_HINT_MODE="$hint_mode" \
    OPERATOR_WEIGHT_GRANULARITY="$granularity"
done

echo "[operator_burgers_fno_polish_fair_v2] done"
