#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_electricity_snapshot_anchor_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_electricity_snapshot_anchor_v1}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-mse}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
CASE_SPECS="${CASE_SPECS:-esa_w15_a10_t18_r50_20:0.15:10:18:0.50:0.20 esa_w20_a10_t20_r55_22:0.20:10:20:0.55:0.22 esa_w25_a12_t22_r60_25:0.25:12:22:0.60:0.25}"

if [[ -z "${INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  INIT_CHECKPOINT_TEMPLATE='results/time_series_electricity_followup_v1/best_known/time_series/{dataset}/{model}_independent_{regression_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  PEER_INIT_CHECKPOINT_TEMPLATE='results/time_series_electricity_followup_v1/best_known/time_series/{dataset}/{model}_independent_{regression_imitation_loss}_seed{seed}/model.pt'
fi

run_case() {
  local label="$1"
  shift
  run_locked_job \
    "time_series_electricity_snapshot_anchor_v1/$label" \
    "time_series_electricity_snapshot_anchor_v1/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      OUTPUT_DIR="$OUTPUT_ROOT/$label" \
      DATASETS="electricity" \
      METHODS="ssml" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="1" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      SEQ_LEN="$SEQ_LEN" \
      PRED_LENS="$PRED_LENS" \
      REGRESSION_IMITATION_LOSS="$REGRESSION_IMITATION_LOSS" \
      FEATURE_MODE="$FEATURE_MODE" \
      SSML_GUIDANCE_MODE="corrective" \
      SSML_STUDENT_ONLY="1" \
      SSML_FREEZE_PEER="1" \
      SSML_CORRECTION_THRESHOLD="${SSML_CORRECTION_THRESHOLD:-0.5}" \
      SSML_CORRECTION_SPARSITY_WEIGHT="${SSML_CORRECTION_SPARSITY_WEIGHT:-0.0005}" \
      SSML_CORRECTION_GATE_HIDDEN_DIM="${SSML_CORRECTION_GATE_HIDDEN_DIM:-64}" \
      SSML_CORRECTION_GATE_DROPOUT="${SSML_CORRECTION_GATE_DROPOUT:-0.0}" \
      SSML_CORRECTION_STUDENT_TRAIN_END_EPOCH="-1" \
      WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}" \
      IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-30}" \
      IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-60}" \
      IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.25}" \
      SSML_HANDOFF_END_EPOCH="-1" \
      SSML_SNAPSHOT_ANCHOR_MASK_MODE="${SSML_SNAPSHOT_ANCHOR_MASK_MODE:-selected}" \
      SSML_ACTIVE_RATIO_ADAPT_RATE="${SSML_ACTIVE_RATIO_ADAPT_RATE:-0.50}" \
      EARLY_STOP_PATIENCE="0" \
      EARLY_STOP_MIN_EPOCHS="0" \
      EARLY_STOP_MIN_DELTA="0.0" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      LIVE_PLOT_INTERVAL="10" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_electricity_snapshot_anchor_v1] output_root=$OUTPUT_ROOT"
echo "[time_series_electricity_snapshot_anchor_v1] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS"
echo "[time_series_electricity_snapshot_anchor_v1] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label anchor_weight anchor_start taper_end ratio_start ratio_end <<< "$spec"
  run_case \
    "$label" \
    LAMBDA_IMITATION="${LAMBDA_IMITATION:-0.15}" \
    SSML_SNAPSHOT_ANCHOR_WEIGHT="$anchor_weight" \
    SSML_SNAPSHOT_ANCHOR_START_EPOCH="$anchor_start" \
    SSML_PEER_TAPER_END_EPOCH="$taper_end" \
    SSML_TARGET_ACTIVE_RATIO_START="$ratio_start" \
    SSML_TARGET_ACTIVE_RATIO_END="$ratio_end"
done

echo "[time_series_electricity_snapshot_anchor_v1] done"
