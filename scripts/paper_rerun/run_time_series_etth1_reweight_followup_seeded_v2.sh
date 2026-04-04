#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_reweight_followup_seeded_v2}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-huber}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_reweight_followup_seeded_v2}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LIVE_PLOT_INTERVAL="${LIVE_PLOT_INTERVAL:-10}"
CASE_SPECS="${CASE_SPECS:-rw_a025_e18_top20_l1e3:18:0.25:0.0010:0.020:0.020:1:3:8:0.00:0:0 rw_a050_e18_top20_l1e3:18:0.50:0.0010:0.020:0.020:1:3:8:0.00:0:0 rw_a035_e24_top15_l8e4:24:0.35:0.0008:0.020:0.015:2:5:12:0.00:0:0}"

if [[ -z "${INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  INIT_CHECKPOINT_TEMPLATE='results/paper_rerun_canonical/time_series/time_series/{dataset}/{model}_independent_mse_seed{seed}/model.pt'
fi
if [[ -z "${PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  PEER_INIT_CHECKPOINT_TEMPLATE='results/paper_rerun_canonical/time_series/time_series/{dataset}/{model}_independent_mse_seed{seed}/model.pt'
fi

run_case() {
  local label="$1"
  shift
  run_locked_job \
    "${OUTPUT_ROOT}/${label}/seeds_${SEEDS}" \
    "time_series_etth1_reweight_followup_seeded_v2/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      METHODS="ssml" \
      DATASETS="etth1" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="1" \
      SEEDS="$SEEDS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      SEQ_LEN="$SEQ_LEN" \
      PRED_LENS="$PRED_LENS" \
      FEATURE_MODE="$FEATURE_MODE" \
      REGRESSION_IMITATION_LOSS="$REGRESSION_IMITATION_LOSS" \
      SSML_GUIDANCE_MODE="reweight_only" \
      SSML_GATE_SCORE_MODE="peer_better_student_error" \
      SSML_SCORE_TRANSFORM="none" \
      SSML_TOPK_SCOPE="total" \
      SSML_SUPERVISED_WEIGHT_MODE="score" \
      LIVE_PLOT_INTERVAL="$LIVE_PLOT_INTERVAL" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      OUTPUT_DIR="$OUTPUT_ROOT/$label" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_etth1_reweight_followup_seeded_v2] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_reweight_followup_seeded_v2] gpu=$GPU seeds=$SEEDS"
echo "[time_series_etth1_reweight_followup_seeded_v2] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label epochs alpha lambda margin topk_ratio warmup decay_start decay_end decay_min_scale early_patience early_min_epochs <<< "$spec"
  run_case \
    "$label" \
    EPOCHS="$epochs" \
    SSML_SUPERVISED_HOTSPOT_ALPHA="$alpha" \
    LAMBDA_IMITATION="$lambda" \
    MARGIN="$margin" \
    SSML_TOPK_RATIO="$topk_ratio" \
    WARMUP_EPOCHS="$warmup" \
    IMITATION_DECAY_START_EPOCH="$decay_start" \
    IMITATION_DECAY_END_EPOCH="$decay_end" \
    IMITATION_DECAY_MIN_SCALE="$decay_min_scale" \
    EARLY_STOP_PATIENCE="$early_patience" \
    EARLY_STOP_MIN_EPOCHS="$early_min_epochs" \
    EARLY_STOP_MIN_DELTA="0.0001"
done
