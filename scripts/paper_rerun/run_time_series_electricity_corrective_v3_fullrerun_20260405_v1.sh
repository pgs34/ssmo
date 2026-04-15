#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_electricity_corrective_v3_fullrerun_20260405_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_electricity_corrective_v3_fullrerun_20260405_v1}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-mse}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
SSML_GUIDANCE_MODE="${SSML_GUIDANCE_MODE:-corrective}"
SSML_STUDENT_ONLY="${SSML_STUDENT_ONLY:-1}"
SSML_FREEZE_PEER="${SSML_FREEZE_PEER:-1}"
SSML_CORRECTION_THRESHOLD="${SSML_CORRECTION_THRESHOLD:-0.5}"
CASE_SPECS="${CASE_SPECS:-corr_handoff_e12_l15_sp5e4:0.15:0.0005:64:0.00:0:30:60:0.25:12}"

if [[ -z "${INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  INIT_CHECKPOINT_TEMPLATE='results/time_series_electricity_followup_v1/best_known/time_series/{dataset}/{model}_independent_{regression_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  PEER_INIT_CHECKPOINT_TEMPLATE='results/time_series_electricity_followup_v1/best_known/time_series/{dataset}/{model}_independent_{regression_imitation_loss}_seed{seed}/model.pt'
fi

run_case() {
  local label="$1"
  shift
  run_logged_job \
    "time_series_electricity_corrective_v3_fullrerun_20260405_v1/$label" \
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
      SSML_GUIDANCE_MODE="$SSML_GUIDANCE_MODE" \
      SSML_STUDENT_ONLY="$SSML_STUDENT_ONLY" \
      SSML_FREEZE_PEER="$SSML_FREEZE_PEER" \
      SSML_CORRECTION_THRESHOLD="$SSML_CORRECTION_THRESHOLD" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      LIVE_PLOT_INTERVAL="10" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_electricity_corrective_v3_fullrerun_20260405_v1] output_root=$OUTPUT_ROOT"
echo "[time_series_electricity_corrective_v3_fullrerun_20260405_v1] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS"
echo "[time_series_electricity_corrective_v3_fullrerun_20260405_v1] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label lambda sparsity hidden_dim dropout warmup decay_start decay_end decay_min_scale train_end <<< "$spec"
  run_case \
    "$label" \
    LAMBDA_IMITATION="$lambda" \
    SSML_CORRECTION_SPARSITY_WEIGHT="$sparsity" \
    SSML_CORRECTION_GATE_HIDDEN_DIM="$hidden_dim" \
    SSML_CORRECTION_GATE_DROPOUT="$dropout" \
    WARMUP_EPOCHS="$warmup" \
    IMITATION_DECAY_START_EPOCH="$decay_start" \
    IMITATION_DECAY_END_EPOCH="$decay_end" \
    IMITATION_DECAY_MIN_SCALE="$decay_min_scale" \
    SSML_CORRECTION_STUDENT_TRAIN_END_EPOCH="$train_end" \
    EARLY_STOP_PATIENCE="0" \
    EARLY_STOP_MIN_EPOCHS="0" \
    EARLY_STOP_MIN_DELTA="0.0"
done

echo "[time_series_electricity_corrective_v3_fullrerun_20260405_v1] done"
