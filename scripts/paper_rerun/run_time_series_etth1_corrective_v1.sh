#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_corrective_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_corrective_v1}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-mse}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
SSML_GUIDANCE_MODE="${SSML_GUIDANCE_MODE:-corrective}"
SSML_STUDENT_ONLY="${SSML_STUDENT_ONLY:-1}"
SSML_FREEZE_PEER="${SSML_FREEZE_PEER:-1}"
SSML_CORRECTION_THRESHOLD="${SSML_CORRECTION_THRESHOLD:-0.5}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-10}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
CASE_SPECS="${CASE_SPECS:-corr_lr3e4_l10_sp5e4_h64:0.0003:0.10:0.0005:64:0.00:0:15:60:0.25 corr_lr1e4_l15_sp5e4_h64:0.0001:0.15:0.0005:64:0.00:0:20:70:0.25 corr_lr3e4_l5_sp2e3_h32:0.0003:0.05:0.0020:32:0.00:0:10:40:0.20 corr_lr3e4_l10_sp1e3_h64_do10:0.0003:0.10:0.0010:64:0.10:0:15:60:0.25}"

if [[ -z "${INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  INIT_CHECKPOINT_TEMPLATE='results/paper_rerun_canonical/time_series/time_series/{dataset}/{model}_independent_mse_seed{seed}/model.pt'
fi
if [[ -z "${PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  PEER_INIT_CHECKPOINT_TEMPLATE='results/paper_rerun_canonical/time_series/time_series/{dataset}/{model}_independent_mse_seed{seed}/model.pt'
fi

run_case() {
  local label="$1"
  shift
  run_logged_job \
    "time_series_etth1_corrective_v1/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      OUTPUT_DIR="$OUTPUT_ROOT/$label" \
      DATASETS="etth1" \
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
      EARLY_STOP_PATIENCE="$EARLY_STOP_PATIENCE" \
      EARLY_STOP_MIN_EPOCHS="$EARLY_STOP_MIN_EPOCHS" \
      EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_etth1_corrective_v1] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_corrective_v1] gpu=$GPU seeds=$SEEDS"
echo "[time_series_etth1_corrective_v1] model_pairs=$MODEL_PAIRS"
echo "[time_series_etth1_corrective_v1] init_checkpoint_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_corrective_v1] peer_init_checkpoint_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_corrective_v1] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label lr lambda sparsity hidden_dim dropout warmup decay_start decay_end decay_min_scale <<< "$spec"
  run_case \
    "$label" \
    LR="$lr" \
    LAMBDA_IMITATION="$lambda" \
    SSML_CORRECTION_SPARSITY_WEIGHT="$sparsity" \
    SSML_CORRECTION_GATE_HIDDEN_DIM="$hidden_dim" \
    SSML_CORRECTION_GATE_DROPOUT="$dropout" \
    WARMUP_EPOCHS="$warmup" \
    IMITATION_DECAY_START_EPOCH="$decay_start" \
    IMITATION_DECAY_END_EPOCH="$decay_end" \
    IMITATION_DECAY_MIN_SCALE="$decay_min_scale"
done
