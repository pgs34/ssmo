#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_correction_only_v5}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_correction_only_v5}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-mse}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-8}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
CASE_SPECS="${CASE_SPECS:-corronly_tail60_q70_lr8e4:0.0008:0.08:0.0002:64:0.00:-2.5:1:6:0.60:0.70:2.0:9 corronly_tail50_q60_lr1e3:0.0010:0.10:0.0001:64:0.00:-2.0:1:5:0.50:0.60:1.5:13 corronly_tail66_q80_lr6e4:0.0006:0.08:0.0003:96:0.00:-3.0:2:8:0.66:0.80:3.0:9}"

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
    "time_series_etth1_correction_only_v5/$label" \
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
      SSML_GUIDANCE_MODE="corrective" \
      SSML_STUDENT_ONLY="1" \
      SSML_FREEZE_PEER="1" \
      SSML_CORRECTION_ONLY="1" \
      SSML_CORRECTION_FEATURE_MODE="trend_residual" \
      SSML_CORRECTION_USE_REGIME_FEATURES="1" \
      SSML_CORRECTION_FREEZE_STUDENT_EPOCHS="0" \
      WARMUP_EPOCHS="0" \
      IMITATION_DECAY_START_EPOCH="12" \
      IMITATION_DECAY_END_EPOCH="40" \
      IMITATION_DECAY_MIN_SCALE="0.50" \
      EARLY_STOP_PATIENCE="$EARLY_STOP_PATIENCE" \
      EARLY_STOP_MIN_EPOCHS="$EARLY_STOP_MIN_EPOCHS" \
      EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_etth1_correction_only_v5] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_correction_only_v5] gpu=$GPU seeds=$SEEDS batch_size=$BATCH_SIZE"
echo "[time_series_etth1_correction_only_v5] model_pairs=$MODEL_PAIRS"
echo "[time_series_etth1_correction_only_v5] init_checkpoint_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_correction_only_v5] peer_init_checkpoint_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_correction_only_v5] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label lr lambda sparsity hidden_dim dropout init_bias ramp_start ramp_end tail_start regime_q focus_alpha decomp_kernel <<< "$spec"
  run_case \
    "$label" \
    LR="$lr" \
    LAMBDA_IMITATION="$lambda" \
    SSML_CORRECTION_SPARSITY_WEIGHT="$sparsity" \
    SSML_CORRECTION_GATE_HIDDEN_DIM="$hidden_dim" \
    SSML_CORRECTION_GATE_DROPOUT="$dropout" \
    SSML_CORRECTION_INIT_BIAS="$init_bias" \
    SSML_CORRECTION_RAMP_START_EPOCH="$ramp_start" \
    SSML_CORRECTION_RAMP_END_EPOCH="$ramp_end" \
    SSML_CORRECTION_TAIL_START_RATIO="$tail_start" \
    SSML_CORRECTION_REGIME_FOCUS_QUANTILE="$regime_q" \
    SSML_CORRECTION_FOCUS_LOSS_ALPHA="$focus_alpha" \
    SSML_CORRECTION_DECOMPOSITION_KERNEL="$decomp_kernel"
done
