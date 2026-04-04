#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_teacher_ft_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-90}"
BATCH_SIZE="${BATCH_SIZE:-768}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_teacher_ft_v1}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-huber}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-0}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
CASE_SPECS="${CASE_SPECS:-tft_tail20_reg20_l015_lr3e4:0.0003:0.015:0.00002:96:0.00:-0.40:12:30:0.20:0.20:0.20:9:0.10 tft_tail30_reg25_l020_lr4e4:0.0004:0.020:0.00005:96:0.00:-0.70:12:32:0.30:0.25:0.25:9:0.15 tft_tail10_reg15_l010_lr2e4:0.0002:0.010:0.00001:64:0.00:-0.20:10:26:0.10:0.15:0.15:13:0.05 tft_tail40_reg30_l025_lr5e4:0.0005:0.025:0.00005:128:0.00:-0.90:14:34:0.40:0.30:0.30:13:0.20}"

if [[ -z "${INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  INIT_CHECKPOINT_TEMPLATE='results/time_series_etth1_all_methods_long_v3/time_series/etth1/{model}_independent_huber_seed{seed}/best_model.pt'
fi
if [[ -z "${PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  PEER_INIT_CHECKPOINT_TEMPLATE='results/time_series_etth1_all_methods_long_v3/time_series/etth1/{model}_independent_huber_seed{seed}/best_model.pt'
fi

run_case() {
  local label="$1"
  shift
  run_logged_job \
    "time_series_etth1_teacher_ft_v1/$label" \
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
      WARMUP_EPOCHS="5" \
      IMITATION_DECAY_START_EPOCH="28" \
      IMITATION_DECAY_END_EPOCH="70" \
      IMITATION_DECAY_MIN_SCALE="0.40" \
      EARLY_STOP_PATIENCE="$EARLY_STOP_PATIENCE" \
      EARLY_STOP_MIN_EPOCHS="$EARLY_STOP_MIN_EPOCHS" \
      EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      LIVE_PLOT_INTERVAL="10" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_etth1_teacher_ft_v1] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_teacher_ft_v1] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "[time_series_etth1_teacher_ft_v1] init_checkpoint_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_teacher_ft_v1] peer_init_checkpoint_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_teacher_ft_v1] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label lr lambda sparsity hidden_dim dropout init_bias ramp_start ramp_end tail_start regime_q focus_alpha decomp_kernel anchor_weight <<< "$spec"
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
    SSML_CORRECTION_DECOMPOSITION_KERNEL="$decomp_kernel" \
    SSML_ANCHOR_WEIGHT="$anchor_weight"
done

echo "[time_series_etth1_teacher_ft_v1] done"
