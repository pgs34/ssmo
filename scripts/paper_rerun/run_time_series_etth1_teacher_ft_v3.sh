#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_teacher_ft_v3}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-768}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_teacher_ft_v3}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-huber}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-0}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
SSML_CORRECTION_THRESHOLD="${SSML_CORRECTION_THRESHOLD:-0.05}"
ALLOWED_CASES_FILE="${ALLOWED_CASES_FILE:-scripts/paper_rerun/configs/etth1_teacher_ft_v3_allowed_cases.txt}"
CASE_SPECS="${CASE_SPECS:-b10_q80_m20_t14_r00_lr2e4:0.0002:0.009:0.00002:64:0.00:-0.40:10:22:0.10:0.10:0.12:13:0.08:1.4:0.0:0.80:0.0020:3:0.10:4:18:32:0.00}"

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
    "time_series_etth1_teacher_ft_v3/$label" \
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
      SSML_CORRECTION_THRESHOLD="$SSML_CORRECTION_THRESHOLD" \
      EARLY_STOP_PATIENCE="$EARLY_STOP_PATIENCE" \
      EARLY_STOP_MIN_EPOCHS="$EARLY_STOP_MIN_EPOCHS" \
      EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      LIVE_PLOT_INTERVAL="10" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_etth1_teacher_ft_v3] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_teacher_ft_v3] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "[time_series_etth1_teacher_ft_v3] init_checkpoint_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_teacher_ft_v3] peer_init_checkpoint_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_teacher_ft_v3] correction_threshold=$SSML_CORRECTION_THRESHOLD"
echo "[time_series_etth1_teacher_ft_v3] allowed_cases_file=$ALLOWED_CASES_FILE"
echo "[time_series_etth1_teacher_ft_v3] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label lr lambda sparsity hidden_dim dropout init_bias ramp_start ramp_end tail_start regime_q focus_alpha decomp_kernel anchor_weight trend_scale residual_scale peer_adv_q peer_adv_min peer_adv_k budget_ratio warmup decay_start decay_end decay_min_scale <<< "$spec"
  if [[ -f "$ALLOWED_CASES_FILE" ]] && ! grep -Fxq "$label" "$ALLOWED_CASES_FILE"; then
    echo "[time_series_etth1_teacher_ft_v3] skip disallowed_case=$label"
    continue
  fi
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
    SSML_ANCHOR_WEIGHT="$anchor_weight" \
    SSML_CORRECTION_TREND_SCALE="$trend_scale" \
    SSML_CORRECTION_RESIDUAL_SCALE="$residual_scale" \
    SSML_CORRECTION_PEER_ADVANTAGE_QUANTILE="$peer_adv_q" \
    SSML_CORRECTION_PEER_ADVANTAGE_MIN="$peer_adv_min" \
    SSML_CORRECTION_PEER_ADVANTAGE_SMOOTHING_KERNEL="$peer_adv_k" \
    SSML_CORRECTION_BUDGET_RATIO="$budget_ratio" \
    WARMUP_EPOCHS="$warmup" \
    IMITATION_DECAY_START_EPOCH="$decay_start" \
    IMITATION_DECAY_END_EPOCH="$decay_end" \
    IMITATION_DECAY_MIN_SCALE="$decay_min_scale"
done

echo "[time_series_etth1_teacher_ft_v3] done"
