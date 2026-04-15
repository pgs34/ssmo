#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_teacher_ft_snapshot_handoff_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-768}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_teacher_ft_snapshot_handoff_v1}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-huber}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
CASE_SPECS="${CASE_SPECS:-tft_h18_t10_l008_lr15e4:0.00015:0.008:64:0.00:-0.20:8:18:18:0.10:0.15:0.15:13:0.05 tft_h22_t12_l010_lr2e4:0.00020:0.010:64:0.00:-0.20:10:22:22:0.12:0.18:0.15:13:0.05 tft_h26_t14_l012_lr2e4:0.00020:0.012:64:0.00:-0.20:12:26:26:0.14:0.18:0.15:13:0.05}"

if [[ -z "${INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  INIT_CHECKPOINT_TEMPLATE='results/time_series_etth1_all_methods_long_v3/time_series/etth1/{model}_independent_huber_seed{seed}/best_model.pt'
fi
if [[ -z "${PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  PEER_INIT_CHECKPOINT_TEMPLATE='results/time_series_etth1_all_methods_long_v3/time_series/etth1/{model}_independent_huber_seed{seed}/best_model.pt'
fi

run_case() {
  local label="$1"
  shift
  run_locked_job \
    "time_series_etth1_teacher_ft_snapshot_handoff_v1/$label" \
    "time_series_etth1_teacher_ft_snapshot_handoff_v1/$label" \
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
      SSML_EVAL_OUTPUT_MODE="${SSML_EVAL_OUTPUT_MODE:-best_branch}" \
      SSML_STUDENT_ONLY="1" \
      SSML_FREEZE_PEER="1" \
      SSML_CORRECTION_ONLY="0" \
      SSML_CORRECTION_FEATURE_MODE="trend_residual" \
      SSML_CORRECTION_USE_REGIME_FEATURES="1" \
      SSML_TREND_ONLY_TEACHING="1" \
      SSML_CORRECTION_RESIDUAL_SCALE="0.0" \
      SSML_CORRECTION_TREND_SCALE="1.0" \
      SSML_CORRECTION_SPARSITY_WEIGHT="${SSML_CORRECTION_SPARSITY_WEIGHT:-0.00001}" \
      SSML_CORRECTION_THRESHOLD="${SSML_CORRECTION_THRESHOLD:-0.5}" \
      SSML_CORRECTION_PEER_ADVANTAGE_QUANTILE="${SSML_CORRECTION_PEER_ADVANTAGE_QUANTILE:-0.0}" \
      SSML_CORRECTION_PEER_ADVANTAGE_MIN="${SSML_CORRECTION_PEER_ADVANTAGE_MIN:-0.0}" \
      SSML_CORRECTION_PEER_ADVANTAGE_SMOOTHING_KERNEL="${SSML_CORRECTION_PEER_ADVANTAGE_SMOOTHING_KERNEL:-3}" \
      SSML_CORRECTION_BUDGET_RATIO="${SSML_CORRECTION_BUDGET_RATIO:-0.0}" \
      SSML_ROUTER_BIN_ENDPOINTS="${SSML_ROUTER_BIN_ENDPOINTS:-8,16,24}" \
      SSML_ROUTER_EMA_DECAY="${SSML_ROUTER_EMA_DECAY:-0.20}" \
      WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}" \
      IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-12}" \
      IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-26}" \
      IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.30}" \
      EARLY_STOP_PATIENCE="0" \
      EARLY_STOP_MIN_EPOCHS="0" \
      EARLY_STOP_MIN_DELTA="0.0" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      LIVE_PLOT_INTERVAL="10" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_etth1_teacher_ft_snapshot_handoff_v1] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_teacher_ft_snapshot_handoff_v1] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "[time_series_etth1_teacher_ft_snapshot_handoff_v1] init_checkpoint_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_teacher_ft_snapshot_handoff_v1] peer_init_checkpoint_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_teacher_ft_snapshot_handoff_v1] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label lr lambda hidden_dim dropout init_bias ramp_start ramp_end handoff_end tail_start regime_q focus_alpha decomp_kernel anchor_weight <<< "$spec"
  run_case \
    "$label" \
    LR="$lr" \
    LAMBDA_IMITATION="$lambda" \
    SSML_CORRECTION_GATE_HIDDEN_DIM="$hidden_dim" \
    SSML_CORRECTION_GATE_DROPOUT="$dropout" \
    SSML_CORRECTION_INIT_BIAS="$init_bias" \
    SSML_CORRECTION_RAMP_START_EPOCH="$ramp_start" \
    SSML_CORRECTION_RAMP_END_EPOCH="$ramp_end" \
    SSML_HANDOFF_END_EPOCH="$handoff_end" \
    SSML_CORRECTION_TAIL_START_RATIO="$tail_start" \
    SSML_CORRECTION_REGIME_FOCUS_QUANTILE="$regime_q" \
    SSML_CORRECTION_FOCUS_LOSS_ALPHA="$focus_alpha" \
    SSML_CORRECTION_DECOMPOSITION_KERNEL="$decomp_kernel" \
    SSML_ANCHOR_WEIGHT="$anchor_weight"
done

echo "[time_series_etth1_teacher_ft_snapshot_handoff_v1] done"
