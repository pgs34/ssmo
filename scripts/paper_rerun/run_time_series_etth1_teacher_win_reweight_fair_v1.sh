#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_teacher_win_reweight_fair_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-768}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_teacher_win_reweight_fair_v1}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-huber}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
CASE_SPECS="${CASE_SPECS:-twr_q85_top18_a40_h22_r75_lr2e4:0.0002:0.18:0.40:0.0000:6:22:0.75:0.030:0.85:3:3 twr_q90_top15_a45_h18_r80_lr15e4:0.00015:0.15:0.45:0.0000:8:18:0.80:0.050:0.90:5:3 twr_q80_top20_a35_h26_r70_lr25e4:0.00025:0.20:0.35:0.0000:5:26:0.70:0.020:0.80:3:5}"

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
    "time_series_etth1_teacher_win_reweight_fair_v1/$label" \
    "time_series_etth1_teacher_win_reweight_fair_v1/$label" \
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
      LR="$LR" \
      SEQ_LEN="$SEQ_LEN" \
      PRED_LENS="$PRED_LENS" \
      REGRESSION_IMITATION_LOSS="$REGRESSION_IMITATION_LOSS" \
      FEATURE_MODE="$FEATURE_MODE" \
      SSML_GUIDANCE_MODE="reweight_only" \
      SSML_EVAL_OUTPUT_MODE="${SSML_EVAL_OUTPUT_MODE:-best_branch}" \
      SSML_STUDENT_ONLY="1" \
      SSML_FREEZE_PEER="1" \
      SSML_TOPK_SCOPE="positive" \
      SSML_GATE_SCORE_MODE="peer_better_student_error" \
      SSML_SCORE_TRANSFORM="none" \
      SSML_SUPERVISED_WEIGHT_MODE="score" \
      SSML_ROUTER_BIN_ENDPOINTS="${SSML_ROUTER_BIN_ENDPOINTS:-8,16,24}" \
      SSML_WINDOW_SCORE_KERNEL="${SSML_WINDOW_SCORE_KERNEL:-3}" \
      SSML_WINDOW_EXPAND_KERNEL="${SSML_WINDOW_EXPAND_KERNEL:-3}" \
      IMITATION_DECAY_START_EPOCH="-1" \
      IMITATION_DECAY_END_EPOCH="-1" \
      IMITATION_DECAY_MIN_SCALE="0.0" \
      EARLY_STOP_PATIENCE="0" \
      EARLY_STOP_MIN_EPOCHS="0" \
      EARLY_STOP_MIN_DELTA="0.0" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      LIVE_PLOT_INTERVAL="10" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_etth1_teacher_win_reweight_fair_v1] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_teacher_win_reweight_fair_v1] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS batch_size=$BATCH_SIZE"
echo "[time_series_etth1_teacher_win_reweight_fair_v1] init_checkpoint_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_teacher_win_reweight_fair_v1] peer_init_checkpoint_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_teacher_win_reweight_fair_v1] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label lr topk_ratio alpha margin warmup handoff_end router_decay anchor_weight upper_q score_kernel expand_kernel <<< "$spec"
  run_case \
    "$label" \
    LR="$lr" \
    SSML_TOPK_RATIO="$topk_ratio" \
    SSML_SUPERVISED_HOTSPOT_ALPHA="$alpha" \
    MARGIN="$margin" \
    WARMUP_EPOCHS="$warmup" \
    SSML_HANDOFF_END_EPOCH="$handoff_end" \
    SSML_ROUTER_EMA_DECAY="$router_decay" \
    SSML_ANCHOR_WEIGHT="$anchor_weight" \
    SSML_POSITIVE_UPPER_QUANTILE="$upper_q" \
    SSML_WINDOW_SCORE_KERNEL="$score_kernel" \
    SSML_WINDOW_EXPAND_KERNEL="$expand_kernel"
done

echo "[time_series_etth1_teacher_win_reweight_fair_v1] done"
