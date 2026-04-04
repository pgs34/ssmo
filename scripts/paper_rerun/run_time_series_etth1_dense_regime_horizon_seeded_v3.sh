#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_dense_regime_horizon_seeded_v3}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_dense_regime_horizon_seeded_v3}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-mse}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-18}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-15}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
LIVE_PLOT_INTERVAL="${LIVE_PLOT_INTERVAL:-20}"
CASE_SPECS="${CASE_SPECS:-dhr_tail50_reg60_ad45_ws3_exp5_res:0.00008:0.12:0.00025:96:0.00:-2.2:0.35:2:8:10:12:36:0.25:13:0.50:0.60:0.08:0.35:0.45:0.60:3:5:0.30:1.10:residual:0.0000 dhr_tail65_reg70_ad40_ws5_exp7_res:0.00006:0.14:0.00030:96:0.00:-2.6:0.35:2:10:12:15:45:0.25:17:0.65:0.70:0.10:0.30:0.40:0.70:5:7:0.40:1.20:residual:0.0000 dhr_full_reg50_ad35_ws3_exp9_raw:0.00010:0.10:0.00020:64:0.00:-2.0:0.30:2:6:8:10:30:0.20:9:0.00:0.50:0.05:0.40:0.35:0.80:3:9:0.00:1.00:raw:0.0000 dhr_tail40_reg80_ad50_ws1_exp3_delta:0.00008:0.16:0.00040:128:0.00:-2.8:0.40:2:10:12:12:40:0.20:17:0.40:0.80:0.12:0.28:0.50:0.50:1:3:0.25:1.30:delta:0.0000}"

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
    "time_series_etth1_dense_regime_horizon_seeded_v3/$label" \
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
      SSML_TOPK_SCOPE="total" \
      SSML_MAX_SELECTED_RATIO="1.0" \
      SSML_ADAPTIVE_DENSE_TOPK_SCOPE="positive" \
      SSML_ADAPTIVE_DENSE_MAX_SELECTED_RATIO="1.0" \
      SSML_SUPERVISED_WEIGHT_MODE="score" \
      SSML_GATE_SCORE_MODE="peer_better_student_error" \
      SSML_SCORE_TRANSFORM="none" \
      SSML_POSITIVE_UPPER_QUANTILE="1.0" \
      SSML_CORRECTION_FEATURE_MODE="trend_residual" \
      SSML_CORRECTION_USE_REGIME_FEATURES="1" \
      EARLY_STOP_PATIENCE="$EARLY_STOP_PATIENCE" \
      EARLY_STOP_MIN_EPOCHS="$EARLY_STOP_MIN_EPOCHS" \
      EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA" \
      LIVE_PLOT_INTERVAL="$LIVE_PLOT_INTERVAL" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_etth1_dense_regime_horizon_seeded_v3] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_dense_regime_horizon_seeded_v3] gpu=$GPU seeds=$SEEDS"
echo "[time_series_etth1_dense_regime_horizon_seeded_v3] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label lr lambda sparsity hidden_dim dropout init_bias threshold ramp_start ramp_end freeze_epochs decay_start decay_end decay_min_scale decomp_kernel corr_tail regime_q focus_alpha topk_ratio adaptive_dense_threshold adaptive_dense_ratio score_smooth_kernel window_expand_kernel tail_start residual_beta imitation_space anchor_weight <<< "$spec"
  run_case \
    "$label" \
    LR="$lr" \
    LAMBDA_IMITATION="$lambda" \
    SSML_CORRECTION_SPARSITY_WEIGHT="$sparsity" \
    SSML_CORRECTION_GATE_HIDDEN_DIM="$hidden_dim" \
    SSML_CORRECTION_GATE_DROPOUT="$dropout" \
    SSML_CORRECTION_INIT_BIAS="$init_bias" \
    SSML_CORRECTION_THRESHOLD="$threshold" \
    SSML_CORRECTION_RAMP_START_EPOCH="$ramp_start" \
    SSML_CORRECTION_RAMP_END_EPOCH="$ramp_end" \
    SSML_CORRECTION_FREEZE_STUDENT_EPOCHS="$freeze_epochs" \
    IMITATION_DECAY_START_EPOCH="$decay_start" \
    IMITATION_DECAY_END_EPOCH="$decay_end" \
    IMITATION_DECAY_MIN_SCALE="$decay_min_scale" \
    SSML_CORRECTION_DECOMPOSITION_KERNEL="$decomp_kernel" \
    SSML_CORRECTION_TAIL_START_RATIO="$corr_tail" \
    SSML_CORRECTION_REGIME_FOCUS_QUANTILE="$regime_q" \
    SSML_CORRECTION_FOCUS_LOSS_ALPHA="$focus_alpha" \
    SSML_TOPK_RATIO="$topk_ratio" \
    SSML_ADAPTIVE_DENSE_THRESHOLD="$adaptive_dense_threshold" \
    SSML_ADAPTIVE_DENSE_TOPK_RATIO="$adaptive_dense_ratio" \
    SSML_ADAPTIVE_DENSE_SCORE_SMOOTHING_KERNEL="$score_smooth_kernel" \
    SSML_ADAPTIVE_DENSE_WINDOW_EXPAND_KERNEL="$window_expand_kernel" \
    SSML_SCORE_SMOOTHING_KERNEL="$score_smooth_kernel" \
    SSML_WINDOW_SCORE_KERNEL="$score_smooth_kernel" \
    SSML_WINDOW_EXPAND_KERNEL="$window_expand_kernel" \
    SSML_TAIL_START_RATIO="$tail_start" \
    SSML_RESIDUAL_BETA="$residual_beta" \
    SSML_IMITATION_SPACE="$imitation_space" \
    SSML_ANCHOR_WEIGHT="$anchor_weight" \
    WARMUP_EPOCHS="0"
done
