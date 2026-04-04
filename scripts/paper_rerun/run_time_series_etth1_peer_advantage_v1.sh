#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_peer_advantage_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_peer_advantage_v1}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-mse}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
SSML_GUIDANCE_MODE="${SSML_GUIDANCE_MODE:-corrective}"
SSML_STUDENT_ONLY="${SSML_STUDENT_ONLY:-1}"
SSML_FREEZE_PEER="${SSML_FREEZE_PEER:-1}"
SSML_CORRECTION_THRESHOLD="${SSML_CORRECTION_THRESHOLD:-0.5}"
SSML_CORRECTION_FEATURE_MODE="${SSML_CORRECTION_FEATURE_MODE:-trend_residual}"
SSML_CORRECTION_USE_REGIME_FEATURES="${SSML_CORRECTION_USE_REGIME_FEATURES:-1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-10}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
CASE_SPECS="${CASE_SPECS:-advq80_tail60_rg70_min5e3:0.0008:0.20:0.0004:96:0.00:-2.5:2:12:16:0.60:0.70:0.80:0.005:5:13:24:48:0.30 advq85_tail60_rg70_min1e2:0.0006:0.25:0.0004:128:0.00:-2.0:2:14:18:0.60:0.70:0.85:0.010:5:13:24:52:0.25 advq75_tail50_rg80_min0:0.0010:0.15:0.0005:96:0.00:-3.0:2:10:14:0.50:0.80:0.75:0.000:3:9:20:44:0.35}"

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
    "time_series_etth1_peer_advantage_v1/$label" \
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
      SSML_CORRECTION_FEATURE_MODE="$SSML_CORRECTION_FEATURE_MODE" \
      SSML_CORRECTION_USE_REGIME_FEATURES="$SSML_CORRECTION_USE_REGIME_FEATURES" \
      EARLY_STOP_PATIENCE="$EARLY_STOP_PATIENCE" \
      EARLY_STOP_MIN_EPOCHS="$EARLY_STOP_MIN_EPOCHS" \
      EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA" \
      WARMUP_EPOCHS="$WARMUP_EPOCHS" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_etth1_peer_advantage_v1] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_peer_advantage_v1] gpu=$GPU seeds=$SEEDS"
echo "[time_series_etth1_peer_advantage_v1] model_pairs=$MODEL_PAIRS"
echo "[time_series_etth1_peer_advantage_v1] correction_feature_mode=$SSML_CORRECTION_FEATURE_MODE"
echo "[time_series_etth1_peer_advantage_v1] correction_regime_features=$SSML_CORRECTION_USE_REGIME_FEATURES"
echo "[time_series_etth1_peer_advantage_v1] init_checkpoint_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_peer_advantage_v1] peer_init_checkpoint_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_peer_advantage_v1] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label lr lambda sparsity hidden_dim dropout init_bias ramp_start ramp_end freeze_epochs tail_start regime_q peer_adv_q peer_adv_min peer_adv_k decomp_k decay_start decay_end decay_min_scale <<< "$spec"
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
    SSML_CORRECTION_FREEZE_STUDENT_EPOCHS="$freeze_epochs" \
    SSML_CORRECTION_TAIL_START_RATIO="$tail_start" \
    SSML_CORRECTION_REGIME_FOCUS_QUANTILE="$regime_q" \
    SSML_CORRECTION_PEER_ADVANTAGE_QUANTILE="$peer_adv_q" \
    SSML_CORRECTION_PEER_ADVANTAGE_MIN="$peer_adv_min" \
    SSML_CORRECTION_PEER_ADVANTAGE_SMOOTHING_KERNEL="$peer_adv_k" \
    SSML_CORRECTION_DECOMPOSITION_KERNEL="$decomp_k" \
    IMITATION_DECAY_START_EPOCH="$decay_start" \
    IMITATION_DECAY_END_EPOCH="$decay_end" \
    IMITATION_DECAY_MIN_SCALE="$decay_min_scale"
done
