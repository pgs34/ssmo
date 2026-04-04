#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_complementary_v4}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_complementary_v4}"
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
CASE_SPECS="${CASE_SPECS:-trcomp_gate8_biasm3_k9_lr1e4_l15_sp5e4_h64:0.0001:0.15:0.0005:64:0.00:-3.0:2:8:8:20:70:0.25:9 trcomp_gate8_biasm3_k13_lr1e4_l15_sp5e4_h64:0.0001:0.15:0.0005:64:0.00:-3.0:2:8:8:20:70:0.25:13 trcomp_gate10_biasm3_k9_lr1e4_l15_sp5e4_h64:0.0001:0.15:0.0005:64:0.00:-3.0:2:10:10:20:70:0.25:9}"

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
    "time_series_etth1_complementary_v4/$label" \
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

echo "[time_series_etth1_complementary_v4] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_complementary_v4] gpu=$GPU seeds=$SEEDS"
echo "[time_series_etth1_complementary_v4] model_pairs=$MODEL_PAIRS"
echo "[time_series_etth1_complementary_v4] correction_feature_mode=$SSML_CORRECTION_FEATURE_MODE"
echo "[time_series_etth1_complementary_v4] correction_regime_features=$SSML_CORRECTION_USE_REGIME_FEATURES"
echo "[time_series_etth1_complementary_v4] init_checkpoint_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_complementary_v4] peer_init_checkpoint_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_etth1_complementary_v4] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label lr lambda sparsity hidden_dim dropout init_bias ramp_start ramp_end freeze_epochs decay_start decay_end decay_min_scale decomp_kernel <<< "$spec"
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
    SSML_CORRECTION_DECOMPOSITION_KERNEL="$decomp_kernel" \
    IMITATION_DECAY_START_EPOCH="$decay_start" \
    IMITATION_DECAY_END_EPOCH="$decay_end" \
    IMITATION_DECAY_MIN_SCALE="$decay_min_scale"
done
