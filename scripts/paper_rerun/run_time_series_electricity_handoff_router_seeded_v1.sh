#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_electricity_handoff_router_seeded_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-70}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_electricity_handoff_router_seeded_v1}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-mse}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
CASE_SPECS="${CASE_SPECS:-elec_handoff_q75_b18:0.10:0.0008:64:0.00:10:10:18:18:24:0.05:0.75:0.0005:3:0.18:12:18:0.0001:22:0.70:1.00:0.00 elec_handoff_q70_b20:0.12:0.0010:64:0.00:8:8:16:16:22:0.05:0.70:0.0003:3:0.20:12:18:0.0001:20:0.65:1.05:0.00 elec_handoff_q80_b15:0.08:0.0008:48:0.00:12:12:20:20:26:0.08:0.80:0.0008:5:0.15:12:20:0.0001:24:0.75:0.95:0.00}"

if [[ -z "${INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  INIT_CHECKPOINT_TEMPLATE='results/time_series_electricity_followup_v1/best_known/time_series/{dataset}/{model}_independent_{regression_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  PEER_INIT_CHECKPOINT_TEMPLATE='results/time_series_electricity_followup_v1/best_known/time_series/{dataset}/{model}_independent_{regression_imitation_loss}_seed{seed}/model.pt'
fi

run_case() {
  local label="$1"
  shift
  run_locked_job \
    "time_series_electricity_handoff_router_seeded_v1/$label" \
    "time_series_electricity_handoff_router_seeded_v1/$label" \
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
      SSML_GUIDANCE_MODE="corrective" \
      SSML_STUDENT_ONLY="1" \
      SSML_FREEZE_PEER="1" \
      SSML_CORRECTION_ONLY="1" \
      SSML_CORRECTION_FEATURE_MODE="trend_residual" \
      SSML_CORRECTION_USE_REGIME_FEATURES="1" \
      SSML_CORRECTION_THRESHOLD="0.5" \
      SSML_ROUTER_BIN_ENDPOINTS="${SSML_ROUTER_BIN_ENDPOINTS:-8,16,24}" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      LIVE_PLOT_INTERVAL="10" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_electricity_handoff_router_seeded_v1] output_root=$OUTPUT_ROOT"
echo "[time_series_electricity_handoff_router_seeded_v1] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS"
echo "[time_series_electricity_handoff_router_seeded_v1] init_checkpoint_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_electricity_handoff_router_seeded_v1] peer_init_checkpoint_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[time_series_electricity_handoff_router_seeded_v1] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label lambda sparsity hidden_dim dropout warmup ramp_start ramp_end decay_start decay_end decay_min_scale peer_adv_q peer_adv_min peer_adv_k budget_ratio patience min_epochs min_delta handoff_end router_decay trend_scale residual_scale <<< "$spec"
  run_case \
    "$label" \
    LAMBDA_IMITATION="$lambda" \
    SSML_CORRECTION_SPARSITY_WEIGHT="$sparsity" \
    SSML_CORRECTION_GATE_HIDDEN_DIM="$hidden_dim" \
    SSML_CORRECTION_GATE_DROPOUT="$dropout" \
    WARMUP_EPOCHS="$warmup" \
    SSML_CORRECTION_RAMP_START_EPOCH="$ramp_start" \
    SSML_CORRECTION_RAMP_END_EPOCH="$ramp_end" \
    IMITATION_DECAY_START_EPOCH="$decay_start" \
    IMITATION_DECAY_END_EPOCH="$decay_end" \
    IMITATION_DECAY_MIN_SCALE="$decay_min_scale" \
    SSML_CORRECTION_PEER_ADVANTAGE_QUANTILE="$peer_adv_q" \
    SSML_CORRECTION_PEER_ADVANTAGE_MIN="$peer_adv_min" \
    SSML_CORRECTION_PEER_ADVANTAGE_SMOOTHING_KERNEL="$peer_adv_k" \
    SSML_CORRECTION_BUDGET_RATIO="$budget_ratio" \
    EARLY_STOP_PATIENCE="$patience" \
    EARLY_STOP_MIN_EPOCHS="$min_epochs" \
    EARLY_STOP_MIN_DELTA="$min_delta" \
    SSML_HANDOFF_END_EPOCH="$handoff_end" \
    SSML_ROUTER_EMA_DECAY="$router_decay" \
    SSML_TREND_ONLY_TEACHING="1" \
    SSML_CORRECTION_TREND_SCALE="$trend_scale" \
    SSML_CORRECTION_RESIDUAL_SCALE="$residual_scale"
done

echo "[time_series_electricity_handoff_router_seeded_v1] done"
