#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_weather_ssml_v16}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer_wide:dlinear}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-96}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-huber}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_weather_ssml_rescue_v16}"

WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-2}"
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-18}"
IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.35}"
SSML_GUIDANCE_MODE="${SSML_GUIDANCE_MODE:-hybrid}"
SSML_GATE_SCORE_MODE="${SSML_GATE_SCORE_MODE:-peer_better_student_error_relgain}"
SSML_SCORE_TRANSFORM="${SSML_SCORE_TRANSFORM:-log1p}"
SSML_TOPK_SCOPE="${SSML_TOPK_SCOPE:-total}"
SSML_SUPERVISED_WEIGHT_MODE="${SSML_SUPERVISED_WEIGHT_MODE:-binary}"
SSML_STUDENT_ONLY="${SSML_STUDENT_ONLY:-1}"
SSML_FREEZE_PEER="${SSML_FREEZE_PEER:-1}"
SSML_WORSE_ONLY_UPDATE="${SSML_WORSE_ONLY_UPDATE:-1}"
CASE_SPECS="${CASE_SPECS:-win_k7_t30:0.15:0.005:0.00:0.30:7:7:0.20:0.0005 win_k9_t50:0.10:0.005:0.00:0.50:9:9:0.25:0.0005 win_k11_t100:0.08:0.003:0.00:1.00:11:11:0.20:0.0001}"

if [[ -z "${INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  INIT_CHECKPOINT_TEMPLATE='results/time_series_neural_ode_full_v11/baseline/time_series/{dataset}/{model}_independent_{regression_imitation_loss}_seed{seed}/model.pt'
fi
if [[ -z "${PEER_INIT_CHECKPOINT_TEMPLATE:-}" ]]; then
  PEER_INIT_CHECKPOINT_TEMPLATE='results/time_series_neural_ode_full_v11/baseline/time_series/{dataset}/{model}_independent_{regression_imitation_loss}_seed{seed}/model.pt'
fi

run_job() {
  local label="$1"
  shift
  run_logged_job \
    "weather_v16/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      METHODS="ssml" \
      DATASETS="weather" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="1" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      SEQ_LEN="$SEQ_LEN" \
      PRED_LENS="$PRED_LENS" \
      FEATURE_MODE="$FEATURE_MODE" \
      REGRESSION_IMITATION_LOSS="$REGRESSION_IMITATION_LOSS" \
      WARMUP_EPOCHS="$WARMUP_EPOCHS" \
      IMITATION_DECAY_START_EPOCH="$IMITATION_DECAY_START_EPOCH" \
      IMITATION_DECAY_END_EPOCH="$IMITATION_DECAY_END_EPOCH" \
      IMITATION_DECAY_MIN_SCALE="$IMITATION_DECAY_MIN_SCALE" \
      SSML_GUIDANCE_MODE="$SSML_GUIDANCE_MODE" \
      SSML_GATE_SCORE_MODE="$SSML_GATE_SCORE_MODE" \
      SSML_SCORE_TRANSFORM="$SSML_SCORE_TRANSFORM" \
      SSML_TOPK_SCOPE="$SSML_TOPK_SCOPE" \
      SSML_SUPERVISED_WEIGHT_MODE="$SSML_SUPERVISED_WEIGHT_MODE" \
      SSML_STUDENT_ONLY="$SSML_STUDENT_ONLY" \
      SSML_FREEZE_PEER="$SSML_FREEZE_PEER" \
      SSML_WORSE_ONLY_UPDATE="$SSML_WORSE_ONLY_UPDATE" \
      INIT_CHECKPOINT_TEMPLATE="$INIT_CHECKPOINT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$PEER_INIT_CHECKPOINT_TEMPLATE" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_weather_ssml_v16] output_root=$OUTPUT_ROOT"
echo "[time_series_weather_ssml_v16] gpu=$GPU seeds=$SEEDS"
echo "[time_series_weather_ssml_v16] model_pairs=$MODEL_PAIRS"
echo "[time_series_weather_ssml_v16] warmup=$WARMUP_EPOCHS decay=${IMITATION_DECAY_START_EPOCH}->${IMITATION_DECAY_END_EPOCH} min_scale=$IMITATION_DECAY_MIN_SCALE"
echo "[time_series_weather_ssml_v16] freeze_peer=$SSML_FREEZE_PEER student_only=$SSML_STUDENT_ONLY worse_only=$SSML_WORSE_ONLY_UPDATE"
echo "[time_series_weather_ssml_v16] gate=$SSML_GATE_SCORE_MODE transform=$SSML_SCORE_TRANSFORM topk_scope=$SSML_TOPK_SCOPE weight_mode=$SSML_SUPERVISED_WEIGHT_MODE guidance=$SSML_GUIDANCE_MODE"
echo "[time_series_weather_ssml_v16] case_specs=$CASE_SPECS"

for spec in $CASE_SPECS; do
  IFS=':' read -r label alpha lambda_imitation margin topk_ratio smooth_k expand_k residual_beta anchor_weight <<< "$spec"
  run_job \
    "$label" \
    LAMBDA_IMITATION="$lambda_imitation" \
    MARGIN="$margin" \
    SSML_TOPK_RATIO="$topk_ratio" \
    SSML_SUPERVISED_HOTSPOT_ALPHA="$alpha" \
    SSML_POSITIVE_UPPER_QUANTILE="1.0" \
    SSML_SCORE_SMOOTHING_KERNEL="$smooth_k" \
    SSML_WINDOW_EXPAND_KERNEL="$expand_k" \
    SSML_RESIDUAL_BETA="$residual_beta" \
    SSML_EMA_DECAY="0.0" \
    SSML_ANCHOR_WEIGHT="$anchor_weight" \
    OUTPUT_DIR="$OUTPUT_ROOT/$label"
done
