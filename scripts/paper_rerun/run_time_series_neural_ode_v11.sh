#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_neural_ode_v11}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-weather}"
SEEDS="${SEEDS:-0 1 2}"
MODEL_PAIRS="${MODEL_PAIRS:-neural_ode:dlinear neural_ode:transformer_wide}"
REQUIRE_DISTINCT_PEER="${REQUIRE_DISTINCT_PEER:-1}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-48}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-huber}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_neural_ode_v11}"

BASELINE_LAMBDA_IMITATION="${BASELINE_LAMBDA_IMITATION:-1.0}"
BASELINE_MARGIN="${BASELINE_MARGIN:-0.0}"
SSML_LAMBDA_IMITATION="${SSML_LAMBDA_IMITATION:-0.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-15}"
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-35}"
IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.0}"
HETERO_SSML_ONE_WAY="${HETERO_SSML_ONE_WAY:-1}"
SSML_GUIDANCE_MODE="${SSML_GUIDANCE_MODE:-reweight_only}"
SSML_GATE_SCORE_MODE="${SSML_GATE_SCORE_MODE:-peer_better_student_error_relgain}"
SSML_SCORE_TRANSFORM="${SSML_SCORE_TRANSFORM:-log1p}"
SSML_TOPK_SCOPE="${SSML_TOPK_SCOPE:-positive}"
SSML_SUPERVISED_WEIGHT_MODE="${SSML_SUPERVISED_WEIGHT_MODE:-binary}"
CASE_SPECS="${CASE_SPECS:-node_a01_t03:0.10:0.03:0.90 node_a02_t03:0.20:0.03:0.90}"

run_core() {
  local label="$1"
  shift
  run_logged_job \
    "time_series_neural_ode_v11/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      DATASETS="$DATASETS" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="$REQUIRE_DISTINCT_PEER" \
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
      HETERO_SSML_ONE_WAY="$HETERO_SSML_ONE_WAY" \
      SSML_GUIDANCE_MODE="$SSML_GUIDANCE_MODE" \
      SSML_GATE_SCORE_MODE="$SSML_GATE_SCORE_MODE" \
      SSML_SCORE_TRANSFORM="$SSML_SCORE_TRANSFORM" \
      SSML_TOPK_SCOPE="$SSML_TOPK_SCOPE" \
      SSML_SUPERVISED_WEIGHT_MODE="$SSML_SUPERVISED_WEIGHT_MODE" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

echo "[time_series_neural_ode_v11] output_root=$OUTPUT_ROOT"
echo "[time_series_neural_ode_v11] gpu=$GPU datasets=$DATASETS seeds=$SEEDS"
echo "[time_series_neural_ode_v11] model_pairs=$MODEL_PAIRS"
echo "[time_series_neural_ode_v11] baseline_lambda=$BASELINE_LAMBDA_IMITATION ssml_lambda=$SSML_LAMBDA_IMITATION"
echo "[time_series_neural_ode_v11] warmup=$WARMUP_EPOCHS decay=${IMITATION_DECAY_START_EPOCH}->${IMITATION_DECAY_END_EPOCH} min_scale=$IMITATION_DECAY_MIN_SCALE"
echo "[time_series_neural_ode_v11] one_way=$HETERO_SSML_ONE_WAY guidance=$SSML_GUIDANCE_MODE"
echo "[time_series_neural_ode_v11] gate=$SSML_GATE_SCORE_MODE transform=$SSML_SCORE_TRANSFORM topk_scope=$SSML_TOPK_SCOPE weight_mode=$SSML_SUPERVISED_WEIGHT_MODE"
echo "[time_series_neural_ode_v11] case_specs=$CASE_SPECS"

run_core \
  "baseline" \
  METHODS="independent dml" \
  LAMBDA_IMITATION="$BASELINE_LAMBDA_IMITATION" \
  MARGIN="$BASELINE_MARGIN" \
  OUTPUT_DIR="$OUTPUT_ROOT/baseline"

for spec in $CASE_SPECS; do
  IFS=':' read -r label alpha topk_ratio upper_q <<< "$spec"
  run_core \
    "$label" \
    METHODS="ssml" \
    LAMBDA_IMITATION="$SSML_LAMBDA_IMITATION" \
    MARGIN="${SSML_MARGIN:-0.0}" \
    SSML_TOPK_RATIO="$topk_ratio" \
    SSML_SUPERVISED_HOTSPOT_ALPHA="$alpha" \
    SSML_POSITIVE_UPPER_QUANTILE="$upper_q" \
    OUTPUT_DIR="$OUTPUT_ROOT/$label"
done
