#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_long_campaign_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
INDEPENDENT_MODELS="${INDEPENDENT_MODELS:-transformer}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-huber}"
LIVE_PLOT_INTERVAL="${LIVE_PLOT_INTERVAL:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_long_campaign_v1}"

run_core_stage() {
  local label="$1"
  shift
  run_logged_job \
    "etth1_long_campaign_v1/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      DATASETS="etth1" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      INDEPENDENT_MODELS="$INDEPENDENT_MODELS" \
      REQUIRE_DISTINCT_PEER="1" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      SEQ_LEN="$SEQ_LEN" \
      PRED_LENS="$PRED_LENS" \
      FEATURE_MODE="$FEATURE_MODE" \
      REGRESSION_IMITATION_LOSS="$REGRESSION_IMITATION_LOSS" \
      LIVE_PLOT_INTERVAL="$LIVE_PLOT_INTERVAL" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

run_script_stage() {
  local label="$1"
  local script_path="$2"
  shift 2
  run_logged_job \
    "etth1_long_campaign_v1/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      GPU="$GPU" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      SEQ_LEN="$SEQ_LEN" \
      PRED_LENS="$PRED_LENS" \
      FEATURE_MODE="$FEATURE_MODE" \
      REGRESSION_IMITATION_LOSS="$REGRESSION_IMITATION_LOSS" \
      LIVE_PLOT_INTERVAL="$LIVE_PLOT_INTERVAL" \
      "$@" \
      bash "$script_path"
}

echo "[time_series_etth1_long_campaign_v1] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_long_campaign_v1] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS"
echo "[time_series_etth1_long_campaign_v1] model_pairs=$MODEL_PAIRS"
echo "[time_series_etth1_long_campaign_v1] plan=all_methods_long -> fastdecay_long -> delayed_guidance_long"

# Stage 1: rerun all ETTh1 methods with long epochs and no early stop.
run_core_stage \
  "all_methods_long120" \
  METHODS="independent dml ssml" \
  OUTPUT_DIR="$OUTPUT_ROOT/all_methods_long120" \
  LAMBDA_IMITATION="0.001" \
  MARGIN="0.02" \
  WARMUP_EPOCHS="5" \
  IMITATION_DECAY_START_EPOCH="10" \
  IMITATION_DECAY_END_EPOCH="50" \
  IMITATION_DECAY_MIN_SCALE="0.1" \
  SSML_TOPK_RATIO="0.02" \
  SSML_SUPERVISED_HOTSPOT_ALPHA="0.5" \
  SSML_GATE_SCORE_MODE="peer_better_student_error" \
  SSML_SCORE_TRANSFORM="none" \
  SSML_GUIDANCE_MODE="reweight_only" \
  HETERO_SSML_ONE_WAY="1" \
  EARLY_STOP_PATIENCE="0" \
  EARLY_STOP_MIN_EPOCHS="0" \
  EARLY_STOP_MIN_DELTA="0.0"

# Stage 2: keep the fast-decay rescue idea but let it run much longer.
run_script_stage \
  "early_best_long120" \
  "scripts/paper_rerun/run_time_series_etth1_early_best_v1.sh" \
  OUTPUT_ROOT="$OUTPUT_ROOT/early_best_long120" \
  CASE_SPECS="a0p5_long:0.5:0.001:0.02:0.02 a025_long:0.25:0.001:0.02:0.02"

# Stage 3: delayed guidance activation to test whether later hotspot reweighting helps.
run_core_stage \
  "ssml_delayed_guidance_a" \
  METHODS="ssml" \
  OUTPUT_DIR="$OUTPUT_ROOT/ssml_delayed_guidance_a" \
  LAMBDA_IMITATION="0.001" \
  MARGIN="0.02" \
  WARMUP_EPOCHS="30" \
  IMITATION_DECAY_START_EPOCH="45" \
  IMITATION_DECAY_END_EPOCH="105" \
  IMITATION_DECAY_MIN_SCALE="0.0" \
  SSML_TOPK_RATIO="0.02" \
  SSML_SUPERVISED_HOTSPOT_ALPHA="0.5" \
  SSML_GATE_SCORE_MODE="peer_better_student_error" \
  SSML_SCORE_TRANSFORM="none" \
  SSML_GUIDANCE_MODE="reweight_only" \
  HETERO_SSML_ONE_WAY="1" \
  EARLY_STOP_PATIENCE="0" \
  EARLY_STOP_MIN_EPOCHS="0" \
  EARLY_STOP_MIN_DELTA="0.0"

run_core_stage \
  "ssml_delayed_guidance_sparse" \
  METHODS="ssml" \
  OUTPUT_DIR="$OUTPUT_ROOT/ssml_delayed_guidance_sparse" \
  LAMBDA_IMITATION="0.002" \
  MARGIN="0.05" \
  WARMUP_EPOCHS="30" \
  IMITATION_DECAY_START_EPOCH="45" \
  IMITATION_DECAY_END_EPOCH="105" \
  IMITATION_DECAY_MIN_SCALE="0.0" \
  SSML_TOPK_RATIO="0.01" \
  SSML_SUPERVISED_HOTSPOT_ALPHA="0.5" \
  SSML_GATE_SCORE_MODE="peer_better_student_error" \
  SSML_SCORE_TRANSFORM="none" \
  SSML_GUIDANCE_MODE="reweight_only" \
  HETERO_SSML_ONE_WAY="1" \
  EARLY_STOP_PATIENCE="0" \
  EARLY_STOP_MIN_EPOCHS="0" \
  EARLY_STOP_MIN_DELTA="0.0"
