#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
PAPER_RERUN_TAG="${PAPER_RERUN_TAG:-story_screen_v2}"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$(paper_results_root)/logs/worker1_story}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
TIME_SERIES_GPU0="${TIME_SERIES_GPU0:-0}"
TIME_SERIES_GPU1="${TIME_SERIES_GPU1:-1}"
SEEDS="${SEEDS:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(paper_results_root)/story_screen/time_series}"

TIME_SERIES_DATASETS_HOMO="${TIME_SERIES_DATASETS_HOMO:-electricity etth1}"
TIME_SERIES_DATASETS_HETERO="${TIME_SERIES_DATASETS_HETERO:-electricity etth1 weather}"
TIME_SERIES_MODEL_PAIRS_HOMO="${TIME_SERIES_MODEL_PAIRS_HOMO:-transformer:transformer dlinear:dlinear}"
TIME_SERIES_MODEL_PAIRS_HETERO="${TIME_SERIES_MODEL_PAIRS_HETERO:-transformer:dlinear}"
TIME_SERIES_EPOCHS="${TIME_SERIES_EPOCHS:-60}"
TIME_SERIES_BATCH_SIZE="${TIME_SERIES_BATCH_SIZE:-64}"
TIME_SERIES_NUM_WORKERS="${TIME_SERIES_NUM_WORKERS:-2}"
TIME_SERIES_SEQ_LEN="${TIME_SERIES_SEQ_LEN:-96}"
TIME_SERIES_PRED_LENS="${TIME_SERIES_PRED_LENS:-24}"
TIME_SERIES_FEATURE_MODE="${TIME_SERIES_FEATURE_MODE:-multivariate}"
TIME_SERIES_REGRESSION_IMITATION_LOSS="${TIME_SERIES_REGRESSION_IMITATION_LOSS:-mse}"
TIME_SERIES_DML_LAMBDAS="${TIME_SERIES_DML_LAMBDAS:-0.05 0.1 0.3}"
TIME_SERIES_SSML_LAMBDAS="${TIME_SERIES_SSML_LAMBDAS:-0.1 0.3}"
TIME_SERIES_SSML_MARGINS="${TIME_SERIES_SSML_MARGINS:-0.0 0.02}"
TIME_SERIES_SSML_TOPK_RATIOS="${TIME_SERIES_SSML_TOPK_RATIOS:-0.1 0.2}"
TIME_SERIES_WARMUP_EPOCHS="${TIME_SERIES_WARMUP_EPOCHS:-5}"
TIME_SERIES_IMITATION_DECAY_START_EPOCH="${TIME_SERIES_IMITATION_DECAY_START_EPOCH:-15}"
TIME_SERIES_IMITATION_DECAY_END_EPOCH="${TIME_SERIES_IMITATION_DECAY_END_EPOCH:-45}"
TIME_SERIES_IMITATION_DECAY_MIN_SCALE="${TIME_SERIES_IMITATION_DECAY_MIN_SCALE:-0.2}"
TIME_SERIES_HETERO_SSML_ONE_WAY="${TIME_SERIES_HETERO_SSML_ONE_WAY:-1}"

slug_float() {
  local value="${1//./p}"
  value="${value//-/m}"
  printf '%s\n' "$value"
}

run_time_series_job() {
  local gpu="$1"
  local label="$2"
  shift 2
  run_logged_job \
    "worker1/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      DEVICE="$DEVICE" \
      SEEDS="$SEEDS" \
      EPOCHS="$TIME_SERIES_EPOCHS" \
      BATCH_SIZE="$TIME_SERIES_BATCH_SIZE" \
      NUM_WORKERS="$TIME_SERIES_NUM_WORKERS" \
      SEQ_LEN="$TIME_SERIES_SEQ_LEN" \
      PRED_LENS="$TIME_SERIES_PRED_LENS" \
      FEATURE_MODE="$TIME_SERIES_FEATURE_MODE" \
      REGRESSION_IMITATION_LOSS="$TIME_SERIES_REGRESSION_IMITATION_LOSS" \
      IMITATION_DECAY_START_EPOCH="$TIME_SERIES_IMITATION_DECAY_START_EPOCH" \
      IMITATION_DECAY_END_EPOCH="$TIME_SERIES_IMITATION_DECAY_END_EPOCH" \
      IMITATION_DECAY_MIN_SCALE="$TIME_SERIES_IMITATION_DECAY_MIN_SCALE" \
      HETERO_SSML_ONE_WAY="$TIME_SERIES_HETERO_SSML_ONE_WAY" \
      "$@" \
      bash scripts/paper_rerun/run_core_time_series.sh
}

run_gpu0_partition() {
  local ts_root="$OUTPUT_ROOT"

  run_time_series_job \
    "$TIME_SERIES_GPU0" \
    "time_series_baseline_seed${SEEDS// /_}" \
    METHODS="independent" \
    DATASETS="$TIME_SERIES_DATASETS_HOMO $TIME_SERIES_DATASETS_HETERO" \
    MODEL_PAIRS="$TIME_SERIES_MODEL_PAIRS_HOMO $TIME_SERIES_MODEL_PAIRS_HETERO" \
    REQUIRE_DISTINCT_PEER="0" \
    LAMBDA_IMITATION="0.0" \
    MARGIN="0.0" \
    WARMUP_EPOCHS="$TIME_SERIES_WARMUP_EPOCHS" \
    SSML_TOPK_RATIO="0.1" \
    OUTPUT_DIR="$ts_root/baseline"

  for lambda_imitation in $TIME_SERIES_DML_LAMBDAS; do
    local lambda_slug
    lambda_slug="$(slug_float "$lambda_imitation")"
    run_time_series_job \
      "$TIME_SERIES_GPU0" \
      "time_series_homo_dml_l${lambda_slug}" \
      METHODS="dml" \
      DATASETS="$TIME_SERIES_DATASETS_HOMO" \
      MODEL_PAIRS="$TIME_SERIES_MODEL_PAIRS_HOMO" \
      REQUIRE_DISTINCT_PEER="0" \
      LAMBDA_IMITATION="$lambda_imitation" \
      MARGIN="0.0" \
      WARMUP_EPOCHS="$TIME_SERIES_WARMUP_EPOCHS" \
      SSML_TOPK_RATIO="0.1" \
      OUTPUT_DIR="$ts_root/homogeneous/dml_l${lambda_slug}"
  done

  for lambda_imitation in $TIME_SERIES_SSML_LAMBDAS; do
    local lambda_slug
    lambda_slug="$(slug_float "$lambda_imitation")"
    for margin in $TIME_SERIES_SSML_MARGINS; do
      local margin_slug
      margin_slug="$(slug_float "$margin")"
      for topk_ratio in $TIME_SERIES_SSML_TOPK_RATIOS; do
        local topk_slug
        topk_slug="$(slug_float "$topk_ratio")"
        run_time_series_job \
          "$TIME_SERIES_GPU0" \
          "time_series_homo_ssml_l${lambda_slug}_m${margin_slug}_t${topk_slug}" \
          METHODS="ssml" \
          DATASETS="$TIME_SERIES_DATASETS_HOMO" \
          MODEL_PAIRS="$TIME_SERIES_MODEL_PAIRS_HOMO" \
          REQUIRE_DISTINCT_PEER="0" \
          LAMBDA_IMITATION="$lambda_imitation" \
          MARGIN="$margin" \
          WARMUP_EPOCHS="$TIME_SERIES_WARMUP_EPOCHS" \
          SSML_TOPK_RATIO="$topk_ratio" \
          OUTPUT_DIR="$ts_root/homogeneous/ssml_l${lambda_slug}_m${margin_slug}_t${topk_slug}"
      done
    done
  done
}

run_gpu1_partition() {
  local ts_root="$OUTPUT_ROOT"

  for lambda_imitation in $TIME_SERIES_DML_LAMBDAS; do
    local lambda_slug
    lambda_slug="$(slug_float "$lambda_imitation")"
    run_time_series_job \
      "$TIME_SERIES_GPU1" \
      "time_series_hetero_dml_l${lambda_slug}" \
      METHODS="dml" \
      DATASETS="$TIME_SERIES_DATASETS_HETERO" \
      MODEL_PAIRS="$TIME_SERIES_MODEL_PAIRS_HETERO" \
      REQUIRE_DISTINCT_PEER="1" \
      LAMBDA_IMITATION="$lambda_imitation" \
      MARGIN="0.0" \
      WARMUP_EPOCHS="$TIME_SERIES_WARMUP_EPOCHS" \
      SSML_TOPK_RATIO="0.1" \
      OUTPUT_DIR="$ts_root/heterogeneous/dml_l${lambda_slug}"
  done

  for lambda_imitation in $TIME_SERIES_SSML_LAMBDAS; do
    local lambda_slug
    lambda_slug="$(slug_float "$lambda_imitation")"
    for margin in $TIME_SERIES_SSML_MARGINS; do
      local margin_slug
      margin_slug="$(slug_float "$margin")"
      for topk_ratio in $TIME_SERIES_SSML_TOPK_RATIOS; do
        local topk_slug
        topk_slug="$(slug_float "$topk_ratio")"
        run_time_series_job \
          "$TIME_SERIES_GPU1" \
          "time_series_hetero_ssml_l${lambda_slug}_m${margin_slug}_t${topk_slug}" \
          METHODS="ssml" \
          DATASETS="$TIME_SERIES_DATASETS_HETERO" \
          MODEL_PAIRS="$TIME_SERIES_MODEL_PAIRS_HETERO" \
          REQUIRE_DISTINCT_PEER="1" \
          LAMBDA_IMITATION="$lambda_imitation" \
          MARGIN="$margin" \
          WARMUP_EPOCHS="$TIME_SERIES_WARMUP_EPOCHS" \
          SSML_TOPK_RATIO="$topk_ratio" \
          OUTPUT_DIR="$ts_root/heterogeneous/ssml_l${lambda_slug}_m${margin_slug}_t${topk_slug}"
      done
    done
  done
}

echo "[worker1_story] results_root=$(paper_results_root)"
echo "[worker1_story] output_root=$OUTPUT_ROOT"
echo "[worker1_story] gpu0=$TIME_SERIES_GPU0 gpu1=$TIME_SERIES_GPU1 seeds=$SEEDS"

if [[ -n "$TIME_SERIES_GPU1" && "$TIME_SERIES_GPU1" != "$TIME_SERIES_GPU0" ]]; then
  run_gpu0_partition &
  PID_GPU0=$!
  run_gpu1_partition &
  PID_GPU1=$!

  echo "[worker1_story] started gpu${TIME_SERIES_GPU0} pid=$PID_GPU0"
  echo "[worker1_story] started gpu${TIME_SERIES_GPU1} pid=$PID_GPU1"

  wait "$PID_GPU0"
  wait "$PID_GPU1"
else
  run_gpu0_partition
  run_gpu1_partition
fi

echo "[worker1_story] all jobs finished"
