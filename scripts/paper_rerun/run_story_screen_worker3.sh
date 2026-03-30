#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
PAPER_RERUN_TAG="${PAPER_RERUN_TAG:-story_screen_v2}"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$(paper_results_root)/logs/worker3_story}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
TIME_SERIES_GPU="${TIME_SERIES_GPU:-0}"
CLASSIFICATION_GPU="${CLASSIFICATION_GPU:-$TIME_SERIES_GPU}"
SEEDS="${SEEDS:-0}"
WORKER3_PHASES="${WORKER3_PHASES:-time_series classification}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(paper_results_root)/story_screen}"

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

CLASSIFICATION_DATASETS_HOMO="${CLASSIFICATION_DATASETS_HOMO:-cifar100}"
CLASSIFICATION_DATASETS_HETERO="${CLASSIFICATION_DATASETS_HETERO:-cifar100}"
CLASSIFICATION_MODEL_PAIRS_HOMO="${CLASSIFICATION_MODEL_PAIRS_HOMO:-resnet18:resnet18 vit_b16:vit_b16}"
CLASSIFICATION_MODEL_PAIRS_HETERO="${CLASSIFICATION_MODEL_PAIRS_HETERO:-resnet18:vit_b16}"
CLASSIFICATION_EPOCHS="${CLASSIFICATION_EPOCHS:-100}"
CLASSIFICATION_BATCH_SIZE="${CLASSIFICATION_BATCH_SIZE:-128}"
CLASSIFICATION_NUM_WORKERS="${CLASSIFICATION_NUM_WORKERS:-4}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
CLASSIFICATION_DML_LAMBDAS="${CLASSIFICATION_DML_LAMBDAS:-0.05 0.1 0.3}"
CLASSIFICATION_SSML_LAMBDAS="${CLASSIFICATION_SSML_LAMBDAS:-0.1 0.3}"
CLASSIFICATION_DISTILL_TEMPERATURES="${CLASSIFICATION_DISTILL_TEMPERATURES:-2.0 4.0}"
CLASSIFICATION_WARMUP_EPOCHS="${CLASSIFICATION_WARMUP_EPOCHS:-5}"
CLASSIFICATION_DOWNLOAD="${CLASSIFICATION_DOWNLOAD:-1}"
CLASSIFICATION_IMITATION_DECAY_START_EPOCH="${CLASSIFICATION_IMITATION_DECAY_START_EPOCH:-30}"
CLASSIFICATION_IMITATION_DECAY_END_EPOCH="${CLASSIFICATION_IMITATION_DECAY_END_EPOCH:-80}"
CLASSIFICATION_IMITATION_DECAY_MIN_SCALE="${CLASSIFICATION_IMITATION_DECAY_MIN_SCALE:-0.2}"
CLASSIFICATION_HETERO_SSML_ONE_WAY="${CLASSIFICATION_HETERO_SSML_ONE_WAY:-1}"

slug_float() {
  local value="${1//./p}"
  value="${value//-/m}"
  printf '%s\n' "$value"
}

run_time_series_job() {
  local label="$1"
  shift
  run_logged_job \
    "worker3/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$TIME_SERIES_GPU" \
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

run_classification_job() {
  local label="$1"
  shift
  run_logged_job \
    "worker3/$label" \
    "$LOG_DIR/$label.log" \
    env \
      CUDA_VISIBLE_DEVICES="$CLASSIFICATION_GPU" \
      DEVICE="$DEVICE" \
      SEEDS="$SEEDS" \
      EPOCHS="$CLASSIFICATION_EPOCHS" \
      BATCH_SIZE="$CLASSIFICATION_BATCH_SIZE" \
      NUM_WORKERS="$CLASSIFICATION_NUM_WORKERS" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      DOWNLOAD="$CLASSIFICATION_DOWNLOAD" \
      IMITATION_DECAY_START_EPOCH="$CLASSIFICATION_IMITATION_DECAY_START_EPOCH" \
      IMITATION_DECAY_END_EPOCH="$CLASSIFICATION_IMITATION_DECAY_END_EPOCH" \
      IMITATION_DECAY_MIN_SCALE="$CLASSIFICATION_IMITATION_DECAY_MIN_SCALE" \
      HETERO_SSML_ONE_WAY="$CLASSIFICATION_HETERO_SSML_ONE_WAY" \
      "$@" \
      bash scripts/paper_rerun/run_core_classification.sh
}

run_time_series_phase() {
  local ts_root="$OUTPUT_ROOT/time_series"

  echo "[worker3_story] time_series_root=$ts_root"

  run_time_series_job \
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

    run_time_series_job \
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

        run_time_series_job \
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

run_classification_phase() {
  local cls_root="$OUTPUT_ROOT/classification"

  echo "[worker3_story] classification_root=$cls_root"

  run_classification_job \
    "classification_baseline_seed${SEEDS// /_}" \
    METHODS="independent" \
    DATASETS="$CLASSIFICATION_DATASETS_HOMO $CLASSIFICATION_DATASETS_HETERO" \
    MODEL_PAIRS="$CLASSIFICATION_MODEL_PAIRS_HOMO $CLASSIFICATION_MODEL_PAIRS_HETERO" \
    REQUIRE_DISTINCT_PEER="0" \
    DISTILL_TEMPERATURE="2.0" \
    LAMBDA_IMITATION="0.0" \
    MARGIN="0.0" \
    WARMUP_EPOCHS="$CLASSIFICATION_WARMUP_EPOCHS" \
    OUTPUT_DIR="$cls_root/baseline"

  for lambda_imitation in $CLASSIFICATION_DML_LAMBDAS; do
    local lambda_slug
    lambda_slug="$(slug_float "$lambda_imitation")"
    run_classification_job \
      "classification_homo_dml_l${lambda_slug}" \
      METHODS="dml" \
      DATASETS="$CLASSIFICATION_DATASETS_HOMO" \
      MODEL_PAIRS="$CLASSIFICATION_MODEL_PAIRS_HOMO" \
      REQUIRE_DISTINCT_PEER="0" \
      DISTILL_TEMPERATURE="2.0" \
      LAMBDA_IMITATION="$lambda_imitation" \
      MARGIN="0.0" \
      WARMUP_EPOCHS="$CLASSIFICATION_WARMUP_EPOCHS" \
      OUTPUT_DIR="$cls_root/homogeneous/dml_l${lambda_slug}"

    run_classification_job \
      "classification_hetero_dml_l${lambda_slug}" \
      METHODS="dml" \
      DATASETS="$CLASSIFICATION_DATASETS_HETERO" \
      MODEL_PAIRS="$CLASSIFICATION_MODEL_PAIRS_HETERO" \
      REQUIRE_DISTINCT_PEER="1" \
      DISTILL_TEMPERATURE="2.0" \
      LAMBDA_IMITATION="$lambda_imitation" \
      MARGIN="0.0" \
      WARMUP_EPOCHS="$CLASSIFICATION_WARMUP_EPOCHS" \
      OUTPUT_DIR="$cls_root/heterogeneous/dml_l${lambda_slug}"
  done

  for lambda_imitation in $CLASSIFICATION_SSML_LAMBDAS; do
    local lambda_slug
    lambda_slug="$(slug_float "$lambda_imitation")"
    for temperature in $CLASSIFICATION_DISTILL_TEMPERATURES; do
      local temp_slug
      temp_slug="$(slug_float "$temperature")"
      run_classification_job \
        "classification_homo_ssml_l${lambda_slug}_t${temp_slug}" \
        METHODS="ssml" \
        DATASETS="$CLASSIFICATION_DATASETS_HOMO" \
        MODEL_PAIRS="$CLASSIFICATION_MODEL_PAIRS_HOMO" \
        REQUIRE_DISTINCT_PEER="0" \
        DISTILL_TEMPERATURE="$temperature" \
        LAMBDA_IMITATION="$lambda_imitation" \
        MARGIN="0.0" \
        WARMUP_EPOCHS="$CLASSIFICATION_WARMUP_EPOCHS" \
        OUTPUT_DIR="$cls_root/homogeneous/ssml_l${lambda_slug}_t${temp_slug}"

      run_classification_job \
        "classification_hetero_ssml_l${lambda_slug}_t${temp_slug}" \
        METHODS="ssml" \
        DATASETS="$CLASSIFICATION_DATASETS_HETERO" \
        MODEL_PAIRS="$CLASSIFICATION_MODEL_PAIRS_HETERO" \
        REQUIRE_DISTINCT_PEER="1" \
        DISTILL_TEMPERATURE="$temperature" \
        LAMBDA_IMITATION="$lambda_imitation" \
        MARGIN="0.0" \
        WARMUP_EPOCHS="$CLASSIFICATION_WARMUP_EPOCHS" \
        OUTPUT_DIR="$cls_root/heterogeneous/ssml_l${lambda_slug}_t${temp_slug}"
    done
  done
}

echo "[worker3_story] results_root=$(paper_results_root)"
echo "[worker3_story] output_root=$OUTPUT_ROOT"
echo "[worker3_story] phases=$WORKER3_PHASES"
echo "[worker3_story] time_series_gpu=$TIME_SERIES_GPU classification_gpu=$CLASSIFICATION_GPU"

for phase in $WORKER3_PHASES; do
  case "$phase" in
    time_series)
      run_time_series_phase
      ;;
    classification)
      run_classification_phase
      ;;
    *)
      echo "[worker3_story] unknown phase=$phase" >&2
      exit 1
      ;;
  esac
done

echo "[worker3_story] all jobs finished"
