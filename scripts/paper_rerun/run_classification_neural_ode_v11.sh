#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_neural_ode_v11}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-cifar10 cifar100}"
SEEDS="${SEEDS:-0 1 2}"
MODEL_PAIRS="${MODEL_PAIRS:-ode_cnn:ode_cnn ode_cnn:resnet34_gelu resnet34_gelu:ode_cnn}"
REQUIRE_DISTINCT_PEER="${REQUIRE_DISTINCT_PEER:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-96}"
EPOCHS="${EPOCHS:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_neural_ode_v11}"
DOWNLOAD="${DOWNLOAD:-1}"

BASELINE_LAMBDA_IMITATION="${BASELINE_LAMBDA_IMITATION:-1.0}"
BASELINE_MARGIN="${BASELINE_MARGIN:-0.0}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-4.0}"
SSML_LAMBDA_IMITATION="${SSML_LAMBDA_IMITATION:-0.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-35}"
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-85}"
IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.2}"
HETERO_SSML_ONE_WAY="${HETERO_SSML_ONE_WAY:-0}"
SSML_TOPK_SCOPE="${SSML_TOPK_SCOPE:-positive}"
SSML_SUPERVISED_HOTSPOT_ALPHA="${SSML_SUPERVISED_HOTSPOT_ALPHA:-0.2}"
SSML_SUPERVISED_WEIGHT_MODE="${SSML_SUPERVISED_WEIGHT_MODE:-binary}"
SSML_GATE_SCORE_MODE="${SSML_GATE_SCORE_MODE:-peer_better_true_prob_gap_weighted}"
SSML_SCORE_TRANSFORM="${SSML_SCORE_TRANSFORM:-none}"
SSML_GUIDANCE_MODE="${SSML_GUIDANCE_MODE:-reweight_only}"
SSML_PEER_CORRECT_ONLY="${SSML_PEER_CORRECT_ONLY:-1}"
SSML_STUDENT_INCORRECT_ONLY="${SSML_STUDENT_INCORRECT_ONLY:-1}"
SSML_PEER_TRUE_PROB_THRESHOLD="${SSML_PEER_TRUE_PROB_THRESHOLD:-0.4}"
CIFAR10_TOPKS="${CIFAR10_TOPKS:-0.15 0.2}"
CIFAR100_TOPKS="${CIFAR100_TOPKS:-0.2 0.3}"
SSML_MARGIN="${SSML_MARGIN:-0.0}"

slug_float() {
  local value="${1//./p}"
  value="${value//-/m}"
  printf '%s\n' "$value"
}

run_core() {
  local label="$1"
  shift
  run_logged_job \
    "classification_neural_ode_v11/$label" \
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
      DOWNLOAD="$DOWNLOAD" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE" \
      WARMUP_EPOCHS="$WARMUP_EPOCHS" \
      IMITATION_DECAY_START_EPOCH="$IMITATION_DECAY_START_EPOCH" \
      IMITATION_DECAY_END_EPOCH="$IMITATION_DECAY_END_EPOCH" \
      IMITATION_DECAY_MIN_SCALE="$IMITATION_DECAY_MIN_SCALE" \
      HETERO_SSML_ONE_WAY="$HETERO_SSML_ONE_WAY" \
      SSML_TOPK_SCOPE="$SSML_TOPK_SCOPE" \
      SSML_SUPERVISED_HOTSPOT_ALPHA="$SSML_SUPERVISED_HOTSPOT_ALPHA" \
      SSML_SUPERVISED_WEIGHT_MODE="$SSML_SUPERVISED_WEIGHT_MODE" \
      SSML_GATE_SCORE_MODE="$SSML_GATE_SCORE_MODE" \
      SSML_SCORE_TRANSFORM="$SSML_SCORE_TRANSFORM" \
      SSML_GUIDANCE_MODE="$SSML_GUIDANCE_MODE" \
      SSML_PEER_CORRECT_ONLY="$SSML_PEER_CORRECT_ONLY" \
      SSML_STUDENT_INCORRECT_ONLY="$SSML_STUDENT_INCORRECT_ONLY" \
      SSML_PEER_TRUE_PROB_THRESHOLD="$SSML_PEER_TRUE_PROB_THRESHOLD" \
      "$@" \
      bash scripts/paper_rerun/run_core_classification.sh
}

echo "[classification_neural_ode_v11] output_root=$OUTPUT_ROOT"
echo "[classification_neural_ode_v11] gpu=$GPU datasets=$DATASETS seeds=$SEEDS"
echo "[classification_neural_ode_v11] model_pairs=$MODEL_PAIRS"
echo "[classification_neural_ode_v11] baseline_lambda=$BASELINE_LAMBDA_IMITATION ssml_lambda=$SSML_LAMBDA_IMITATION"
echo "[classification_neural_ode_v11] cifar10_topks=$CIFAR10_TOPKS"
echo "[classification_neural_ode_v11] cifar100_topks=$CIFAR100_TOPKS"
echo "[classification_neural_ode_v11] warmup=$WARMUP_EPOCHS decay=${IMITATION_DECAY_START_EPOCH}->${IMITATION_DECAY_END_EPOCH} min_scale=$IMITATION_DECAY_MIN_SCALE"
echo "[classification_neural_ode_v11] topk_scope=$SSML_TOPK_SCOPE hotspot_alpha=$SSML_SUPERVISED_HOTSPOT_ALPHA weight_mode=$SSML_SUPERVISED_WEIGHT_MODE"
echo "[classification_neural_ode_v11] gate=$SSML_GATE_SCORE_MODE transform=$SSML_SCORE_TRANSFORM guidance=$SSML_GUIDANCE_MODE"
echo "[classification_neural_ode_v11] peer_correct_only=$SSML_PEER_CORRECT_ONLY student_incorrect_only=$SSML_STUDENT_INCORRECT_ONLY peer_true_prob_threshold=$SSML_PEER_TRUE_PROB_THRESHOLD"

run_core \
  "baseline" \
  METHODS="independent dml" \
  LAMBDA_IMITATION="$BASELINE_LAMBDA_IMITATION" \
  MARGIN="$BASELINE_MARGIN" \
  OUTPUT_DIR="$OUTPUT_ROOT/baseline"

for topk in $CIFAR10_TOPKS; do
  topk_slug="$(slug_float "$topk")"
  run_core \
    "cifar10_ssml_t${topk_slug}" \
    METHODS="ssml" \
    DATASETS="cifar10" \
    LAMBDA_IMITATION="$SSML_LAMBDA_IMITATION" \
    MARGIN="$SSML_MARGIN" \
    SSML_TOPK_RATIO="$topk" \
    OUTPUT_DIR="$OUTPUT_ROOT/cifar10/ssml_t${topk_slug}"
done

for topk in $CIFAR100_TOPKS; do
  topk_slug="$(slug_float "$topk")"
  run_core \
    "cifar100_ssml_t${topk_slug}" \
    METHODS="ssml" \
    DATASETS="cifar100" \
    LAMBDA_IMITATION="$SSML_LAMBDA_IMITATION" \
    MARGIN="${SSML_MARGIN_CIFAR100:-$SSML_MARGIN}" \
    SSML_TOPK_RATIO="$topk" \
    OUTPUT_DIR="$OUTPUT_ROOT/cifar100/ssml_t${topk_slug}"
done
