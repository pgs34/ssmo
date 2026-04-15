#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_scaled_fair_cifarstem_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1536}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD="${DOWNLOAD:-0}"
RUN_GROUP="${RUN_GROUP:-all}"
MODEL_PAIRS="${MODEL_PAIRS:-resnet34_cifar_gelu:resnet34_cifar_gelu}"
INDEPENDENT_MODELS="${INDEPENDENT_MODELS:-resnet34_cifar_gelu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_cifar100_scaled_fair_cifarstem_v1}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
PROTOCOL_ID="${PROTOCOL_ID:-scaled_fair_cifarstem_bs${BATCH_SIZE}_v1}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-unknown}"
INDEPENDENT_LABEL="${INDEPENDENT_LABEL:-scaled_cifarstem_independent_v1}"
DML_LABEL="${DML_LABEL:-scaled_cifarstem_dml_v1}"
DEFAULT_POOL_TEMPLATE="results/classification_cifar100_bestckpt_pool_v2/classification/classification/cifar100/{model}_independent_${CLASSIFICATION_IMITATION_LOSS}_seed{seed}/best_model.pt"
BEST_CKPT_TEMPLATE="${BEST_CKPT_TEMPLATE:-$DEFAULT_POOL_TEMPLATE}"
DML_LAMBDA="${DML_LAMBDA:-0.04}"
DML_TEMPERATURE="${DML_TEMPERATURE:-6.0}"
DML_MARGIN="${DML_MARGIN:-0.0}"
SSML_CASE_SPECS="${SSML_CASE_SPECS:-oxtra42_cifarstem_v1:0.42:0.020:0.018:0.000:0.42:0.01:18:6.0:0.0002:0.93:1.25:3:0.50:0.04:12:18:36:0.25}"

echo "[classification_cifar100_scaled_fair_cifarstem_v1] gpu=$GPU seeds=$SEEDS run_group=$RUN_GROUP batch_size=$BATCH_SIZE"
echo "[classification_cifar100_scaled_fair_cifarstem_v1] output_root=$OUTPUT_ROOT"
echo "[classification_cifar100_scaled_fair_cifarstem_v1] best_ckpt_template=$BEST_CKPT_TEMPLATE"

if [[ "$RUN_GROUP" == "all" || "$RUN_GROUP" == "independent" ]]; then
  run_logged_job \
    "classification_cifar100_scaled_fair_cifarstem_v1/${INDEPENDENT_LABEL}" \
    "$LOG_DIR/${INDEPENDENT_LABEL}.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      METHODS="independent" \
      DATASETS="cifar100" \
      INDEPENDENT_MODELS="$INDEPENDENT_MODELS" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="0" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      DOWNLOAD="$DOWNLOAD" \
      OUTPUT_DIR="$OUTPUT_ROOT/${INDEPENDENT_LABEL}" \
      PROTOCOL_ID="$PROTOCOL_ID" \
      HARDWARE_PROFILE="$HARDWARE_PROFILE" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      OPTIMIZER="sgd_nesterov" \
      MOMENTUM="0.9" \
      LR_SCHEDULER="cosine" \
      SCHEDULER_WARMUP_EPOCHS="5" \
      SCHEDULER_MIN_SCALE="0.10" \
      LABEL_SMOOTHING="0.1" \
      MODEL_EMA_DECAY="0.999" \
      GRAD_CLIP="1.0" \
      TRAIN_AUG_MODE="strong" \
      FREEZE_BN_STATS="1" \
      LR="${LR:-0.05}" \
      WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}" \
      INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      bash scripts/paper_rerun/run_core_classification.sh
fi

if [[ "$RUN_GROUP" == "all" || "$RUN_GROUP" == "dml" ]]; then
  run_logged_job \
    "classification_cifar100_scaled_fair_cifarstem_v1/${DML_LABEL}" \
    "$LOG_DIR/${DML_LABEL}.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      METHODS="dml" \
      DATASETS="cifar100" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="0" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      DOWNLOAD="$DOWNLOAD" \
      OUTPUT_DIR="$OUTPUT_ROOT/${DML_LABEL}" \
      PROTOCOL_ID="$PROTOCOL_ID" \
      HARDWARE_PROFILE="$HARDWARE_PROFILE" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      OPTIMIZER="sgd_nesterov" \
      MOMENTUM="0.9" \
      LR_SCHEDULER="cosine" \
      SCHEDULER_WARMUP_EPOCHS="5" \
      SCHEDULER_MIN_SCALE="0.10" \
      LABEL_SMOOTHING="0.1" \
      MODEL_EMA_DECAY="0.999" \
      GRAD_CLIP="1.0" \
      TRAIN_AUG_MODE="strong" \
      FREEZE_BN_STATS="1" \
      LR="${LR:-0.05}" \
      WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}" \
      INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      DISTILL_TEMPERATURE="$DML_TEMPERATURE" \
      LAMBDA_IMITATION="$DML_LAMBDA" \
      MARGIN="$DML_MARGIN" \
      WARMUP_EPOCHS="0" \
      IMITATION_DECAY_START_EPOCH="-1" \
      IMITATION_DECAY_END_EPOCH="-1" \
      IMITATION_DECAY_MIN_SCALE="1.0" \
      bash scripts/paper_rerun/run_core_classification.sh
fi

if [[ "$RUN_GROUP" == "all" || "$RUN_GROUP" == "ssml" ]]; then
  for spec in $SSML_CASE_SPECS; do
    IFS=':' read -r label topk_ratio alpha lambda margin prob_threshold prob_gap per_class_budget distill_temperature anchor_weight student_true_prob_max aug_weight aug_shift aug_flip aug_noise warmup decay_start decay_end decay_min_scale <<< "$spec"
    run_logged_job \
      "classification_cifar100_scaled_fair_cifarstem_v1/$label" \
      "$LOG_DIR/$label.log" \
      env \
        CUDA_VISIBLE_DEVICES="$GPU" \
        DEVICE="$DEVICE" \
        METHODS="ssml" \
        DATASETS="cifar100" \
        MODEL_PAIRS="$MODEL_PAIRS" \
        REQUIRE_DISTINCT_PEER="0" \
        SEEDS="$SEEDS" \
        EPOCHS="$EPOCHS" \
        BATCH_SIZE="$BATCH_SIZE" \
        NUM_WORKERS="$NUM_WORKERS" \
        DOWNLOAD="$DOWNLOAD" \
        OUTPUT_DIR="$OUTPUT_ROOT/$label" \
        PROTOCOL_ID="$PROTOCOL_ID" \
        HARDWARE_PROFILE="$HARDWARE_PROFILE" \
        CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
        OPTIMIZER="sgd_nesterov" \
        MOMENTUM="0.9" \
        LR_SCHEDULER="cosine" \
        SCHEDULER_WARMUP_EPOCHS="5" \
        SCHEDULER_MIN_SCALE="0.10" \
        LABEL_SMOOTHING="0.1" \
        MODEL_EMA_DECAY="0.999" \
        GRAD_CLIP="1.0" \
        TRAIN_AUG_MODE="strong" \
        FREEZE_BN_STATS="1" \
        LR="${LR:-0.05}" \
        WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}" \
        INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
        PEER_INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
        SSML_TOPK_SCOPE="positive" \
        SSML_SUPERVISED_WEIGHT_MODE="score" \
        SSML_GATE_SCORE_MODE="peer_confident_student_uncertain" \
        SSML_SCORE_TRANSFORM="none" \
        SSML_GUIDANCE_MODE="hybrid" \
        SSML_PEER_CORRECT_ONLY="1" \
        SSML_STUDENT_INCORRECT_ONLY="0" \
        SSML_DISAGREEMENT_ONLY="0" \
        SSML_CLASS_BALANCED_TOPK="1" \
        SSML_STUDENT_ONLY="1" \
        SSML_FREEZE_PEER="1" \
        SSML_WORSE_ONLY_UPDATE="0" \
        WARMUP_EPOCHS="$warmup" \
        IMITATION_DECAY_START_EPOCH="$decay_start" \
        IMITATION_DECAY_END_EPOCH="$decay_end" \
        IMITATION_DECAY_MIN_SCALE="$decay_min_scale" \
        SSML_TOPK_RATIO="$topk_ratio" \
        SSML_SUPERVISED_HOTSPOT_ALPHA="$alpha" \
        LAMBDA_IMITATION="$lambda" \
        MARGIN="$margin" \
        SSML_PEER_TRUE_PROB_THRESHOLD="$prob_threshold" \
        SSML_PEER_STUDENT_PROB_GAP_MIN="$prob_gap" \
        SSML_PER_CLASS_BUDGET="$per_class_budget" \
        DISTILL_TEMPERATURE="$distill_temperature" \
        SSML_ANCHOR_WEIGHT="$anchor_weight" \
        SSML_STUDENT_TRUE_PROB_MAX="$student_true_prob_max" \
        SSML_AUG_CONSISTENCY_WEIGHT="$aug_weight" \
        SSML_AUG_CONSISTENCY_SHIFT="$aug_shift" \
        SSML_AUG_CONSISTENCY_FLIP_PROB="$aug_flip" \
        SSML_AUG_CONSISTENCY_NOISE_STD="$aug_noise" \
        bash scripts/paper_rerun/run_core_classification.sh
  done
fi

echo "[classification_cifar100_scaled_fair_cifarstem_v1] done"
