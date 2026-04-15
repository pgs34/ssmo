#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_strict128_aggressive_v1}"
mkdir -p "$LOG_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DOWNLOAD="${DOWNLOAD:-0}"
RUN_GROUP="${RUN_GROUP:-all}"
MODEL_PAIRS="${MODEL_PAIRS:-resnet34_gelu:resnet34_gelu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_cifar100_strict128_aggressive_v1}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
PROTOCOL_ID="${PROTOCOL_ID:-strict128_aggressive_v1}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-unknown}"
STRICT_INDEPENDENT_LABEL="${STRICT_INDEPENDENT_LABEL:-strict128_independent_v2}"
STRICT_DML_LABEL="${STRICT_DML_LABEL:-strict128_dml_v2}"
DEFAULT_POOL_TEMPLATE="results/classification_cifar100_bestckpt_pool_v2/classification/classification/cifar100/{model}_independent_${CLASSIFICATION_IMITATION_LOSS}_seed{seed}/best_model.pt"
BEST_CKPT_TEMPLATE="${BEST_CKPT_TEMPLATE:-$DEFAULT_POOL_TEMPLATE}"
DML_LAMBDA="${DML_LAMBDA:-0.04}"
DML_TEMPERATURE="${DML_TEMPERATURE:-6.0}"
DML_MARGIN="${DML_MARGIN:-0.0}"
LR="${LR:-0.02}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
SCHEDULER_WARMUP_EPOCHS="${SCHEDULER_WARMUP_EPOCHS:-2}"
SCHEDULER_MIN_SCALE="${SCHEDULER_MIN_SCALE:-0.05}"
SSML_CASE_SPECS="${SSML_CASE_SPECS:-pcu_ramp_wide:peer_confident_student_uncertain:0.28:0.14:0.05:0.012:0.000:0.32:0.40:0.00:0.02:5:6.0:0.0004:0.85:0.9:2:0.5:0.02:0.00:1.0:0.00:4:12:45:0.35:0.00:0.00:0.00:-1:-1:0.00 pcu_ramp_tight:peer_confident_student_uncertain:0.24:0.10:0.05:0.012:0.000:0.35:0.42:0.01:0.03:5:6.0:0.0004:0.85:0.9:2:0.5:0.02:0.00:1.0:0.00:4:12:45:0.35:0.00:0.00:0.00:-1:-1:0.00 uh_sched_mem:useful_hard_sample_confident:0.24:0.12:0.04:0.012:0.000:0.33:0.38:0.00:0.02:4:6.0:0.0004:0.88:0.8:2:0.5:0.02:0.00:1.0:0.00:5:18:55:0.30:0.10:0.90:0.50:30:60:0.00 pcu_sched_mem:peer_confident_student_uncertain:0.24:0.12:0.05:0.012:0.000:0.33:0.38:0.00:0.02:5:6.0:0.0004:0.88:0.8:2:0.5:0.02:0.00:1.0:0.00:5:18:55:0.30:0.15:0.90:0.80:30:60:0.00 pcu_dual55:peer_confident_student_uncertain:0.28:0.14:0.05:0.012:0.000:0.32:0.40:0.00:0.02:5:6.0:0.0004:0.85:0.9:2:0.5:0.02:0.00:1.0:0.00:4:12:45:0.35:0.00:0.00:0.00:-1:-1:0.55 pcu_dual65:peer_confident_student_uncertain:0.28:0.14:0.05:0.012:0.000:0.32:0.40:0.00:0.02:5:6.0:0.0004:0.85:0.9:2:0.5:0.02:0.00:1.0:0.00:4:12:45:0.35:0.00:0.00:0.00:-1:-1:0.65}"

run_common() {
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    DEVICE="$DEVICE" \
    DATASETS="cifar100" \
    MODEL_PAIRS="$MODEL_PAIRS" \
    REQUIRE_DISTINCT_PEER="0" \
    SEEDS="$SEEDS" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    DOWNLOAD="$DOWNLOAD" \
    PROTOCOL_ID="$PROTOCOL_ID" \
    HARDWARE_PROFILE="$HARDWARE_PROFILE" \
    CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
    OPTIMIZER="sgd_nesterov" \
    MOMENTUM="0.9" \
    LR_SCHEDULER="cosine" \
    SCHEDULER_WARMUP_EPOCHS="$SCHEDULER_WARMUP_EPOCHS" \
    SCHEDULER_MIN_SCALE="$SCHEDULER_MIN_SCALE" \
    LABEL_SMOOTHING="0.1" \
    MODEL_EMA_DECAY="0.999" \
    GRAD_CLIP="1.0" \
    TRAIN_AUG_MODE="strong" \
    LR="$LR" \
    WEIGHT_DECAY="$WEIGHT_DECAY" \
    INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
    PEER_INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
    bash scripts/paper_rerun/run_core_classification.sh
}

run_independent() {
  run_logged_job \
    "classification_cifar100_strict128_aggressive_v1/${STRICT_INDEPENDENT_LABEL}" \
    "$LOG_DIR/${STRICT_INDEPENDENT_LABEL}.log" \
    env \
      METHODS="independent" \
      INDEPENDENT_MODELS="resnet34_gelu" \
      OUTPUT_DIR="$OUTPUT_ROOT/${STRICT_INDEPENDENT_LABEL}" \
      "$(declare -f run_common >/dev/null 2>&1 && printf '')" \
      run_common
}

run_dml() {
  run_logged_job \
    "classification_cifar100_strict128_aggressive_v1/${STRICT_DML_LABEL}" \
    "$LOG_DIR/${STRICT_DML_LABEL}.log" \
    env \
      METHODS="dml" \
      OUTPUT_DIR="$OUTPUT_ROOT/${STRICT_DML_LABEL}" \
      DISTILL_TEMPERATURE="$DML_TEMPERATURE" \
      LAMBDA_IMITATION="$DML_LAMBDA" \
      MARGIN="$DML_MARGIN" \
      WARMUP_EPOCHS="0" \
      IMITATION_DECAY_START_EPOCH="-1" \
      IMITATION_DECAY_END_EPOCH="-1" \
      IMITATION_DECAY_MIN_SCALE="1.0" \
      bash scripts/paper_rerun/run_core_classification.sh
}

run_ssml_case() {
  local label="$1"
  shift
  run_logged_job \
    "classification_cifar100_strict128_aggressive_v1/$label" \
    "$LOG_DIR/$label.log" \
    env \
      METHODS="ssml" \
      OUTPUT_DIR="$OUTPUT_ROOT/$label" \
      SSML_TOPK_SCOPE="positive" \
      SSML_SUPERVISED_WEIGHT_MODE="score" \
      SSML_SCORE_TRANSFORM="none" \
      SSML_GUIDANCE_MODE="hybrid" \
      SSML_PEER_CORRECT_ONLY="1" \
      SSML_STUDENT_INCORRECT_ONLY="1" \
      SSML_DISAGREEMENT_ONLY="1" \
      SSML_CLASS_BALANCED_TOPK="1" \
      SSML_STUDENT_ONLY="1" \
      SSML_FREEZE_PEER="1" \
      SSML_WORSE_ONLY_UPDATE="0" \
      "$@" \
      bash scripts/paper_rerun/run_core_classification.sh
}

echo "[classification_cifar100_strict128_aggressive_v1] gpu=$GPU seeds=$SEEDS run_group=$RUN_GROUP"
echo "[classification_cifar100_strict128_aggressive_v1] output_root=$OUTPUT_ROOT"
echo "[classification_cifar100_strict128_aggressive_v1] best_ckpt_template=$BEST_CKPT_TEMPLATE"

if [[ "$RUN_GROUP" == "all" || "$RUN_GROUP" == "independent" ]]; then
  run_logged_job \
    "classification_cifar100_strict128_aggressive_v1/${STRICT_INDEPENDENT_LABEL}" \
    "$LOG_DIR/${STRICT_INDEPENDENT_LABEL}.log" \
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      METHODS="independent" \
      DATASETS="cifar100" \
      INDEPENDENT_MODELS="resnet34_gelu" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="0" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      DOWNLOAD="$DOWNLOAD" \
      OUTPUT_DIR="$OUTPUT_ROOT/${STRICT_INDEPENDENT_LABEL}" \
      PROTOCOL_ID="$PROTOCOL_ID" \
      HARDWARE_PROFILE="$HARDWARE_PROFILE" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      OPTIMIZER="sgd_nesterov" \
      MOMENTUM="0.9" \
      LR_SCHEDULER="cosine" \
      SCHEDULER_WARMUP_EPOCHS="$SCHEDULER_WARMUP_EPOCHS" \
      SCHEDULER_MIN_SCALE="$SCHEDULER_MIN_SCALE" \
      LABEL_SMOOTHING="0.1" \
      MODEL_EMA_DECAY="0.999" \
      GRAD_CLIP="1.0" \
      TRAIN_AUG_MODE="strong" \
      LR="$LR" \
      WEIGHT_DECAY="$WEIGHT_DECAY" \
      INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      bash scripts/paper_rerun/run_core_classification.sh
fi

if [[ "$RUN_GROUP" == "all" || "$RUN_GROUP" == "dml" ]]; then
  run_logged_job \
    "classification_cifar100_strict128_aggressive_v1/${STRICT_DML_LABEL}" \
    "$LOG_DIR/${STRICT_DML_LABEL}.log" \
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
      OUTPUT_DIR="$OUTPUT_ROOT/${STRICT_DML_LABEL}" \
      PROTOCOL_ID="$PROTOCOL_ID" \
      HARDWARE_PROFILE="$HARDWARE_PROFILE" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      OPTIMIZER="sgd_nesterov" \
      MOMENTUM="0.9" \
      LR_SCHEDULER="cosine" \
      SCHEDULER_WARMUP_EPOCHS="$SCHEDULER_WARMUP_EPOCHS" \
      SCHEDULER_MIN_SCALE="$SCHEDULER_MIN_SCALE" \
      LABEL_SMOOTHING="0.1" \
      MODEL_EMA_DECAY="0.999" \
      GRAD_CLIP="1.0" \
      TRAIN_AUG_MODE="strong" \
      LR="$LR" \
      WEIGHT_DECAY="$WEIGHT_DECAY" \
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
    IFS=':' read -r label gate_mode topk_start topk_end alpha lambda margin prob_thr_start prob_thr_end prob_gap_start prob_gap_end per_class_budget distill_temperature anchor_weight student_true_prob_max aug_weight aug_shift aug_flip aug_noise peer_aug_min student_aug_max aug_gap_min warmup decay_start decay_end decay_min_scale disagreement_floor_ratio deficit_ema_momentum extra_class_budget_scale complement_ramp_start complement_ramp_end secondary_agreement_min <<< "$spec"
    run_ssml_case \
      "$label" \
      CUDA_VISIBLE_DEVICES="$GPU" \
      DEVICE="$DEVICE" \
      DATASETS="cifar100" \
      MODEL_PAIRS="$MODEL_PAIRS" \
      REQUIRE_DISTINCT_PEER="0" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      NUM_WORKERS="$NUM_WORKERS" \
      DOWNLOAD="$DOWNLOAD" \
      PROTOCOL_ID="$PROTOCOL_ID" \
      HARDWARE_PROFILE="$HARDWARE_PROFILE" \
      CLASSIFICATION_IMITATION_LOSS="$CLASSIFICATION_IMITATION_LOSS" \
      OPTIMIZER="sgd_nesterov" \
      MOMENTUM="0.9" \
      LR_SCHEDULER="cosine" \
      SCHEDULER_WARMUP_EPOCHS="$SCHEDULER_WARMUP_EPOCHS" \
      SCHEDULER_MIN_SCALE="$SCHEDULER_MIN_SCALE" \
      LABEL_SMOOTHING="0.1" \
      MODEL_EMA_DECAY="0.999" \
      GRAD_CLIP="1.0" \
      TRAIN_AUG_MODE="strong" \
      LR="$LR" \
      WEIGHT_DECAY="$WEIGHT_DECAY" \
      INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      PEER_INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      SSML_GATE_SCORE_MODE="$gate_mode" \
      SSML_TOPK_RATIO="$topk_end" \
      SSML_TOPK_RATIO_START="$topk_start" \
      SSML_TOPK_RATIO_END="$topk_end" \
      SSML_TOPK_RAMP_START_EPOCH="$warmup" \
      SSML_TOPK_RAMP_END_EPOCH="$decay_end" \
      SSML_SUPERVISED_HOTSPOT_ALPHA="$alpha" \
      LAMBDA_IMITATION="$lambda" \
      MARGIN="$margin" \
      SSML_PEER_TRUE_PROB_THRESHOLD="$prob_thr_end" \
      SSML_PEER_TRUE_PROB_THRESHOLD_START="$prob_thr_start" \
      SSML_PEER_TRUE_PROB_THRESHOLD_END="$prob_thr_end" \
      SSML_PEER_STUDENT_PROB_GAP_MIN="$prob_gap_end" \
      SSML_PEER_STUDENT_PROB_GAP_MIN_START="$prob_gap_start" \
      SSML_PEER_STUDENT_PROB_GAP_MIN_END="$prob_gap_end" \
      SSML_PER_CLASS_BUDGET="$per_class_budget" \
      DISTILL_TEMPERATURE="$distill_temperature" \
      SSML_ANCHOR_WEIGHT="$anchor_weight" \
      SSML_STUDENT_TRUE_PROB_MAX="$student_true_prob_max" \
      SSML_AUG_CONSISTENCY_WEIGHT="$aug_weight" \
      SSML_AUG_CONSISTENCY_SHIFT="$aug_shift" \
      SSML_AUG_CONSISTENCY_FLIP_PROB="$aug_flip" \
      SSML_AUG_CONSISTENCY_NOISE_STD="$aug_noise" \
      SSML_PEER_AUG_CONSISTENCY_MIN="$peer_aug_min" \
      SSML_STUDENT_AUG_CONSISTENCY_MAX="$student_aug_max" \
      SSML_PEER_STUDENT_AUG_CONSISTENCY_GAP_MIN="$aug_gap_min" \
      WARMUP_EPOCHS="$warmup" \
      IMITATION_DECAY_START_EPOCH="$decay_start" \
      IMITATION_DECAY_END_EPOCH="$decay_end" \
      IMITATION_DECAY_MIN_SCALE="$decay_min_scale" \
      SSML_DISAGREEMENT_FLOOR_RATIO="$disagreement_floor_ratio" \
      SSML_DEFICIT_EMA_MOMENTUM="$deficit_ema_momentum" \
      SSML_EXTRA_CLASS_BUDGET_SCALE="$extra_class_budget_scale" \
      SSML_COMPLEMENT_RAMP_START_EPOCH="$complement_ramp_start" \
      SSML_COMPLEMENT_RAMP_END_EPOCH="$complement_ramp_end" \
      SSML_SECONDARY_PEER_REQUIRE_SAME_LABEL="$([[ "$secondary_agreement_min" != "0.00" ]] && echo 1 || echo 0)" \
      SSML_SECONDARY_PEER_INIT_CHECKPOINT_TEMPLATE="$BEST_CKPT_TEMPLATE" \
      SSML_SECONDARY_PEER_AGREEMENT_MIN="$secondary_agreement_min"
  done
fi

echo "[classification_cifar100_strict128_aggressive_v1] done"
