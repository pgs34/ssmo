#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

METHODS="${METHODS:-independent dml ssml}"
DATASETS="${DATASETS:-cifar10 cifar100}"
SEEDS="${SEEDS:-0 1 2}"
MODEL_PAIRS="${MODEL_PAIRS:-resnet18:vit_b16}"
INDEPENDENT_MODELS="${INDEPENDENT_MODELS:-$(collect_unique_models "$MODEL_PAIRS")}"
REQUIRE_DISTINCT_PEER="${REQUIRE_DISTINCT_PEER:-1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-$(paper_results_root)/classification}"
PROTOCOL_ID="${PROTOCOL_ID:-default}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-}"
OPTIMIZER="${OPTIMIZER:-adamw}"
MOMENTUM="${MOMENTUM:-0.9}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-none}"
SCHEDULER_WARMUP_EPOCHS="${SCHEDULER_WARMUP_EPOCHS:-0}"
SCHEDULER_MIN_SCALE="${SCHEDULER_MIN_SCALE:-0.0}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
MODEL_EMA_DECAY="${MODEL_EMA_DECAY:-0.0}"
TRAIN_AUG_MODE="${TRAIN_AUG_MODE:-basic}"
TRAIN_SUBSET_SIZE="${TRAIN_SUBSET_SIZE:-}"
VAL_SUBSET_SIZE="${VAL_SUBSET_SIZE:-}"
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
LAMBDA_IMITATION="${LAMBDA_IMITATION:-1.0}"
MARGIN="${MARGIN:-0.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:--1}"
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:--1}"
IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-1.0}"
FREEZE_BN_STATS="${FREEZE_BN_STATS:-0}"
FREEZE_BN_STATS_UNTIL_EPOCH="${FREEZE_BN_STATS_UNTIL_EPOCH:--1}"
HETERO_SSML_ONE_WAY="${HETERO_SSML_ONE_WAY:-0}"
SSML_STUDENT_ONLY="${SSML_STUDENT_ONLY:-0}"
SSML_FREEZE_PEER="${SSML_FREEZE_PEER:-0}"
SSML_WORSE_ONLY_UPDATE="${SSML_WORSE_ONLY_UPDATE:-0}"
SSML_ANCHOR_WEIGHT="${SSML_ANCHOR_WEIGHT:-0.0}"
SSML_TOPK_RATIO="${SSML_TOPK_RATIO:-0.3}"
SSML_TOPK_RATIO_START="${SSML_TOPK_RATIO_START:-}"
SSML_TOPK_RATIO_END="${SSML_TOPK_RATIO_END:-}"
SSML_TOPK_RAMP_START_EPOCH="${SSML_TOPK_RAMP_START_EPOCH:--1}"
SSML_TOPK_RAMP_END_EPOCH="${SSML_TOPK_RAMP_END_EPOCH:--1}"
SSML_TOPK_SCOPE="${SSML_TOPK_SCOPE:-total}"
SSML_SUPERVISED_HOTSPOT_ALPHA="${SSML_SUPERVISED_HOTSPOT_ALPHA:-0.0}"
SSML_SUPERVISED_WEIGHT_MODE="${SSML_SUPERVISED_WEIGHT_MODE:-score}"
SSML_GATE_SCORE_MODE="${SSML_GATE_SCORE_MODE:-peer_better_student_error}"
SSML_SCORE_TRANSFORM="${SSML_SCORE_TRANSFORM:-log1p}"
SSML_GUIDANCE_MODE="${SSML_GUIDANCE_MODE:-hybrid}"
SSML_PEER_CORRECT_ONLY="${SSML_PEER_CORRECT_ONLY:-0}"
SSML_STUDENT_INCORRECT_ONLY="${SSML_STUDENT_INCORRECT_ONLY:-0}"
SSML_STUDENT_TRUE_PROB_MAX="${SSML_STUDENT_TRUE_PROB_MAX:-1.0}"
SSML_PEER_TRUE_PROB_THRESHOLD="${SSML_PEER_TRUE_PROB_THRESHOLD:-0.0}"
SSML_PEER_TRUE_PROB_THRESHOLD_START="${SSML_PEER_TRUE_PROB_THRESHOLD_START:-}"
SSML_PEER_TRUE_PROB_THRESHOLD_END="${SSML_PEER_TRUE_PROB_THRESHOLD_END:-}"
SSML_PEER_STUDENT_PROB_GAP_MIN="${SSML_PEER_STUDENT_PROB_GAP_MIN:-0.0}"
SSML_PEER_STUDENT_PROB_GAP_MIN_START="${SSML_PEER_STUDENT_PROB_GAP_MIN_START:-}"
SSML_PEER_STUDENT_PROB_GAP_MIN_END="${SSML_PEER_STUDENT_PROB_GAP_MIN_END:-}"
DOWNLOAD="${DOWNLOAD:-1}"
LABEL_NOISE_CONDITIONS="${LABEL_NOISE_CONDITIONS:-none:0.0}"
SSML_DISAGREEMENT_ONLY="${SSML_DISAGREEMENT_ONLY:-0}"
SSML_CLASS_BALANCED_TOPK="${SSML_CLASS_BALANCED_TOPK:-0}"
SSML_PER_CLASS_BUDGET="${SSML_PER_CLASS_BUDGET:-0}"
SSML_DISAGREEMENT_FLOOR_RATIO="${SSML_DISAGREEMENT_FLOOR_RATIO:-0.0}"
SSML_DEFICIT_EMA_MOMENTUM="${SSML_DEFICIT_EMA_MOMENTUM:-0.0}"
SSML_EXTRA_CLASS_BUDGET_SCALE="${SSML_EXTRA_CLASS_BUDGET_SCALE:-0.0}"
SSML_COMPLEMENT_RAMP_START_EPOCH="${SSML_COMPLEMENT_RAMP_START_EPOCH:--1}"
SSML_COMPLEMENT_RAMP_END_EPOCH="${SSML_COMPLEMENT_RAMP_END_EPOCH:--1}"
SSML_SECONDARY_PEER_INIT_CHECKPOINT_TEMPLATE="${SSML_SECONDARY_PEER_INIT_CHECKPOINT_TEMPLATE:-}"
SSML_SECONDARY_PEER_REQUIRE_SAME_LABEL="${SSML_SECONDARY_PEER_REQUIRE_SAME_LABEL:-0}"
SSML_SECONDARY_PEER_AGREEMENT_MIN="${SSML_SECONDARY_PEER_AGREEMENT_MIN:-0.0}"
SSML_AUG_CONSISTENCY_WEIGHT="${SSML_AUG_CONSISTENCY_WEIGHT:-0.0}"
SSML_AUG_CONSISTENCY_SHIFT="${SSML_AUG_CONSISTENCY_SHIFT:-0}"
SSML_AUG_CONSISTENCY_FLIP_PROB="${SSML_AUG_CONSISTENCY_FLIP_PROB:-0.0}"
SSML_AUG_CONSISTENCY_NOISE_STD="${SSML_AUG_CONSISTENCY_NOISE_STD:-0.0}"
SSML_PEER_AUG_CONSISTENCY_MIN="${SSML_PEER_AUG_CONSISTENCY_MIN:-0.0}"
SSML_STUDENT_AUG_CONSISTENCY_MAX="${SSML_STUDENT_AUG_CONSISTENCY_MAX:-1.0}"
SSML_PEER_STUDENT_AUG_CONSISTENCY_GAP_MIN="${SSML_PEER_STUDENT_AUG_CONSISTENCY_GAP_MIN:-0.0}"
INIT_CHECKPOINT_TEMPLATE="${INIT_CHECKPOINT_TEMPLATE:-}"
PEER_INIT_CHECKPOINT_TEMPLATE="${PEER_INIT_CHECKPOINT_TEMPLATE:-}"
CLASSIFICATION_MAX_PARALLEL_RUNS="${CLASSIFICATION_MAX_PARALLEL_RUNS:-${MAX_PARALLEL_RUNS:-all}}"

render_checkpoint_template() {
  local template="$1"
  local dataset="$2"
  local model="$3"
  local peer_model="$4"
  local seed="$5"
  if [[ -z "$template" ]]; then
    return 0
  fi
  template="${template//\{dataset\}/$dataset}"
  template="${template//\{model\}/$model}"
  template="${template//\{peer_model\}/$peer_model}"
  template="${template//\{seed\}/$seed}"
  template="${template//\{classification_imitation_loss\}/$CLASSIFICATION_IMITATION_LOSS}"
  printf '%s\n' "$template"
}

resolve_checkpoint_path() {
  local path="$1"
  local fallback_path=""
  if [[ -z "$path" ]]; then
    return 0
  fi
  if [[ -f "$path" ]]; then
    printf '%s\n' "$path"
    return 0
  fi
  if [[ "$(basename "$path")" == "best_model.pt" ]]; then
    fallback_path="$(dirname "$path")/model.pt"
    if [[ -f "$fallback_path" ]]; then
      echo "[classification] checkpoint fallback: $path -> $fallback_path" >&2
      printf '%s\n' "$fallback_path"
      return 0
    fi
  fi
  printf '%s\n' "$path"
}

echo "[classification] output_dir=$OUTPUT_DIR"
echo "[classification] protocol_id=$PROTOCOL_ID"
echo "[classification] hardware_profile=$HARDWARE_PROFILE"
echo "[classification] methods=$METHODS"
echo "[classification] model_pairs=$MODEL_PAIRS"
echo "[classification] device_request=$DEVICE"
echo "[classification] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[classification] optimizer=$OPTIMIZER momentum=$MOMENTUM"
echo "[classification] lr=$LR weight_decay=$WEIGHT_DECAY"
echo "[classification] lr_scheduler=$LR_SCHEDULER warmup=$SCHEDULER_WARMUP_EPOCHS min_scale=$SCHEDULER_MIN_SCALE"
echo "[classification] label_smoothing=$LABEL_SMOOTHING grad_clip=$GRAD_CLIP model_ema_decay=$MODEL_EMA_DECAY"
echo "[classification] train_aug_mode=$TRAIN_AUG_MODE"
echo "[classification] train_subset_size=${TRAIN_SUBSET_SIZE:-<full>} val_subset_size=${VAL_SUBSET_SIZE:-<full>}"
echo "[classification] distill_temperature=$DISTILL_TEMPERATURE"
echo "[classification] freeze_bn_stats=$FREEZE_BN_STATS"
echo "[classification] freeze_bn_stats_until_epoch=$FREEZE_BN_STATS_UNTIL_EPOCH"
echo "[classification] ssml_topk_ratio=$SSML_TOPK_RATIO"
echo "[classification] ssml_topk_ratio_start=$SSML_TOPK_RATIO_START ssml_topk_ratio_end=$SSML_TOPK_RATIO_END"
echo "[classification] ssml_topk_ramp_start_epoch=$SSML_TOPK_RAMP_START_EPOCH ssml_topk_ramp_end_epoch=$SSML_TOPK_RAMP_END_EPOCH"
echo "[classification] ssml_topk_scope=$SSML_TOPK_SCOPE"
echo "[classification] ssml_supervised_hotspot_alpha=$SSML_SUPERVISED_HOTSPOT_ALPHA"
echo "[classification] ssml_supervised_weight_mode=$SSML_SUPERVISED_WEIGHT_MODE"
echo "[classification] ssml_gate_score_mode=$SSML_GATE_SCORE_MODE"
echo "[classification] ssml_score_transform=$SSML_SCORE_TRANSFORM"
echo "[classification] ssml_guidance_mode=$SSML_GUIDANCE_MODE"
echo "[classification] ssml_peer_correct_only=$SSML_PEER_CORRECT_ONLY"
echo "[classification] ssml_student_incorrect_only=$SSML_STUDENT_INCORRECT_ONLY"
echo "[classification] ssml_student_true_prob_max=$SSML_STUDENT_TRUE_PROB_MAX"
echo "[classification] ssml_disagreement_only=$SSML_DISAGREEMENT_ONLY"
echo "[classification] ssml_class_balanced_topk=$SSML_CLASS_BALANCED_TOPK"
echo "[classification] ssml_per_class_budget=$SSML_PER_CLASS_BUDGET"
echo "[classification] ssml_disagreement_floor_ratio=$SSML_DISAGREEMENT_FLOOR_RATIO"
echo "[classification] ssml_deficit_ema_momentum=$SSML_DEFICIT_EMA_MOMENTUM"
echo "[classification] ssml_extra_class_budget_scale=$SSML_EXTRA_CLASS_BUDGET_SCALE"
echo "[classification] ssml_complement_ramp_start_epoch=$SSML_COMPLEMENT_RAMP_START_EPOCH"
echo "[classification] ssml_complement_ramp_end_epoch=$SSML_COMPLEMENT_RAMP_END_EPOCH"
echo "[classification] ssml_secondary_peer_init_checkpoint_template=$SSML_SECONDARY_PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[classification] ssml_secondary_peer_require_same_label=$SSML_SECONDARY_PEER_REQUIRE_SAME_LABEL"
echo "[classification] ssml_secondary_peer_agreement_min=$SSML_SECONDARY_PEER_AGREEMENT_MIN"
echo "[classification] ssml_peer_true_prob_threshold=$SSML_PEER_TRUE_PROB_THRESHOLD"
echo "[classification] ssml_peer_true_prob_threshold_start=$SSML_PEER_TRUE_PROB_THRESHOLD_START"
echo "[classification] ssml_peer_true_prob_threshold_end=$SSML_PEER_TRUE_PROB_THRESHOLD_END"
echo "[classification] ssml_peer_student_prob_gap_min=$SSML_PEER_STUDENT_PROB_GAP_MIN"
echo "[classification] ssml_peer_student_prob_gap_min_start=$SSML_PEER_STUDENT_PROB_GAP_MIN_START"
echo "[classification] ssml_peer_student_prob_gap_min_end=$SSML_PEER_STUDENT_PROB_GAP_MIN_END"
echo "[classification] ssml_aug_consistency_weight=$SSML_AUG_CONSISTENCY_WEIGHT"
echo "[classification] ssml_aug_consistency_shift=$SSML_AUG_CONSISTENCY_SHIFT"
echo "[classification] ssml_aug_consistency_flip_prob=$SSML_AUG_CONSISTENCY_FLIP_PROB"
echo "[classification] ssml_aug_consistency_noise_std=$SSML_AUG_CONSISTENCY_NOISE_STD"
echo "[classification] ssml_peer_aug_consistency_min=$SSML_PEER_AUG_CONSISTENCY_MIN"
echo "[classification] ssml_student_aug_consistency_max=$SSML_STUDENT_AUG_CONSISTENCY_MAX"
echo "[classification] ssml_peer_student_aug_consistency_gap_min=$SSML_PEER_STUDENT_AUG_CONSISTENCY_GAP_MIN"
echo "[classification] ssml_freeze_peer=$SSML_FREEZE_PEER"
echo "[classification] ssml_worse_only_update=$SSML_WORSE_ONLY_UPDATE"
echo "[classification] ssml_anchor_weight=$SSML_ANCHOR_WEIGHT"
echo "[classification] init_checkpoint_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[classification] peer_init_checkpoint_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[classification] max_parallel_runs=$CLASSIFICATION_MAX_PARALLEL_RUNS"

parallel_exec_init "$CLASSIFICATION_MAX_PARALLEL_RUNS"
trap parallel_exec_cleanup INT TERM

for dataset in $DATASETS; do
  for noise_condition in $LABEL_NOISE_CONDITIONS; do
    IFS=':' read -r NOISE_TYPE NOISE_RATE <<< "$noise_condition"
    if [[ -z "${NOISE_RATE:-}" ]]; then
      NOISE_RATE="0.0"
    fi

    for method in $METHODS; do
      if [[ "$method" == "independent" ]]; then
          for MODEL in $INDEPENDENT_MODELS; do
          for seed in $SEEDS; do
            init_checkpoint="$(render_checkpoint_template "$INIT_CHECKPOINT_TEMPLATE" "$dataset" "$MODEL" "" "$seed")"
            init_checkpoint="$(resolve_checkpoint_path "$init_checkpoint")"
            cmd=(
              python -m runners.run_classification
              --method "$method"
              --model "$MODEL"
              --dataset "$dataset"
              --epochs "$EPOCHS"
              --batch-size "$BATCH_SIZE"
              --num-workers "$NUM_WORKERS"
              --seed "$seed"
              --device "$DEVICE"
              --output-dir "$OUTPUT_DIR"
              --protocol-id "$PROTOCOL_ID"
              --hardware-profile "$HARDWARE_PROFILE"
              --optimizer "$OPTIMIZER"
              --momentum "$MOMENTUM"
              --lr "$LR"
              --weight-decay "$WEIGHT_DECAY"
              --lr-scheduler "$LR_SCHEDULER"
              --scheduler-warmup-epochs "$SCHEDULER_WARMUP_EPOCHS"
              --scheduler-min-scale "$SCHEDULER_MIN_SCALE"
              --label-smoothing "$LABEL_SMOOTHING"
              --grad-clip "$GRAD_CLIP"
              --model-ema-decay "$MODEL_EMA_DECAY"
              --train-aug-mode "$TRAIN_AUG_MODE"
              --classification-imitation-loss "$CLASSIFICATION_IMITATION_LOSS"
              --distill-temperature "$DISTILL_TEMPERATURE"
              --lambda-imitation "$LAMBDA_IMITATION"
              --margin "$MARGIN"
              --ssml-topk-ratio "$SSML_TOPK_RATIO"
              --ssml-topk-scope "$SSML_TOPK_SCOPE"
              --ssml-supervised-hotspot-alpha "$SSML_SUPERVISED_HOTSPOT_ALPHA"
              --ssml-supervised-weight-mode "$SSML_SUPERVISED_WEIGHT_MODE"
              --ssml-gate-score-mode "$SSML_GATE_SCORE_MODE"
              --ssml-score-transform "$SSML_SCORE_TRANSFORM"
              --ssml-guidance-mode "$SSML_GUIDANCE_MODE"
              --ssml-per-class-budget "$SSML_PER_CLASS_BUDGET"
              --ssml-disagreement-floor-ratio "$SSML_DISAGREEMENT_FLOOR_RATIO"
              --ssml-deficit-ema-momentum "$SSML_DEFICIT_EMA_MOMENTUM"
              --ssml-extra-class-budget-scale "$SSML_EXTRA_CLASS_BUDGET_SCALE"
              --ssml-complement-ramp-start-epoch "$SSML_COMPLEMENT_RAMP_START_EPOCH"
              --ssml-complement-ramp-end-epoch "$SSML_COMPLEMENT_RAMP_END_EPOCH"
              --ssml-secondary-peer-agreement-min "$SSML_SECONDARY_PEER_AGREEMENT_MIN"
              --ssml-peer-true-prob-threshold "$SSML_PEER_TRUE_PROB_THRESHOLD"
              --ssml-peer-student-prob-gap-min "$SSML_PEER_STUDENT_PROB_GAP_MIN"
              --ssml-student-true-prob-max "$SSML_STUDENT_TRUE_PROB_MAX"
              --ssml-aug-consistency-weight "$SSML_AUG_CONSISTENCY_WEIGHT"
              --ssml-aug-consistency-shift "$SSML_AUG_CONSISTENCY_SHIFT"
              --ssml-aug-consistency-flip-prob "$SSML_AUG_CONSISTENCY_FLIP_PROB"
              --ssml-aug-consistency-noise-std "$SSML_AUG_CONSISTENCY_NOISE_STD"
              --ssml-peer-aug-consistency-min "$SSML_PEER_AUG_CONSISTENCY_MIN"
              --ssml-student-aug-consistency-max "$SSML_STUDENT_AUG_CONSISTENCY_MAX"
              --ssml-peer-student-aug-consistency-gap-min "$SSML_PEER_STUDENT_AUG_CONSISTENCY_GAP_MIN"
              --warmup-epochs "$WARMUP_EPOCHS"
              --imitation-decay-start-epoch "$IMITATION_DECAY_START_EPOCH"
              --imitation-decay-end-epoch "$IMITATION_DECAY_END_EPOCH"
              --imitation-decay-min-scale "$IMITATION_DECAY_MIN_SCALE"
              --freeze-bn-stats-until-epoch "$FREEZE_BN_STATS_UNTIL_EPOCH"
            )
            if [[ -n "$TRAIN_SUBSET_SIZE" ]]; then
              cmd+=(--train-subset-size "$TRAIN_SUBSET_SIZE")
            fi
            if [[ -n "$VAL_SUBSET_SIZE" ]]; then
              cmd+=(--val-subset-size "$VAL_SUBSET_SIZE")
            fi

            if [[ -n "$SSML_TOPK_RATIO_START" ]]; then
              cmd+=(--ssml-topk-ratio-start "$SSML_TOPK_RATIO_START")
            fi
            if [[ -n "$SSML_TOPK_RATIO_END" ]]; then
              cmd+=(--ssml-topk-ratio-end "$SSML_TOPK_RATIO_END")
            fi
            cmd+=(--ssml-topk-ramp-start-epoch "$SSML_TOPK_RAMP_START_EPOCH")
            cmd+=(--ssml-topk-ramp-end-epoch "$SSML_TOPK_RAMP_END_EPOCH")
            if [[ -n "$SSML_PEER_TRUE_PROB_THRESHOLD_START" ]]; then
              cmd+=(--ssml-peer-true-prob-threshold-start "$SSML_PEER_TRUE_PROB_THRESHOLD_START")
            fi
            if [[ -n "$SSML_PEER_TRUE_PROB_THRESHOLD_END" ]]; then
              cmd+=(--ssml-peer-true-prob-threshold-end "$SSML_PEER_TRUE_PROB_THRESHOLD_END")
            fi
            if [[ -n "$SSML_PEER_STUDENT_PROB_GAP_MIN_START" ]]; then
              cmd+=(--ssml-peer-student-prob-gap-min-start "$SSML_PEER_STUDENT_PROB_GAP_MIN_START")
            fi
            if [[ -n "$SSML_PEER_STUDENT_PROB_GAP_MIN_END" ]]; then
              cmd+=(--ssml-peer-student-prob-gap-min-end "$SSML_PEER_STUDENT_PROB_GAP_MIN_END")
            fi
            if [[ "$NOISE_TYPE" != "none" && "$NOISE_RATE" != "0" && "$NOISE_RATE" != "0.0" ]]; then
              cmd+=(--label-noise-type "$NOISE_TYPE" --label-noise-rate "$NOISE_RATE")
            fi
            if [[ "$DOWNLOAD" == "1" ]]; then
              cmd+=(--download)
            fi
            if [[ "$FREEZE_BN_STATS" == "1" ]]; then
              cmd+=(--freeze-bn-stats)
            fi
            if [[ "$HETERO_SSML_ONE_WAY" == "1" ]]; then
              cmd+=(--hetero-ssml-one-way)
            fi
            if [[ "$SSML_STUDENT_ONLY" == "1" ]]; then
              cmd+=(--ssml-student-only)
            fi
            if [[ "$SSML_FREEZE_PEER" == "1" ]]; then
              cmd+=(--ssml-freeze-peer)
            fi
            if [[ "$SSML_WORSE_ONLY_UPDATE" == "1" ]]; then
              cmd+=(--ssml-worse-only-update)
            fi
            cmd+=(--ssml-anchor-weight "$SSML_ANCHOR_WEIGHT")
            if [[ "$SSML_PEER_CORRECT_ONLY" == "1" ]]; then
              cmd+=(--ssml-peer-correct-only)
            fi
            if [[ "$SSML_STUDENT_INCORRECT_ONLY" == "1" ]]; then
              cmd+=(--ssml-student-incorrect-only)
            fi
            if [[ "$SSML_DISAGREEMENT_ONLY" == "1" ]]; then
              cmd+=(--ssml-disagreement-only)
            fi
            if [[ "$SSML_CLASS_BALANCED_TOPK" == "1" ]]; then
              cmd+=(--ssml-class-balanced-topk)
            fi
            if [[ "$SSML_SECONDARY_PEER_REQUIRE_SAME_LABEL" == "1" ]]; then
              cmd+=(--ssml-secondary-peer-require-same-label)
            fi
            if [[ -n "$init_checkpoint" ]]; then
              cmd+=(--init-checkpoint "$init_checkpoint")
            fi

            job_label="dataset=$dataset model=$MODEL method=$method seed=$seed noise=$NOISE_TYPE:$NOISE_RATE"
            echo "[classification][queue] $job_label"
            parallel_exec_submit "$job_label" "${cmd[@]}"
          done
        done
        continue
      fi

      for pair in $MODEL_PAIRS; do
        IFS=':' read -r MODEL PEER_MODEL <<< "$pair"
        if [[ "${REQUIRE_DISTINCT_PEER}" == "1" ]] && ! pair_is_distinct "$MODEL" "${PEER_MODEL:-}"; then
          echo "[classification][skip] pair must be heterogeneous: $pair" >&2
          continue
        fi

        for seed in $SEEDS; do
          init_checkpoint="$(render_checkpoint_template "$INIT_CHECKPOINT_TEMPLATE" "$dataset" "$MODEL" "$PEER_MODEL" "$seed")"
          peer_init_checkpoint="$(render_checkpoint_template "$PEER_INIT_CHECKPOINT_TEMPLATE" "$dataset" "$PEER_MODEL" "$MODEL" "$seed")"
          secondary_peer_init_checkpoint="$(render_checkpoint_template "$SSML_SECONDARY_PEER_INIT_CHECKPOINT_TEMPLATE" "$dataset" "$PEER_MODEL" "$MODEL" "$seed")"
          init_checkpoint="$(resolve_checkpoint_path "$init_checkpoint")"
          peer_init_checkpoint="$(resolve_checkpoint_path "$peer_init_checkpoint")"
          secondary_peer_init_checkpoint="$(resolve_checkpoint_path "$secondary_peer_init_checkpoint")"
          cmd=(
            python -m runners.run_classification
            --method "$method"
            --model "$MODEL"
            --peer-model "$PEER_MODEL"
            --dataset "$dataset"
            --epochs "$EPOCHS"
            --batch-size "$BATCH_SIZE"
            --num-workers "$NUM_WORKERS"
            --seed "$seed"
            --device "$DEVICE"
            --output-dir "$OUTPUT_DIR"
            --protocol-id "$PROTOCOL_ID"
            --hardware-profile "$HARDWARE_PROFILE"
            --optimizer "$OPTIMIZER"
            --momentum "$MOMENTUM"
            --lr "$LR"
            --weight-decay "$WEIGHT_DECAY"
            --lr-scheduler "$LR_SCHEDULER"
            --scheduler-warmup-epochs "$SCHEDULER_WARMUP_EPOCHS"
            --scheduler-min-scale "$SCHEDULER_MIN_SCALE"
            --label-smoothing "$LABEL_SMOOTHING"
            --grad-clip "$GRAD_CLIP"
            --model-ema-decay "$MODEL_EMA_DECAY"
            --train-aug-mode "$TRAIN_AUG_MODE"
            --classification-imitation-loss "$CLASSIFICATION_IMITATION_LOSS"
            --distill-temperature "$DISTILL_TEMPERATURE"
            --lambda-imitation "$LAMBDA_IMITATION"
            --margin "$MARGIN"
            --ssml-topk-ratio "$SSML_TOPK_RATIO"
            --ssml-topk-scope "$SSML_TOPK_SCOPE"
            --ssml-supervised-hotspot-alpha "$SSML_SUPERVISED_HOTSPOT_ALPHA"
            --ssml-supervised-weight-mode "$SSML_SUPERVISED_WEIGHT_MODE"
            --ssml-gate-score-mode "$SSML_GATE_SCORE_MODE"
            --ssml-score-transform "$SSML_SCORE_TRANSFORM"
            --ssml-guidance-mode "$SSML_GUIDANCE_MODE"
            --ssml-per-class-budget "$SSML_PER_CLASS_BUDGET"
            --ssml-disagreement-floor-ratio "$SSML_DISAGREEMENT_FLOOR_RATIO"
            --ssml-deficit-ema-momentum "$SSML_DEFICIT_EMA_MOMENTUM"
            --ssml-extra-class-budget-scale "$SSML_EXTRA_CLASS_BUDGET_SCALE"
            --ssml-complement-ramp-start-epoch "$SSML_COMPLEMENT_RAMP_START_EPOCH"
            --ssml-complement-ramp-end-epoch "$SSML_COMPLEMENT_RAMP_END_EPOCH"
            --ssml-secondary-peer-agreement-min "$SSML_SECONDARY_PEER_AGREEMENT_MIN"
            --ssml-peer-true-prob-threshold "$SSML_PEER_TRUE_PROB_THRESHOLD"
            --ssml-peer-student-prob-gap-min "$SSML_PEER_STUDENT_PROB_GAP_MIN"
            --ssml-student-true-prob-max "$SSML_STUDENT_TRUE_PROB_MAX"
            --ssml-aug-consistency-weight "$SSML_AUG_CONSISTENCY_WEIGHT"
            --ssml-aug-consistency-shift "$SSML_AUG_CONSISTENCY_SHIFT"
            --ssml-aug-consistency-flip-prob "$SSML_AUG_CONSISTENCY_FLIP_PROB"
            --ssml-aug-consistency-noise-std "$SSML_AUG_CONSISTENCY_NOISE_STD"
            --ssml-peer-aug-consistency-min "$SSML_PEER_AUG_CONSISTENCY_MIN"
            --ssml-student-aug-consistency-max "$SSML_STUDENT_AUG_CONSISTENCY_MAX"
            --ssml-peer-student-aug-consistency-gap-min "$SSML_PEER_STUDENT_AUG_CONSISTENCY_GAP_MIN"
            --warmup-epochs "$WARMUP_EPOCHS"
            --imitation-decay-start-epoch "$IMITATION_DECAY_START_EPOCH"
            --imitation-decay-end-epoch "$IMITATION_DECAY_END_EPOCH"
            --imitation-decay-min-scale "$IMITATION_DECAY_MIN_SCALE"
            --freeze-bn-stats-until-epoch "$FREEZE_BN_STATS_UNTIL_EPOCH"
          )
          if [[ -n "$TRAIN_SUBSET_SIZE" ]]; then
            cmd+=(--train-subset-size "$TRAIN_SUBSET_SIZE")
          fi
          if [[ -n "$VAL_SUBSET_SIZE" ]]; then
            cmd+=(--val-subset-size "$VAL_SUBSET_SIZE")
          fi

          if [[ -n "$SSML_TOPK_RATIO_START" ]]; then
            cmd+=(--ssml-topk-ratio-start "$SSML_TOPK_RATIO_START")
          fi
          if [[ -n "$SSML_TOPK_RATIO_END" ]]; then
            cmd+=(--ssml-topk-ratio-end "$SSML_TOPK_RATIO_END")
          fi
          cmd+=(--ssml-topk-ramp-start-epoch "$SSML_TOPK_RAMP_START_EPOCH")
          cmd+=(--ssml-topk-ramp-end-epoch "$SSML_TOPK_RAMP_END_EPOCH")
          if [[ -n "$SSML_PEER_TRUE_PROB_THRESHOLD_START" ]]; then
            cmd+=(--ssml-peer-true-prob-threshold-start "$SSML_PEER_TRUE_PROB_THRESHOLD_START")
          fi
          if [[ -n "$SSML_PEER_TRUE_PROB_THRESHOLD_END" ]]; then
            cmd+=(--ssml-peer-true-prob-threshold-end "$SSML_PEER_TRUE_PROB_THRESHOLD_END")
          fi
          if [[ -n "$SSML_PEER_STUDENT_PROB_GAP_MIN_START" ]]; then
            cmd+=(--ssml-peer-student-prob-gap-min-start "$SSML_PEER_STUDENT_PROB_GAP_MIN_START")
          fi
          if [[ -n "$SSML_PEER_STUDENT_PROB_GAP_MIN_END" ]]; then
            cmd+=(--ssml-peer-student-prob-gap-min-end "$SSML_PEER_STUDENT_PROB_GAP_MIN_END")
          fi
          if [[ "$NOISE_TYPE" != "none" && "$NOISE_RATE" != "0" && "$NOISE_RATE" != "0.0" ]]; then
            cmd+=(--label-noise-type "$NOISE_TYPE" --label-noise-rate "$NOISE_RATE")
          fi
          if [[ "$DOWNLOAD" == "1" ]]; then
            cmd+=(--download)
          fi
          if [[ "$FREEZE_BN_STATS" == "1" ]]; then
            cmd+=(--freeze-bn-stats)
          fi
          if [[ "$HETERO_SSML_ONE_WAY" == "1" ]]; then
            cmd+=(--hetero-ssml-one-way)
          fi
          if [[ "$SSML_STUDENT_ONLY" == "1" ]]; then
            cmd+=(--ssml-student-only)
          fi
          if [[ "$SSML_FREEZE_PEER" == "1" ]]; then
            cmd+=(--ssml-freeze-peer)
          fi
          if [[ "$SSML_WORSE_ONLY_UPDATE" == "1" ]]; then
            cmd+=(--ssml-worse-only-update)
          fi
          cmd+=(--ssml-anchor-weight "$SSML_ANCHOR_WEIGHT")
          if [[ "$SSML_PEER_CORRECT_ONLY" == "1" ]]; then
            cmd+=(--ssml-peer-correct-only)
          fi
          if [[ "$SSML_STUDENT_INCORRECT_ONLY" == "1" ]]; then
            cmd+=(--ssml-student-incorrect-only)
          fi
          if [[ "$SSML_DISAGREEMENT_ONLY" == "1" ]]; then
            cmd+=(--ssml-disagreement-only)
          fi
          if [[ "$SSML_CLASS_BALANCED_TOPK" == "1" ]]; then
            cmd+=(--ssml-class-balanced-topk)
          fi
          if [[ "$SSML_SECONDARY_PEER_REQUIRE_SAME_LABEL" == "1" ]]; then
            cmd+=(--ssml-secondary-peer-require-same-label)
          fi
          if [[ -n "$init_checkpoint" ]]; then
            cmd+=(--init-checkpoint "$init_checkpoint")
          fi
          if [[ -n "$peer_init_checkpoint" ]]; then
            cmd+=(--peer-init-checkpoint "$peer_init_checkpoint")
          fi
          if [[ -n "$secondary_peer_init_checkpoint" ]]; then
            cmd+=(--ssml-secondary-peer-init-checkpoint "$secondary_peer_init_checkpoint")
          fi

          job_label="dataset=$dataset pair=$MODEL:$PEER_MODEL method=$method seed=$seed noise=$NOISE_TYPE:$NOISE_RATE"
          echo "[classification][queue] $job_label"
          parallel_exec_submit "$job_label" "${cmd[@]}"
        done
      done
    done
  done
done

parallel_exec_wait_all
