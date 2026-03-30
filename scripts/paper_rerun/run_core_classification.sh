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
CLASSIFICATION_IMITATION_LOSS="${CLASSIFICATION_IMITATION_LOSS:-kl}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
LAMBDA_IMITATION="${LAMBDA_IMITATION:-1.0}"
MARGIN="${MARGIN:-0.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:--1}"
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:--1}"
IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-1.0}"
HETERO_SSML_ONE_WAY="${HETERO_SSML_ONE_WAY:-0}"
SSML_STUDENT_ONLY="${SSML_STUDENT_ONLY:-0}"
SSML_FREEZE_PEER="${SSML_FREEZE_PEER:-0}"
SSML_WORSE_ONLY_UPDATE="${SSML_WORSE_ONLY_UPDATE:-0}"
SSML_ANCHOR_WEIGHT="${SSML_ANCHOR_WEIGHT:-0.0}"
SSML_TOPK_RATIO="${SSML_TOPK_RATIO:-0.3}"
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
SSML_PEER_STUDENT_PROB_GAP_MIN="${SSML_PEER_STUDENT_PROB_GAP_MIN:-0.0}"
DOWNLOAD="${DOWNLOAD:-1}"
LABEL_NOISE_CONDITIONS="${LABEL_NOISE_CONDITIONS:-none:0.0}"
SSML_DISAGREEMENT_ONLY="${SSML_DISAGREEMENT_ONLY:-0}"
SSML_CLASS_BALANCED_TOPK="${SSML_CLASS_BALANCED_TOPK:-0}"
SSML_PER_CLASS_BUDGET="${SSML_PER_CLASS_BUDGET:-0}"
INIT_CHECKPOINT_TEMPLATE="${INIT_CHECKPOINT_TEMPLATE:-}"
PEER_INIT_CHECKPOINT_TEMPLATE="${PEER_INIT_CHECKPOINT_TEMPLATE:-}"

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

echo "[classification] output_dir=$OUTPUT_DIR"
echo "[classification] methods=$METHODS"
echo "[classification] model_pairs=$MODEL_PAIRS"
echo "[classification] distill_temperature=$DISTILL_TEMPERATURE"
echo "[classification] ssml_topk_ratio=$SSML_TOPK_RATIO"
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
echo "[classification] ssml_peer_true_prob_threshold=$SSML_PEER_TRUE_PROB_THRESHOLD"
echo "[classification] ssml_peer_student_prob_gap_min=$SSML_PEER_STUDENT_PROB_GAP_MIN"
echo "[classification] ssml_freeze_peer=$SSML_FREEZE_PEER"
echo "[classification] ssml_worse_only_update=$SSML_WORSE_ONLY_UPDATE"
echo "[classification] ssml_anchor_weight=$SSML_ANCHOR_WEIGHT"
echo "[classification] init_checkpoint_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[classification] peer_init_checkpoint_template=$PEER_INIT_CHECKPOINT_TEMPLATE"

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
              --ssml-peer-true-prob-threshold "$SSML_PEER_TRUE_PROB_THRESHOLD"
              --ssml-peer-student-prob-gap-min "$SSML_PEER_STUDENT_PROB_GAP_MIN"
              --ssml-student-true-prob-max "$SSML_STUDENT_TRUE_PROB_MAX"
              --warmup-epochs "$WARMUP_EPOCHS"
              --imitation-decay-start-epoch "$IMITATION_DECAY_START_EPOCH"
              --imitation-decay-end-epoch "$IMITATION_DECAY_END_EPOCH"
              --imitation-decay-min-scale "$IMITATION_DECAY_MIN_SCALE"
            )

            if [[ "$NOISE_TYPE" != "none" && "$NOISE_RATE" != "0" && "$NOISE_RATE" != "0.0" ]]; then
              cmd+=(--label-noise-type "$NOISE_TYPE" --label-noise-rate "$NOISE_RATE")
            fi
            if [[ "$DOWNLOAD" == "1" ]]; then
              cmd+=(--download)
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
            if [[ -n "$init_checkpoint" ]]; then
              cmd+=(--init-checkpoint "$init_checkpoint")
            fi

            echo "[classification] dataset=$dataset model=$MODEL method=$method seed=$seed noise=$NOISE_TYPE:$NOISE_RATE"
            "${cmd[@]}"
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
            --ssml-peer-true-prob-threshold "$SSML_PEER_TRUE_PROB_THRESHOLD"
            --ssml-peer-student-prob-gap-min "$SSML_PEER_STUDENT_PROB_GAP_MIN"
            --ssml-student-true-prob-max "$SSML_STUDENT_TRUE_PROB_MAX"
            --warmup-epochs "$WARMUP_EPOCHS"
            --imitation-decay-start-epoch "$IMITATION_DECAY_START_EPOCH"
            --imitation-decay-end-epoch "$IMITATION_DECAY_END_EPOCH"
            --imitation-decay-min-scale "$IMITATION_DECAY_MIN_SCALE"
          )

          if [[ "$NOISE_TYPE" != "none" && "$NOISE_RATE" != "0" && "$NOISE_RATE" != "0.0" ]]; then
            cmd+=(--label-noise-type "$NOISE_TYPE" --label-noise-rate "$NOISE_RATE")
          fi
          if [[ "$DOWNLOAD" == "1" ]]; then
            cmd+=(--download)
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
          if [[ -n "$init_checkpoint" ]]; then
            cmd+=(--init-checkpoint "$init_checkpoint")
          fi
          if [[ -n "$peer_init_checkpoint" ]]; then
            cmd+=(--peer-init-checkpoint "$peer_init_checkpoint")
          fi

          echo "[classification] dataset=$dataset pair=$MODEL:$PEER_MODEL method=$method seed=$seed noise=$NOISE_TYPE:$NOISE_RATE"
          "${cmd[@]}"
        done
      done
    done
  done
done
