#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

METHODS="${METHODS:-independent dml ssml}"
DATASETS="${DATASETS:-darcy}"
SEEDS="${SEEDS:-0 1 2}"
MODEL_PAIRS="${MODEL_PAIRS:-fno:deeponet}"
INDEPENDENT_MODELS="${INDEPENDENT_MODELS:-$(collect_unique_models "$MODEL_PAIRS")}"
REQUIRE_DISTINCT_PEER="${REQUIRE_DISTINCT_PEER:-1}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-$(paper_results_root)/operator}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-mse}"
LAMBDA_IMITATION="${LAMBDA_IMITATION:-1.0}"
MARGIN="${MARGIN:-0.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:--1}"
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:--1}"
IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-1.0}"
HETERO_SSML_ONE_WAY="${HETERO_SSML_ONE_WAY:-0}"
SSML_STUDENT_ONLY="${SSML_STUDENT_ONLY:-0}"
SSML_FREEZE_PEER="${SSML_FREEZE_PEER:-0}"
OPERATOR_WEIGHT_GRANULARITY="${OPERATOR_WEIGHT_GRANULARITY:-sample}"
RELAY_STAGE_EPOCHS="${RELAY_STAGE_EPOCHS:-}"
RELAY_HINT_MODE="${RELAY_HINT_MODE:-full}"
RELAY_TAPER_SCHEDULE="${RELAY_TAPER_SCHEDULE:-linear}"
INIT_CHECKPOINT_TEMPLATE="${INIT_CHECKPOINT_TEMPLATE:-}"
PEER_INIT_CHECKPOINT_TEMPLATE="${PEER_INIT_CHECKPOINT_TEMPLATE:-}"
DOWNLOAD="${DOWNLOAD:-0}"

echo "[operator] output_dir=$OUTPUT_DIR"
echo "[operator] methods=$METHODS"
echo "[operator] model_pairs=$MODEL_PAIRS"
echo "[operator] relay_stage_epochs=$RELAY_STAGE_EPOCHS"
echo "[operator] relay_hint_mode=$RELAY_HINT_MODE"
echo "[operator] relay_taper_schedule=$RELAY_TAPER_SCHEDULE"

for dataset in $DATASETS; do
  for method in $METHODS; do
    if [[ "$method" == "independent" ]]; then
      for MODEL in $INDEPENDENT_MODELS; do
        for seed in $SEEDS; do
          cmd=(
            python -m runners.run_operator
            --method "$method"
            --model "$MODEL"
            --dataset "$dataset"
            --epochs "$EPOCHS"
            --batch-size "$BATCH_SIZE"
            --lr "$LR"
            --weight-decay "$WEIGHT_DECAY"
            --num-workers "$NUM_WORKERS"
            --seed "$seed"
            --device "$DEVICE"
            --output-dir "$OUTPUT_DIR"
            --regression-imitation-loss "$REGRESSION_IMITATION_LOSS"
            --lambda-imitation "$LAMBDA_IMITATION"
            --margin "$MARGIN"
            --warmup-epochs "$WARMUP_EPOCHS"
            --imitation-decay-start-epoch "$IMITATION_DECAY_START_EPOCH"
            --imitation-decay-end-epoch "$IMITATION_DECAY_END_EPOCH"
            --imitation-decay-min-scale "$IMITATION_DECAY_MIN_SCALE"
            --operator-weight-granularity "$OPERATOR_WEIGHT_GRANULARITY"
            --relay-stage-epochs "$RELAY_STAGE_EPOCHS"
            --relay-hint-mode "$RELAY_HINT_MODE"
            --relay-taper-schedule "$RELAY_TAPER_SCHEDULE"
          )

          init_checkpoint="${INIT_CHECKPOINT_TEMPLATE//\{seed\}/$seed}"

          if [[ "$DOWNLOAD" == "1" ]]; then
            cmd+=(--download)
          fi
          if [[ "$HETERO_SSML_ONE_WAY" == "1" ]]; then
            cmd+=(--hetero-ssml-one-way)
          fi
          if [[ -n "$init_checkpoint" ]]; then
            cmd+=(--init-checkpoint "$init_checkpoint")
          fi

          echo "[operator] dataset=$dataset model=$MODEL method=$method seed=$seed"
          "${cmd[@]}"
        done
      done
      continue
    fi

    for pair in $MODEL_PAIRS; do
      IFS=':' read -r MODEL PEER_MODEL <<< "$pair"
      if [[ "${REQUIRE_DISTINCT_PEER}" == "1" ]] && ! pair_is_distinct "$MODEL" "${PEER_MODEL:-}"; then
        echo "[operator][skip] pair must be heterogeneous: $pair" >&2
        continue
      fi

      for seed in $SEEDS; do
        cmd=(
          python -m runners.run_operator
          --method "$method"
          --model "$MODEL"
          --peer-model "$PEER_MODEL"
          --dataset "$dataset"
          --epochs "$EPOCHS"
          --batch-size "$BATCH_SIZE"
          --lr "$LR"
          --weight-decay "$WEIGHT_DECAY"
          --num-workers "$NUM_WORKERS"
          --seed "$seed"
          --device "$DEVICE"
          --output-dir "$OUTPUT_DIR"
          --regression-imitation-loss "$REGRESSION_IMITATION_LOSS"
          --lambda-imitation "$LAMBDA_IMITATION"
          --margin "$MARGIN"
          --warmup-epochs "$WARMUP_EPOCHS"
          --imitation-decay-start-epoch "$IMITATION_DECAY_START_EPOCH"
          --imitation-decay-end-epoch "$IMITATION_DECAY_END_EPOCH"
          --imitation-decay-min-scale "$IMITATION_DECAY_MIN_SCALE"
          --operator-weight-granularity "$OPERATOR_WEIGHT_GRANULARITY"
          --relay-stage-epochs "$RELAY_STAGE_EPOCHS"
          --relay-hint-mode "$RELAY_HINT_MODE"
          --relay-taper-schedule "$RELAY_TAPER_SCHEDULE"
        )

        init_checkpoint="${INIT_CHECKPOINT_TEMPLATE//\{seed\}/$seed}"
        peer_init_checkpoint="${PEER_INIT_CHECKPOINT_TEMPLATE//\{seed\}/$seed}"

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
        if [[ -n "$init_checkpoint" ]]; then
          cmd+=(--init-checkpoint "$init_checkpoint")
        fi
        if [[ -n "$peer_init_checkpoint" ]]; then
          cmd+=(--peer-init-checkpoint "$peer_init_checkpoint")
        fi

        echo "[operator] dataset=$dataset pair=$MODEL:$PEER_MODEL method=$method seed=$seed"
        "${cmd[@]}"
      done
    done
  done
done
