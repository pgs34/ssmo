#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

METHODS="${METHODS:-independent dml ssml}"
DATASETS="${DATASETS:-etth1 electricity weather}"
SEEDS="${SEEDS:-0 1 2}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
INDEPENDENT_MODELS="${INDEPENDENT_MODELS:-$(collect_unique_models "$MODEL_PAIRS")}"
REQUIRE_DISTINCT_PEER="${REQUIRE_DISTINCT_PEER:-1}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-$(paper_results_root)/time_series}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-mse}"
LAMBDA_IMITATION="${LAMBDA_IMITATION:-1.0}"
MARGIN="${MARGIN:-0.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
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
SSML_MAX_SELECTED_RATIO="${SSML_MAX_SELECTED_RATIO:-1.0}"
SSML_ADAPTIVE_DENSE_THRESHOLD="${SSML_ADAPTIVE_DENSE_THRESHOLD:-1.1}"
SSML_ADAPTIVE_DENSE_TOPK_RATIO="${SSML_ADAPTIVE_DENSE_TOPK_RATIO:-1.0}"
SSML_ADAPTIVE_DENSE_TOPK_SCOPE="${SSML_ADAPTIVE_DENSE_TOPK_SCOPE:-positive}"
SSML_ADAPTIVE_DENSE_MAX_SELECTED_RATIO="${SSML_ADAPTIVE_DENSE_MAX_SELECTED_RATIO:-1.0}"
SSML_ADAPTIVE_DENSE_SCORE_SMOOTHING_KERNEL="${SSML_ADAPTIVE_DENSE_SCORE_SMOOTHING_KERNEL:--1}"
SSML_ADAPTIVE_DENSE_WINDOW_EXPAND_KERNEL="${SSML_ADAPTIVE_DENSE_WINDOW_EXPAND_KERNEL:--1}"
SSML_SUPERVISED_HOTSPOT_ALPHA="${SSML_SUPERVISED_HOTSPOT_ALPHA:-0.0}"
SSML_SUPERVISED_WEIGHT_MODE="${SSML_SUPERVISED_WEIGHT_MODE:-score}"
SSML_GATE_SCORE_MODE="${SSML_GATE_SCORE_MODE:-peer_better_student_error}"
SSML_SCORE_TRANSFORM="${SSML_SCORE_TRANSFORM:-none}"
SSML_POSITIVE_UPPER_QUANTILE="${SSML_POSITIVE_UPPER_QUANTILE:-1.0}"
SSML_GUIDANCE_MODE="${SSML_GUIDANCE_MODE:-hybrid}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
SSML_SCORE_SMOOTHING_KERNEL="${SSML_SCORE_SMOOTHING_KERNEL:-1}"
SSML_WINDOW_SCORE_KERNEL="${SSML_WINDOW_SCORE_KERNEL:-1}"
SSML_WINDOW_EXPAND_KERNEL="${SSML_WINDOW_EXPAND_KERNEL:-1}"
SSML_TAIL_START_RATIO="${SSML_TAIL_START_RATIO:-0.0}"
SSML_RESIDUAL_BETA="${SSML_RESIDUAL_BETA:-1.0}"
SSML_EMA_DECAY="${SSML_EMA_DECAY:-0.0}"
SSML_IMITATION_SPACE="${SSML_IMITATION_SPACE:-raw}"
SSML_RESIDUAL_SPACE_KERNEL="${SSML_RESIDUAL_SPACE_KERNEL:-9}"
SSML_CONFLICT_AWARE_PROJECTION="${SSML_CONFLICT_AWARE_PROJECTION:-0}"
INIT_CHECKPOINT_TEMPLATE="${INIT_CHECKPOINT_TEMPLATE:-}"
PEER_INIT_CHECKPOINT_TEMPLATE="${PEER_INIT_CHECKPOINT_TEMPLATE:-}"
LIVE_PLOT_INTERVAL="${LIVE_PLOT_INTERVAL:-20}"

render_checkpoint_template() {
  local template="$1"
  local dataset="$2"
  local model="$3"
  local peer_model="$4"
  local seed="$5"
  local pred_len="$6"
  if [[ -z "$template" ]]; then
    return 0
  fi
  template="${template//\{dataset\}/$dataset}"
  template="${template//\{model\}/$model}"
  template="${template//\{peer_model\}/$peer_model}"
  template="${template//\{seed\}/$seed}"
  template="${template//\{pred_len\}/$pred_len}"
  template="${template//\{regression_imitation_loss\}/$REGRESSION_IMITATION_LOSS}"
  template="${template//\{feature_mode\}/$FEATURE_MODE}"
  printf '%s\n' "$template"
}

echo "[time_series] output_dir=$OUTPUT_DIR"
echo "[time_series] methods=$METHODS"
echo "[time_series] model_pairs=$MODEL_PAIRS"
echo "[time_series] ssml_topk_ratio=$SSML_TOPK_RATIO"
echo "[time_series] ssml_topk_scope=$SSML_TOPK_SCOPE"
echo "[time_series] ssml_max_selected_ratio=$SSML_MAX_SELECTED_RATIO"
echo "[time_series] ssml_adaptive_dense_threshold=$SSML_ADAPTIVE_DENSE_THRESHOLD"
echo "[time_series] ssml_adaptive_dense_topk_ratio=$SSML_ADAPTIVE_DENSE_TOPK_RATIO"
echo "[time_series] ssml_adaptive_dense_topk_scope=$SSML_ADAPTIVE_DENSE_TOPK_SCOPE"
echo "[time_series] ssml_adaptive_dense_max_selected_ratio=$SSML_ADAPTIVE_DENSE_MAX_SELECTED_RATIO"
echo "[time_series] ssml_adaptive_dense_score_smoothing_kernel=$SSML_ADAPTIVE_DENSE_SCORE_SMOOTHING_KERNEL"
echo "[time_series] ssml_adaptive_dense_window_expand_kernel=$SSML_ADAPTIVE_DENSE_WINDOW_EXPAND_KERNEL"
echo "[time_series] ssml_supervised_hotspot_alpha=$SSML_SUPERVISED_HOTSPOT_ALPHA"
echo "[time_series] ssml_supervised_weight_mode=$SSML_SUPERVISED_WEIGHT_MODE"
echo "[time_series] ssml_gate_score_mode=$SSML_GATE_SCORE_MODE"
echo "[time_series] ssml_score_transform=$SSML_SCORE_TRANSFORM"
echo "[time_series] ssml_positive_upper_quantile=$SSML_POSITIVE_UPPER_QUANTILE"
echo "[time_series] ssml_guidance_mode=$SSML_GUIDANCE_MODE"
echo "[time_series] ssml_score_smoothing_kernel=$SSML_SCORE_SMOOTHING_KERNEL"
echo "[time_series] ssml_window_score_kernel=$SSML_WINDOW_SCORE_KERNEL"
echo "[time_series] ssml_window_expand_kernel=$SSML_WINDOW_EXPAND_KERNEL"
echo "[time_series] ssml_tail_start_ratio=$SSML_TAIL_START_RATIO"
echo "[time_series] ssml_residual_beta=$SSML_RESIDUAL_BETA"
echo "[time_series] ssml_ema_decay=$SSML_EMA_DECAY"
echo "[time_series] ssml_imitation_space=$SSML_IMITATION_SPACE"
echo "[time_series] ssml_residual_space_kernel=$SSML_RESIDUAL_SPACE_KERNEL"
echo "[time_series] ssml_conflict_aware_projection=$SSML_CONFLICT_AWARE_PROJECTION"
echo "[time_series] ssml_freeze_peer=$SSML_FREEZE_PEER"
echo "[time_series] ssml_worse_only_update=$SSML_WORSE_ONLY_UPDATE"
echo "[time_series] ssml_anchor_weight=$SSML_ANCHOR_WEIGHT"
echo "[time_series] init_checkpoint_template=$INIT_CHECKPOINT_TEMPLATE"
echo "[time_series] peer_init_checkpoint_template=$PEER_INIT_CHECKPOINT_TEMPLATE"
echo "[time_series] live_plot_interval=$LIVE_PLOT_INTERVAL"

for dataset in $DATASETS; do
  for pred_len in $PRED_LENS; do
    for method in $METHODS; do
      if [[ "$method" == "independent" ]]; then
        for MODEL in $INDEPENDENT_MODELS; do
          for seed in $SEEDS; do
            init_checkpoint="$(render_checkpoint_template "$INIT_CHECKPOINT_TEMPLATE" "$dataset" "$MODEL" "" "$seed" "$pred_len")"
            cmd=(
              python -m runners.run_time_series
              --method "$method"
              --model "$MODEL"
              --dataset "$dataset"
              --epochs "$EPOCHS"
              --batch-size "$BATCH_SIZE"
              --num-workers "$NUM_WORKERS"
              --seed "$seed"
              --device "$DEVICE"
              --output-dir "$OUTPUT_DIR"
              --seq-len "$SEQ_LEN"
              --pred-len "$pred_len"
              --regression-imitation-loss "$REGRESSION_IMITATION_LOSS"
              --lambda-imitation "$LAMBDA_IMITATION"
              --margin "$MARGIN"
              --warmup-epochs "$WARMUP_EPOCHS"
              --imitation-decay-start-epoch "$IMITATION_DECAY_START_EPOCH"
              --imitation-decay-end-epoch "$IMITATION_DECAY_END_EPOCH"
              --imitation-decay-min-scale "$IMITATION_DECAY_MIN_SCALE"
              --ssml-topk-ratio "$SSML_TOPK_RATIO"
              --ssml-topk-scope "$SSML_TOPK_SCOPE"
              --ssml-max-selected-ratio "$SSML_MAX_SELECTED_RATIO"
              --ssml-adaptive-dense-threshold "$SSML_ADAPTIVE_DENSE_THRESHOLD"
              --ssml-adaptive-dense-topk-ratio "$SSML_ADAPTIVE_DENSE_TOPK_RATIO"
              --ssml-adaptive-dense-topk-scope "$SSML_ADAPTIVE_DENSE_TOPK_SCOPE"
              --ssml-adaptive-dense-max-selected-ratio "$SSML_ADAPTIVE_DENSE_MAX_SELECTED_RATIO"
              --ssml-adaptive-dense-score-smoothing-kernel "$SSML_ADAPTIVE_DENSE_SCORE_SMOOTHING_KERNEL"
              --ssml-adaptive-dense-window-expand-kernel "$SSML_ADAPTIVE_DENSE_WINDOW_EXPAND_KERNEL"
              --ssml-supervised-hotspot-alpha "$SSML_SUPERVISED_HOTSPOT_ALPHA"
              --ssml-supervised-weight-mode "$SSML_SUPERVISED_WEIGHT_MODE"
              --ssml-gate-score-mode "$SSML_GATE_SCORE_MODE"
              --ssml-score-transform "$SSML_SCORE_TRANSFORM"
              --ssml-positive-upper-quantile "$SSML_POSITIVE_UPPER_QUANTILE"
              --ssml-guidance-mode "$SSML_GUIDANCE_MODE"
              --ssml-score-smoothing-kernel "$SSML_SCORE_SMOOTHING_KERNEL"
              --ssml-window-score-kernel "$SSML_WINDOW_SCORE_KERNEL"
              --ssml-window-expand-kernel "$SSML_WINDOW_EXPAND_KERNEL"
              --ssml-tail-start-ratio "$SSML_TAIL_START_RATIO"
              --ssml-residual-beta "$SSML_RESIDUAL_BETA"
              --ssml-ema-decay "$SSML_EMA_DECAY"
              --ssml-imitation-space "$SSML_IMITATION_SPACE"
              --ssml-residual-space-kernel "$SSML_RESIDUAL_SPACE_KERNEL"
              --feature-mode "$FEATURE_MODE"
              --live-plot-interval "$LIVE_PLOT_INTERVAL"
            )

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
            if [[ "$SSML_CONFLICT_AWARE_PROJECTION" == "1" ]]; then
              cmd+=(--ssml-conflict-aware-projection)
            fi
            cmd+=(--ssml-anchor-weight "$SSML_ANCHOR_WEIGHT")
            if [[ -n "$init_checkpoint" ]]; then
              cmd+=(--init-checkpoint "$init_checkpoint")
            fi

            echo "[time_series] dataset=$dataset model=$MODEL method=$method seed=$seed pred_len=$pred_len"
            "${cmd[@]}"
          done
        done
        continue
      fi

      for pair in $MODEL_PAIRS; do
        IFS=':' read -r MODEL PEER_MODEL <<< "$pair"
        if [[ "${REQUIRE_DISTINCT_PEER}" == "1" ]] && ! pair_is_distinct "$MODEL" "${PEER_MODEL:-}"; then
          echo "[time_series][skip] pair must be heterogeneous: $pair" >&2
          continue
        fi

        for seed in $SEEDS; do
          init_checkpoint="$(render_checkpoint_template "$INIT_CHECKPOINT_TEMPLATE" "$dataset" "$MODEL" "$PEER_MODEL" "$seed" "$pred_len")"
          peer_init_checkpoint="$(render_checkpoint_template "$PEER_INIT_CHECKPOINT_TEMPLATE" "$dataset" "$PEER_MODEL" "$MODEL" "$seed" "$pred_len")"
          cmd=(
            python -m runners.run_time_series
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
            --seq-len "$SEQ_LEN"
            --pred-len "$pred_len"
            --regression-imitation-loss "$REGRESSION_IMITATION_LOSS"
            --lambda-imitation "$LAMBDA_IMITATION"
            --margin "$MARGIN"
            --warmup-epochs "$WARMUP_EPOCHS"
            --imitation-decay-start-epoch "$IMITATION_DECAY_START_EPOCH"
            --imitation-decay-end-epoch "$IMITATION_DECAY_END_EPOCH"
            --imitation-decay-min-scale "$IMITATION_DECAY_MIN_SCALE"
            --ssml-topk-ratio "$SSML_TOPK_RATIO"
            --ssml-topk-scope "$SSML_TOPK_SCOPE"
            --ssml-max-selected-ratio "$SSML_MAX_SELECTED_RATIO"
            --ssml-adaptive-dense-threshold "$SSML_ADAPTIVE_DENSE_THRESHOLD"
            --ssml-adaptive-dense-topk-ratio "$SSML_ADAPTIVE_DENSE_TOPK_RATIO"
            --ssml-adaptive-dense-topk-scope "$SSML_ADAPTIVE_DENSE_TOPK_SCOPE"
            --ssml-adaptive-dense-max-selected-ratio "$SSML_ADAPTIVE_DENSE_MAX_SELECTED_RATIO"
            --ssml-adaptive-dense-score-smoothing-kernel "$SSML_ADAPTIVE_DENSE_SCORE_SMOOTHING_KERNEL"
            --ssml-adaptive-dense-window-expand-kernel "$SSML_ADAPTIVE_DENSE_WINDOW_EXPAND_KERNEL"
            --ssml-supervised-hotspot-alpha "$SSML_SUPERVISED_HOTSPOT_ALPHA"
            --ssml-supervised-weight-mode "$SSML_SUPERVISED_WEIGHT_MODE"
            --ssml-gate-score-mode "$SSML_GATE_SCORE_MODE"
            --ssml-score-transform "$SSML_SCORE_TRANSFORM"
            --ssml-positive-upper-quantile "$SSML_POSITIVE_UPPER_QUANTILE"
            --ssml-guidance-mode "$SSML_GUIDANCE_MODE"
            --ssml-score-smoothing-kernel "$SSML_SCORE_SMOOTHING_KERNEL"
            --ssml-window-score-kernel "$SSML_WINDOW_SCORE_KERNEL"
            --ssml-window-expand-kernel "$SSML_WINDOW_EXPAND_KERNEL"
            --ssml-tail-start-ratio "$SSML_TAIL_START_RATIO"
            --ssml-residual-beta "$SSML_RESIDUAL_BETA"
            --ssml-ema-decay "$SSML_EMA_DECAY"
            --ssml-imitation-space "$SSML_IMITATION_SPACE"
            --ssml-residual-space-kernel "$SSML_RESIDUAL_SPACE_KERNEL"
            --feature-mode "$FEATURE_MODE"
            --live-plot-interval "$LIVE_PLOT_INTERVAL"
          )

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
          if [[ "$SSML_CONFLICT_AWARE_PROJECTION" == "1" ]]; then
            cmd+=(--ssml-conflict-aware-projection)
          fi
          cmd+=(--ssml-anchor-weight "$SSML_ANCHOR_WEIGHT")
          if [[ -n "$init_checkpoint" ]]; then
            cmd+=(--init-checkpoint "$init_checkpoint")
          fi
          if [[ -n "$peer_init_checkpoint" ]]; then
            cmd+=(--peer-init-checkpoint "$peer_init_checkpoint")
          fi

          echo "[time_series] dataset=$dataset pair=$MODEL:$PEER_MODEL method=$method seed=$seed pred_len=$pred_len"
          "${cmd[@]}"
        done
      done
    done
  done
done
