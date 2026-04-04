#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
MODEL_PAIRS="${MODEL_PAIRS:-transformer:dlinear}"
INDEPENDENT_MODELS="${INDEPENDENT_MODELS:-transformer}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEQ_LEN="${SEQ_LEN:-96}"
PRED_LENS="${PRED_LENS:-24}"
FEATURE_MODE="${FEATURE_MODE:-multivariate}"
REGRESSION_IMITATION_LOSS="${REGRESSION_IMITATION_LOSS:-huber}"
LIVE_PLOT_INTERVAL="${LIVE_PLOT_INTERVAL:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_all_methods_long_v2}"
SUMMARY_PLOT_ROOT="${SUMMARY_PLOT_ROOT:-results/plots/time_series_etth1_all_methods_long_v2}"
REFRESH_TOP_LEVEL="${REFRESH_TOP_LEVEL:-1}"

echo "[time_series_etth1_all_methods_long_v2] output_root=$OUTPUT_ROOT"
echo "[time_series_etth1_all_methods_long_v2] summary_plot_root=$SUMMARY_PLOT_ROOT"
echo "[time_series_etth1_all_methods_long_v2] gpu=$GPU seeds=$SEEDS epochs=$EPOCHS"
echo "[time_series_etth1_all_methods_long_v2] model_pairs=$MODEL_PAIRS independent_models=$INDEPENDENT_MODELS"
echo "[time_series_etth1_all_methods_long_v2] methods=independent dml ssml"

CUDA_VISIBLE_DEVICES="$GPU" \
DEVICE="$DEVICE" \
DATASETS="etth1" \
METHODS="independent dml ssml" \
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
OUTPUT_DIR="$OUTPUT_ROOT" \
LAMBDA_IMITATION="${LAMBDA_IMITATION:-0.001}" \
MARGIN="${MARGIN:-0.02}" \
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}" \
IMITATION_DECAY_START_EPOCH="${IMITATION_DECAY_START_EPOCH:-10}" \
IMITATION_DECAY_END_EPOCH="${IMITATION_DECAY_END_EPOCH:-50}" \
IMITATION_DECAY_MIN_SCALE="${IMITATION_DECAY_MIN_SCALE:-0.1}" \
SSML_TOPK_RATIO="${SSML_TOPK_RATIO:-0.02}" \
SSML_SUPERVISED_HOTSPOT_ALPHA="${SSML_SUPERVISED_HOTSPOT_ALPHA:-0.5}" \
SSML_GATE_SCORE_MODE="${SSML_GATE_SCORE_MODE:-peer_better_student_error}" \
SSML_SCORE_TRANSFORM="${SSML_SCORE_TRANSFORM:-none}" \
SSML_GUIDANCE_MODE="${SSML_GUIDANCE_MODE:-reweight_only}" \
HETERO_SSML_ONE_WAY="${HETERO_SSML_ONE_WAY:-1}" \
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}" \
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-0}" \
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}" \
bash scripts/paper_rerun/run_core_time_series.sh

echo "[time_series_etth1_all_methods_long_v2] refreshing summary plots"
bash scripts/paper_rerun/refresh_summary_plots.sh "$OUTPUT_ROOT" "$SUMMARY_PLOT_ROOT"

if [[ "$REFRESH_TOP_LEVEL" == "1" ]]; then
  echo "[time_series_etth1_all_methods_long_v2] refreshing top-level plots"
  bash scripts/paper_rerun/refresh_top_level_best_plots.sh
fi

echo "[time_series_etth1_all_methods_long_v2] done"
