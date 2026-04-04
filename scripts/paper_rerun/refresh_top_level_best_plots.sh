#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

OUT_DIR="${OUT_DIR:-$ROOT_DIR}"
MIRROR_DIR="${MIRROR_DIR:-$ROOT_DIR/Results_Plots}"

mkdir -p "$MIRROR_DIR"

echo "[refresh_top_level_best_plots] out_dir=$OUT_DIR"
python scripts/paper_rerun/generate_top_level_best_plots.py

cp -f "$OUT_DIR/test_error_classification.png" "$MIRROR_DIR/classification_best.png"
cp -f "$OUT_DIR/test_error_time_series.png" "$MIRROR_DIR/time_series_best.png"
cp -f "$OUT_DIR/test_error_operator.png" "$MIRROR_DIR/operator_best.png"

echo "[refresh_top_level_best_plots] mirrored -> $MIRROR_DIR"
