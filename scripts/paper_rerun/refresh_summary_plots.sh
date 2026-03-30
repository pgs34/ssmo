#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

INPUT_DIR="${1:-${INPUT_DIR:-}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-}}"
LOGFILE="${LOGFILE:-}"

if [[ -z "${INPUT_DIR:-}" || -z "${OUTPUT_DIR:-}" ]]; then
  echo "usage: $0 <input_dir> <output_dir>" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

run_refresh() {
  echo "[refresh_summary_plots] input=$INPUT_DIR output=$OUTPUT_DIR"
  python src/utils/pair_visualization.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR"
}

if [[ -n "$LOGFILE" ]]; then
  mkdir -p "$(dirname "$LOGFILE")"
  run_refresh 2>&1 | tee -a "$LOGFILE"
else
  run_refresh
fi
