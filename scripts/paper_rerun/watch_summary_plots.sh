#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
activate_paper_env
cd "$ROOT_DIR"

INPUT_DIR="${1:-${INPUT_DIR:-}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-}}"
INTERVAL_SEC="${INTERVAL_SEC:-300}"
LOGFILE="${LOGFILE:-}"
LABEL="${LABEL:-summary_plot_watch}"

if [[ -z "${INPUT_DIR:-}" || -z "${OUTPUT_DIR:-}" ]]; then
  echo "usage: $0 <input_dir> <output_dir>" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
if [[ -n "$LOGFILE" ]]; then
  mkdir -p "$(dirname "$LOGFILE")"
fi

run_once() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$LABEL] $ts refresh input=$INPUT_DIR output=$OUTPUT_DIR"
  python src/utils/pair_visualization.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR"
}

while true; do
  if [[ -n "$LOGFILE" ]]; then
    run_once >>"$LOGFILE" 2>&1 || true
  else
    run_once || true
  fi
  sleep "$INTERVAL_SEC"
done
