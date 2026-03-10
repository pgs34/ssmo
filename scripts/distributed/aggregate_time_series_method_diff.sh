#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-results/time_series_method_diff}"

python -m src.utils.visualization \
  --input-dir "$OUTPUT_DIR/time_series" \
  --output-dir "$OUTPUT_DIR"
