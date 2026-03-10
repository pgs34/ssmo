#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

bash scripts/distributed/run_time_series_block.sh 0 "etth1" "transformer:transformer"
bash scripts/distributed/run_time_series_block.sh 0 "weather" "dlinear:dlinear"
