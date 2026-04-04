#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LOG_DIR="${LOG_DIR:-results/logs/time_series_etth1_peer_advantage_seeded_v1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/time_series_etth1_peer_advantage_seeded_v1}"

bash "$SCRIPT_DIR/run_time_series_etth1_peer_advantage_v1.sh"
