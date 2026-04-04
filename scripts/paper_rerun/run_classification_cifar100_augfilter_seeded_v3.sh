#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LOG_DIR="${LOG_DIR:-results/logs/classification_cifar100_augfilter_seeded_v3}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/classification_cifar100_augfilter_seeded_v3}"

bash "$SCRIPT_DIR/run_classification_cifar100_augfilter_v1.sh"
