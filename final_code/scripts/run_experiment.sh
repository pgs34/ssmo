#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 1 ]]; then
  echo "usage: bash final_code/scripts/run_experiment.sh <weather|electricity|etth1|burgers|darcy|cifar10|cifar100_cifarstem>" >&2
  exit 1
fi

case "$1" in
  weather) exec bash "$SCRIPT_DIR/experiments/run_weather.sh" ;;
  electricity) exec bash "$SCRIPT_DIR/experiments/run_electricity.sh" ;;
  etth1) exec bash "$SCRIPT_DIR/experiments/run_etth1.sh" ;;
  burgers) exec bash "$SCRIPT_DIR/experiments/run_burgers.sh" ;;
  darcy) exec bash "$SCRIPT_DIR/experiments/run_darcy.sh" ;;
  cifar10) exec bash "$SCRIPT_DIR/experiments/run_cifar10.sh" ;;
  cifar100_cifarstem) exec bash "$SCRIPT_DIR/experiments/run_cifar100_cifarstem.sh" ;;
  *)
    echo "unknown experiment: $1" >&2
    exit 1
    ;;
esac
