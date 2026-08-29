#!/usr/bin/env bash
set -euo pipefail

SEEDS=(101 202 303 404 505)

for seed in "${SEEDS[@]}"; do
  echo
  echo "======================================================================"
  echo "RUNNING RANDOM BASELINE SEED $seed"
  echo "======================================================================"
  bash "$(dirname "$0")/04_run_single_random_seed.sh" "$seed"
done

echo "ALL RANDOM BASELINE SEEDS COMPLETE"
