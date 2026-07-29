#!/usr/bin/env bash
set -euo pipefail
SEEDS=${SEEDS:-"13 21 42 87 100"}
BASE_CONFIG=${BASE_CONFIG:-configs/main_biokg_slmgu.yaml}
mkdir -p outputs/logs outputs/checkpoints/ogbl-biokg/rgcn
ORIGINAL_ROOT=${ORIGINAL_ROOT:-./checkpoints}
if [ ! -e "outputs/checkpoints/ogbl-biokg/rgcn/original" ]; then
  if [ -e "$ORIGINAL_ROOT/ogbl-biokg/rgcn/original" ]; then
    ln -s "$(pwd)/$ORIGINAL_ROOT/ogbl-biokg/rgcn/original" "outputs/checkpoints/ogbl-biokg/rgcn/original"
  else
    echo "Missing original checkpoints: $ORIGINAL_ROOT/ogbl-biokg/rgcn/original"
    exit 1
  fi
fi
for seed in $SEEDS; do
  echo "===== ogbl-biokg / SLMGU / seed ${seed} ====="
  python main.py --config "$BASE_CONFIG" --random_seed "$seed" 2>&1 | tee "outputs/logs/reproduce_biokg_seed${seed}.log"
done
