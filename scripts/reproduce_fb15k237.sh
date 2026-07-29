#!/usr/bin/env bash
set -euo pipefail
SEEDS=${SEEDS:-"13 21 42 87 100"}
BASE_CONFIG=${BASE_CONFIG:-configs/main_fb15k237_slmgu.yaml}
mkdir -p outputs/logs outputs/checkpoints/FB15k-237/rgcn
ORIGINAL_ROOT=${ORIGINAL_ROOT:-./checkpoints}
if [ ! -e "outputs/checkpoints/FB15k-237/rgcn/original" ]; then
  if [ -e "$ORIGINAL_ROOT/FB15k-237/rgcn/original" ]; then
    ln -s "$(pwd)/$ORIGINAL_ROOT/FB15k-237/rgcn/original" "outputs/checkpoints/FB15k-237/rgcn/original"
  else
    echo "Missing original checkpoints: $ORIGINAL_ROOT/FB15k-237/rgcn/original"
    exit 1
  fi
fi
for seed in $SEEDS; do
  echo "===== FB15k-237 / SLMGU / seed ${seed} ====="
  python main.py --config "$BASE_CONFIG" --random_seed "$seed" 2>&1 | tee "outputs/logs/reproduce_fb15k237_seed${seed}.log"
done
