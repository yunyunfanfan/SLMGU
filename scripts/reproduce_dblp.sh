#!/usr/bin/env bash
set -euo pipefail
SEEDS=${SEEDS:-"13 21 42 87 100"}
BASE_CONFIG=${BASE_CONFIG:-configs/main_dblp_slmgu.yaml}
mkdir -p outputs/logs outputs/checkpoints/DBLP/rgcn
ORIGINAL_ROOT=${ORIGINAL_ROOT:-./checkpoints}
if [ ! -e "outputs/checkpoints/DBLP/rgcn/original" ]; then
  if [ -e "$ORIGINAL_ROOT/DBLP/rgcn/original" ]; then
    ln -s "$(pwd)/$ORIGINAL_ROOT/DBLP/rgcn/original" "outputs/checkpoints/DBLP/rgcn/original"
  else
    echo "Missing original checkpoints: $ORIGINAL_ROOT/DBLP/rgcn/original"
    exit 1
  fi
fi
for seed in $SEEDS; do
  echo "===== DBLP / SLMGU / seed ${seed} ====="
  python main.py --config "$BASE_CONFIG" --random_seed "$seed" 2>&1 | tee "outputs/logs/reproduce_dblp_seed${seed}.log"
done
