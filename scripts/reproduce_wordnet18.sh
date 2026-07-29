#!/usr/bin/env bash
set -euo pipefail
SEEDS=${SEEDS:-"13 21 42 87 100"}
BASE_CONFIG=${BASE_CONFIG:-configs/main_wordnet18_slmgu.yaml}
mkdir -p outputs/logs outputs/checkpoints/WordNet18/rgcn
ORIGINAL_ROOT=${ORIGINAL_ROOT:-./checkpoints}
if [ ! -e "outputs/checkpoints/WordNet18/rgcn/original" ]; then
  if [ -e "$ORIGINAL_ROOT/WordNet18/rgcn/original" ]; then
    ln -s "$(pwd)/$ORIGINAL_ROOT/WordNet18/rgcn/original" "outputs/checkpoints/WordNet18/rgcn/original"
  else
    echo "Missing original checkpoints: $ORIGINAL_ROOT/WordNet18/rgcn/original"
    exit 1
  fi
fi
for seed in $SEEDS; do
  echo "===== WordNet18 / SLMGU / seed ${seed} ====="
  python main.py --config "$BASE_CONFIG" --random_seed "$seed" 2>&1 | tee "outputs/logs/reproduce_wordnet18_seed${seed}.log"
done
python tools/summarize_results.py --root checkpoints/WordNet18/rgcn outputs/checkpoints/WordNet18/rgcn || true
