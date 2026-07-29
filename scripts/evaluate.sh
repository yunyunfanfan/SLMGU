#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/quick_start_wordnet18_rgcn.yaml}
mkdir -p outputs/logs outputs/results
python main.py --config "$CONFIG" --mode evaluate 2>&1 | tee "outputs/logs/evaluate_$(basename "${CONFIG%.yaml}").log"
