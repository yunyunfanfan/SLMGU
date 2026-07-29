#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/train_wordnet18_rgcn.yaml}
mkdir -p outputs/logs outputs/checkpoints outputs/results
python main.py --config "$CONFIG" 2>&1 | tee "outputs/logs/train_$(basename "${CONFIG%.yaml}").log"
