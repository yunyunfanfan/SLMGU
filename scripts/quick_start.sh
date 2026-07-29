#!/usr/bin/env bash
set -euo pipefail
DATA_DIR=${DATA_DIR:-./datasets}
CKPT_DIR=${CKPT_DIR:-./checkpoints}
DATASET=${DATASET:-WordNet18}
GNN=${GNN:-rgcn}
SEED=${SEED:-42}
CONFIG=${CONFIG:-configs/quick_start_wordnet18_rgcn.yaml}

python - <<'PY'
import importlib
for m in ['torch','torch_geometric','yaml','ogb','sklearn']:
    importlib.import_module(m)
print('Environment check: OK')
PY

if [ ! -f "$DATA_DIR/$DATASET/d_${SEED}.pkl" ]; then
  echo "Missing dataset file: $DATA_DIR/$DATASET/d_${SEED}.pkl"
  echo "Prepare data as described in datasets/README.md, e.g.: ln -s /path/to/preprocessed_data datasets"
  exit 1
fi
if [ ! -f "$CKPT_DIR/$DATASET/$GNN/original/$SEED/model_best.pt" ]; then
  echo "Missing checkpoint: $CKPT_DIR/$DATASET/$GNN/original/$SEED/model_best.pt"
  echo "Download checkpoints as described in checkpoints/README.md or train one with scripts/train.sh."
  exit 1
fi

python main.py --config "$CONFIG" --mode evaluate --data_dir "$DATA_DIR" --checkpoint_dir "$CKPT_DIR" --dataset "$DATASET" --gnn "$GNN" --random_seed "$SEED"
