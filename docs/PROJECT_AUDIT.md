# Project Audit

## Current Structure Analysis

The original repository is research-code oriented. Core runnable files are at repository root, while model/trainer implementations live under `framework/`.

- `train_gnn.py`: original KG/link-prediction training entry. Loads `data_dir/dataset/d_<seed>.pkl`, builds reverse relations for KG backbones, trains with `KGTrainer`, saves `model_best.pt` and `trainer_log.json`.
- `delete_gnn.py`: unlearning/deletion entry. Loads original checkpoint, constructs deletion (`Df`) and retained (`Dr`) masks from `df_<seed>.pt`, trains deletion operators, evaluates DT/DF metrics.
- `train_node.py`, `delete_node.py`, `delete_node_feature.py`: node-level tasks.
- `framework/models/`: GCN/GAT/GIN, RGCN/RGAT, HGT/HAN, deletion layers, and the added `compgcn.py` backbone.
- `framework/trainer/`: base trainers, KG trainer, retrain, gradient ascent, GNNDelete variants, graph eraser, MI attack support.
- `framework/utils.py`: negative sampling, labels, deletion/loss helper functions.
- `framework/training_args.py`: centralized CLI defaults and dataset relation counts.
- `prepare_*.py`: dataset preprocessing scripts.
- `plot_*.py`, `benchmark_*`, `*_SUMMARY.md`: analysis artifacts from local experiments.

## Core Components

| Functionality | Files |
|---|---|
| Model implementation | `framework/models/*.py` |
| Data loading | `train_gnn.py`, `delete_gnn.py`, `prepare_*.py` |
| Training pipeline | `framework/trainer/base.py`, `framework/trainer/gnndelete*.py` |
| Testing/evaluation | `evaluate_gnn.py`, `framework/trainer/base.py`, `framework/evaluation.py` |
| Loss functions | `framework/trainer/gnndelete*.py`, `framework/utils.py` |
| Configuration | `framework/training_args.py`, `configs/*.yaml` |
| Experiment scripts | `scripts/*.sh` |

## Current Running Flow

### Training

1. Parse CLI/YAML arguments.
2. Load `d_<seed>.pkl` from `data_dir/dataset`.
3. Build undirected message-passing graph by adding reverse edges for KG backbones.
4. Instantiate model through `framework.get_model`.
5. Train via `KGTrainer.train`.
6. Save checkpoints and JSON logs.

### Testing

1. Load the same preprocessed split.
2. Resolve checkpoint directory.
3. Instantiate model and load `model_best.pt`.
4. Run `KGTrainer.test` and save metrics JSON.

### Unlearning

1. Load original checkpoint from `checkpoint_dir/<DATASET>/<GNN>/original/<SEED>/`.
2. Load deletion masks from `df_<seed>.pt`.
3. Construct `Df`, `Dr`, and local subgraph masks.
4. Train selected deletion operator.
5. Evaluate DT and DF metrics.

## Problems Found and Fixes Applied

- **Scattered entry points**: added `main.py` as a YAML-driven unified entry.
- **No public installation recipe**: added `setup.sh`, `requirements.txt`, and `environment.yml`.
- **No public dataset/checkpoint documentation**: added `datasets/README.md` and `checkpoints/README.md`.
- **Runtime artifacts mixed with code**: updated `.gitignore`; outputs go under `outputs/`.
- **No quick-start reproducibility path**: added `scripts/quick_start.sh`, `scripts/evaluate.sh`, and dataset-specific reproduction scripts.
- **CompGCN backbone requested**: added `framework/models/compgcn.py` and registered `compgcn` in model/trainer selection.
- **Parameters hard to reproduce**: added YAML configs under `configs/` for quick start, training, and main experiments.

## Notes for Public Release

Before pushing to GitHub, do not commit large files already ignored locally: `data/`, `dataset/`, `checkpoint*/`, `wandb/`, archives, PDFs, logs, and generated plots unless intentionally curated. Keep only small figures needed by README, such as `figures/framework.png`.
