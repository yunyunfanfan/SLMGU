# Dataset Preparation

Datasets are **not** committed to this repository. Put preprocessed files under `datasets/<DATASET>/` or symlink your local data directory.

Required preprocessed files per seed:

```text
datasets/<DATASET>/
├── d_13.pkl
├── d_21.pkl
├── d_42.pkl
├── d_87.pkl
├── d_100.pkl
├── df_13.pt
├── df_21.pt
├── df_42.pt
├── df_87.pt
└── df_100.pt
```

Raw datasets used in the paper:

- WordNet18 / WN18: https://paperswithcode.com/dataset/wn18
- FB15k-237: https://www.microsoft.com/en-us/download/details.aspx?id=52312
- WN18RR: https://github.com/TimDettmers/ConvE
- YAGO3-10: https://github.com/TimDettmers/ConvE
- OGB BioKG: https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg
- DBLP heterogeneous graph: see `prepare_dblp_rgcn.py` and raw DBLP releases referenced by your paper.

Example:

```bash
ln -s /path/to/preprocessed_data datasets
# or
mkdir -p datasets/WordNet18
cp /path/to/WordNet18/d_42.pkl datasets/WordNet18/
```

Preprocessing scripts are available at repository root: `prepare_wn18.py`, `prepare_dataset.py`, `prepare_dblp_rgcn.py`, `prepare_hetero_datasets.py`, and related helpers.
