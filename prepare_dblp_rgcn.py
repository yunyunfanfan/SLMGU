"""Prepare the heterogeneous DBLP .dat dataset for RGCN experiments.

This script converts the DBLP.zip layout used by many heterogeneous graph
benchmarks::

    DBLP/node.dat
    DBLP/link.dat
    DBLP/label.dat
    DBLP/info.dat
    DBLP/meta.dat

into the KG-style PyG ``Data`` objects expected by this GNNDelete snapshot's
RGCN/RGAT pipeline.  It writes, for each seed:

    data/DBLP/d_<seed>.pkl
    data/DBLP/df_<seed>.pt

The generated Data object contains:

    x                       LongTensor [num_nodes], original node IDs
    train_pos_edge_index    LongTensor [2, num_train_edges]
    train_edge_type         LongTensor [num_train_edges]
    val_pos_edge_index      LongTensor [2, num_val_edges]
    val_edge_type           LongTensor [num_val_edges]
    val_neg_edge_index      LongTensor [2, num_val_edges]
    test_pos_edge_index     LongTensor [2, num_test_edges]
    test_edge_type          LongTensor [num_test_edges]
    test_neg_edge_index     LongTensor [2, num_test_edges]

Notes:
- ``link.dat`` already contains six directed relation types:
  author-paper, paper-term, paper-venue, paper-author, term-paper, venue-paper.
- ``train_gnn.py`` will add another reverse copy for message passing with
  relation IDs offset by ``args.num_edge_type``.  Therefore ``train_edge_type``
  must remain in [0, 5].
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import zipfile
from types import SimpleNamespace

import torch
from torch_geometric.data import Data
from torch_geometric.seed import seed_everything
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

from framework.utils import negative_sampling_kg


DEFAULT_SEEDS = [42, 21, 13, 87, 100]


def _read_text_from_zip(zip_path: str, member: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(member).decode("utf-8", errors="replace")


def load_dblp_edges(zip_path: str):
    meta = json.loads(_read_text_from_zip(zip_path, "DBLP/meta.dat"))
    info = json.loads(_read_text_from_zip(zip_path, "DBLP/info.dat"))
    num_nodes = int(meta["Node Total"])
    num_edge_type = len(info["link.dat"]["link type"])

    rows = []
    cols = []
    rels = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("DBLP/link.dat") as f:
            for raw in tqdm(f, desc="Reading DBLP/link.dat"):
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    raise ValueError(f"Bad link.dat row: {line!r}")
                src, dst, rel = int(parts[0]), int(parts[1]), int(parts[2])
                rows.append(src)
                cols.append(dst)
                rels.append(rel)

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_type = torch.tensor(rels, dtype=torch.long)

    if edge_index.numel() == 0:
        raise RuntimeError("No edges were read from DBLP/link.dat")
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
        raise RuntimeError("Edge endpoint outside node ID range")
    if int(edge_type.min()) < 0 or int(edge_type.max()) >= num_edge_type:
        raise RuntimeError("Relation ID outside edge type range")

    return num_nodes, num_edge_type, edge_index, edge_type, info, meta


def split_edges(edge_index, edge_type, val_ratio: float, test_ratio: float):
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)

    n_val = int(num_edges * val_ratio)
    n_test = int(num_edges * test_ratio)
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    data = Data()
    data.train_pos_edge_index = edge_index[:, train_idx].contiguous()
    data.train_edge_type = edge_type[train_idx].contiguous()
    data.val_pos_edge_index = edge_index[:, val_idx].contiguous()
    data.val_edge_type = edge_type[val_idx].contiguous()
    data.test_pos_edge_index = edge_index[:, test_idx].contiguous()
    data.test_edge_type = edge_type[test_idx].contiguous()

    data.val_neg_edge_index = negative_sampling_kg(
        data.val_pos_edge_index, data.val_edge_type
    ).contiguous()
    data.test_neg_edge_index = negative_sampling_kg(
        data.test_pos_edge_index, data.test_edge_type
    ).contiguous()
    return data


def build_df_masks(data: Data, num_nodes: int):
    # Same idea as prepare_dataset.py: "in" means train edges in the 2-hop
    # subgraph around test endpoints; "out" means all other train edges.
    _, _, _, in_mask = k_hop_subgraph(
        data.test_pos_edge_index.flatten().unique(),
        2,
        data.train_pos_edge_index,
        num_nodes=num_nodes,
    )
    in_mask = in_mask.bool().cpu()
    out_mask = ~in_mask
    return {"out": out_mask, "in": in_mask}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", default="dataset/DBLP.zip")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--dataset", default="DBLP")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument(
        "--skip_df_masks",
        action="store_true",
        help="Write df masks as all-train out / empty in; useful for quick debugging.",
    )
    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    num_nodes, num_edge_type, edge_index, edge_type, info, meta = load_dblp_edges(
        args.zip_path
    )
    print(
        f"DBLP loaded: num_nodes={num_nodes:,}, "
        f"num_edges={edge_index.size(1):,}, num_edge_type={num_edge_type}"
    )

    dataset_stub = SimpleNamespace(
        name=args.dataset,
        num_nodes=num_nodes,
        num_edge_type=num_edge_type,
        info=info,
        meta=meta,
    )

    for seed in args.seeds:
        seed_everything(seed)
        data = split_edges(edge_index, edge_type, args.val_ratio, args.test_ratio)
        data.num_nodes = num_nodes
        data.x = torch.arange(num_nodes, dtype=torch.long)
        # Convenience fields used before train_gnn.py replaces edge_index/edge_type
        # with train + reverse-train edges.
        data.edge_index = data.train_pos_edge_index.clone()
        data.edge_type = data.train_edge_type.clone()

        print(
            f"Seed {seed}: train={data.train_pos_edge_index.size(1):,}, "
            f"val={data.val_pos_edge_index.size(1):,}, "
            f"test={data.test_pos_edge_index.size(1):,}"
        )

        with open(os.path.join(out_dir, f"d_{seed}.pkl"), "wb") as f:
            pickle.dump((dataset_stub, data), f)

        if args.skip_df_masks:
            df_masks = {
                "out": torch.ones(data.train_pos_edge_index.size(1), dtype=torch.bool),
                "in": torch.zeros(data.train_pos_edge_index.size(1), dtype=torch.bool),
            }
        else:
            df_masks = build_df_masks(data, num_nodes)

        torch.save(df_masks, os.path.join(out_dir, f"df_{seed}.pt"))
        print(
            f"Seed {seed}: df_in={int(df_masks['in'].sum()):,}, "
            f"df_out={int(df_masks['out'].sum()):,}"
        )

    print(f"Done. Files written under {out_dir}")


if __name__ == "__main__":
    main()
