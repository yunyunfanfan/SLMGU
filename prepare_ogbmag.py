"""Convert OGB-MAG pre-processed .pt to GNNDelete KG format.

Source: data/OGB-MAG/mag/processed/geometric_data_processed.pt

Entity layout (global IDs, sorted by size ascending for locality):
  [0,           N_inst)                    institution
  [N_inst,      N_inst+N_fos)              field_of_study
  [N_inst+N_fos, N_inst+N_fos+N_paper)    paper
  [N_inst+N_fos+N_paper, ...)              author

Relation types:
  0: author -> institution  (affiliated_with)
  1: author -> paper        (writes)
  2: paper  -> paper        (cites)
  3: paper  -> field_of_study (has_topic)

Outputs per seed:
  data/OGB-MAG/d_<seed>.pkl
  data/OGB-MAG/df_<seed>.pt
"""

from __future__ import annotations

import argparse
import os
import pickle
from types import SimpleNamespace

import torch
from torch_geometric.data import Data
from torch_geometric.seed import seed_everything
from torch_geometric.utils import k_hop_subgraph

from framework.utils import negative_sampling_kg

DEFAULT_SEEDS = [42, 21, 13, 87, 100]
PT_PATH = "data/OGB-MAG/mag/processed/geometric_data_processed.pt"


def load_ogbmag_edges(pt_path: str):
    print("Loading pre-processed OGB-MAG ...")
    raw = torch.load(pt_path, weights_only=False)
    obj = raw[0]

    num_nodes_dict = obj.num_nodes_dict[0]
    edge_index_dict = obj.edge_index_dict[0]

    N_inst  = num_nodes_dict['institution']
    N_fos   = num_nodes_dict['field_of_study']
    N_paper = num_nodes_dict['paper']
    N_auth  = num_nodes_dict['author']
    num_nodes = N_inst + N_fos + N_paper + N_auth

    # Global ID offsets
    off = {
        'institution':    0,
        'field_of_study': N_inst,
        'paper':          N_inst + N_fos,
        'author':         N_inst + N_fos + N_paper,
    }

    rel_map = {
        ('author', 'affiliated_with', 'institution'): 0,
        ('author', 'writes',          'paper'):        1,
        ('paper',  'cites',           'paper'):        2,
        ('paper',  'has_topic',       'field_of_study'): 3,
    }

    all_rows, all_cols, all_rels = [], [], []
    for key, ei in edge_index_dict.items():
        src_type, _, dst_type = key
        rel_id = rel_map[key]
        src_off = off[src_type]
        dst_off = off[dst_type]
        all_rows.append(ei[0] + src_off)
        all_cols.append(ei[1] + dst_off)
        all_rels.append(torch.full((ei.size(1),), rel_id, dtype=torch.long))
        print(f"  {key[0]}-{key[1]}-{key[2]}: {ei.size(1):,} edges")

    edge_index = torch.stack([torch.cat(all_rows), torch.cat(all_cols)])
    edge_type  = torch.cat(all_rels)
    num_edge_type = 4

    print(f"OGB-MAG: num_nodes={num_nodes:,}, total_edges={edge_index.size(1):,}")
    return num_nodes, num_edge_type, edge_index, edge_type


def split_edges(edge_index, edge_type, val_ratio, test_ratio):
    n = edge_index.size(1)
    perm = torch.randperm(n)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)

    test_idx  = perm[:n_test]
    val_idx   = perm[n_test: n_test + n_val]
    train_idx = perm[n_test + n_val:]

    data = Data()
    data.train_pos_edge_index = edge_index[:, train_idx].contiguous()
    data.train_edge_type      = edge_type[train_idx].contiguous()
    data.val_pos_edge_index   = edge_index[:, val_idx].contiguous()
    data.val_edge_type        = edge_type[val_idx].contiguous()
    data.test_pos_edge_index  = edge_index[:, test_idx].contiguous()
    data.test_edge_type       = edge_type[test_idx].contiguous()

    data.val_neg_edge_index  = negative_sampling_kg(
        data.val_pos_edge_index, data.val_edge_type).contiguous()
    data.test_neg_edge_index = negative_sampling_kg(
        data.test_pos_edge_index, data.test_edge_type).contiguous()
    return data


def build_df_masks(data, num_nodes):
    _, _, _, in_mask = k_hop_subgraph(
        data.test_pos_edge_index.flatten().unique(),
        2,
        data.train_pos_edge_index,
        num_nodes=num_nodes,
    )
    in_mask = in_mask.bool().cpu()
    return {"out": ~in_mask, "in": in_mask}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path",   default=PT_PATH)
    parser.add_argument("--data_dir",  default="./data")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, "OGB-MAG")
    os.makedirs(out_dir, exist_ok=True)

    num_nodes, num_edge_type, edge_index, edge_type = load_ogbmag_edges(args.pt_path)

    dataset_stub = SimpleNamespace(
        name="OGB-MAG",
        num_features=0,
        num_nodes=num_nodes,
        num_edge_type=num_edge_type,
    )

    for seed in args.seeds:
        seed_everything(seed)
        data = split_edges(edge_index, edge_type, args.val_ratio, args.test_ratio)
        data.num_nodes  = num_nodes
        data.x          = torch.arange(num_nodes, dtype=torch.long)
        data.edge_index = data.train_pos_edge_index.clone()
        data.edge_type  = data.train_edge_type.clone()

        print(f"Seed {seed}: train={data.train_pos_edge_index.size(1):,}, "
              f"val={data.val_pos_edge_index.size(1):,}, "
              f"test={data.test_pos_edge_index.size(1):,}")

        with open(os.path.join(out_dir, f"d_{seed}.pkl"), "wb") as f:
            pickle.dump((dataset_stub, data), f)

        print(f"  Building df masks (2-hop subgraph, may take a while)...")
        df_masks = build_df_masks(data, num_nodes)
        torch.save(df_masks, os.path.join(out_dir, f"df_{seed}.pt"))
        print(f"  df_in={int(df_masks['in'].sum()):,}, "
              f"df_out={int(df_masks['out'].sum()):,}")

    print(f"\nDone. Files written to {out_dir}/")


if __name__ == "__main__":
    main()
