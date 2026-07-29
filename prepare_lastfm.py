"""Convert HetRec 2011 LastFM-2K dataset to GNNDelete KG format.

Input zip: dataset/hetrec2011-lastfm-2k.zip
  user_artists.dat   -> relation 0: user -[listens]-> artist
  user_friends.dat   -> relation 1: user -[friend]->  user
  user_taggedartists.dat -> relation 2: artist -[has_tag]-> tag

Entity layout (global IDs):
  [0, N_u)               users
  [N_u, N_u+N_a)         artists
  [N_u+N_a, N_u+N_a+N_t) tags

Outputs per seed:
  data/LastFM/d_<seed>.pkl
  data/LastFM/df_<seed>.pt
"""

from __future__ import annotations

import argparse
import os
import pickle
import zipfile
from types import SimpleNamespace

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.seed import seed_everything
from torch_geometric.utils import k_hop_subgraph

from framework.utils import negative_sampling_kg

DEFAULT_SEEDS = [42, 21, 13, 87, 100]
ZIP_PATH = "dataset/hetrec2011-lastfm-2k.zip"


def read_dat(zf: zipfile.ZipFile, name: str, sep: str = "\t") -> pd.DataFrame:
    with zf.open(name) as f:
        return pd.read_csv(f, sep=sep, encoding="latin-1")


def load_lastfm_edges(zip_path: str):
    with zipfile.ZipFile(zip_path) as zf:
        ua = read_dat(zf, "user_artists.dat")   # userID, artistID, weight
        uf = read_dat(zf, "user_friends.dat")   # userID, friendID
        ut = read_dat(zf, "user_taggedartists.dat")  # userID, artistID, tagID, ...

    # Build entity ID mappings
    user_ids = sorted(set(ua["userID"]) | set(uf["userID"]) | set(uf["friendID"]))
    artist_ids = sorted(set(ua["artistID"]) | set(ut["artistID"]))
    tag_ids = sorted(set(ut["tagID"]))

    user2id = {u: i for i, u in enumerate(user_ids)}
    N_u = len(user_ids)
    artist2id = {a: N_u + i for i, a in enumerate(artist_ids)}
    N_a = len(artist_ids)
    tag2id = {t: N_u + N_a + i for i, t in enumerate(tag_ids)}
    N_t = len(tag_ids)
    num_nodes = N_u + N_a + N_t

    rows, cols, rels = [], [], []

    # Relation 0: user -> artist (listen)
    for _, row in ua.iterrows():
        u = user2id[row["userID"]]
        a = artist2id[row["artistID"]]
        rows.append(u); cols.append(a); rels.append(0)

    # Relation 1: user -> user (friend, one direction only to avoid duplication)
    seen_pairs = set()
    for _, row in uf.iterrows():
        u1 = user2id.get(row["userID"])
        u2 = user2id.get(row["friendID"])
        if u1 is None or u2 is None:
            continue
        pair = (min(u1, u2), max(u1, u2))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        rows.append(u1); cols.append(u2); rels.append(1)

    # Relation 2: artist -> tag (has_tag, deduplicated)
    seen_at = set()
    for _, row in ut.iterrows():
        a = artist2id.get(row["artistID"])
        t = tag2id.get(row["tagID"])
        if a is None or t is None:
            continue
        pair = (a, t)
        if pair in seen_at:
            continue
        seen_at.add(pair)
        rows.append(a); cols.append(t); rels.append(2)

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_type = torch.tensor(rels, dtype=torch.long)
    num_edge_type = 3

    print(f"LastFM: num_nodes={num_nodes:,}  (users={N_u}, artists={N_a}, tags={N_t})")
    print(f"  edges={edge_index.size(1):,}  "
          f"(user-artist={int((edge_type==0).sum())}, "
          f"user-user={int((edge_type==1).sum())}, "
          f"artist-tag={int((edge_type==2).sum())})")

    return num_nodes, num_edge_type, edge_index, edge_type


def split_edges(edge_index, edge_type, val_ratio: float, test_ratio: float):
    n = edge_index.size(1)
    perm = torch.randperm(n)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_idx = perm[:n_test]
    val_idx = perm[n_test: n_test + n_val]
    train_idx = perm[n_test + n_val:]

    data = Data()
    data.train_pos_edge_index = edge_index[:, train_idx].contiguous()
    data.train_edge_type = edge_type[train_idx].contiguous()
    data.val_pos_edge_index = edge_index[:, val_idx].contiguous()
    data.val_edge_type = edge_type[val_idx].contiguous()
    data.test_pos_edge_index = edge_index[:, test_idx].contiguous()
    data.test_edge_type = edge_type[test_idx].contiguous()

    data.val_neg_edge_index = negative_sampling_kg(
        data.val_pos_edge_index, data.val_edge_type).contiguous()
    data.test_neg_edge_index = negative_sampling_kg(
        data.test_pos_edge_index, data.test_edge_type).contiguous()
    return data


def build_df_masks(data: Data, num_nodes: int):
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
    parser.add_argument("--zip_path", default=ZIP_PATH)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, "LastFM")
    os.makedirs(out_dir, exist_ok=True)

    num_nodes, num_edge_type, edge_index, edge_type = load_lastfm_edges(args.zip_path)

    dataset_stub = SimpleNamespace(
        name="LastFM",
        num_features=0,
        num_nodes=num_nodes,
        num_edge_type=num_edge_type,
    )

    for seed in args.seeds:
        seed_everything(seed)
        data = split_edges(edge_index, edge_type, args.val_ratio, args.test_ratio)
        data.num_nodes = num_nodes
        data.x = torch.arange(num_nodes, dtype=torch.long)
        data.edge_index = data.train_pos_edge_index.clone()
        data.edge_type = data.train_edge_type.clone()

        print(f"Seed {seed}: train={data.train_pos_edge_index.size(1):,}, "
              f"val={data.val_pos_edge_index.size(1):,}, "
              f"test={data.test_pos_edge_index.size(1):,}")

        with open(os.path.join(out_dir, f"d_{seed}.pkl"), "wb") as f:
            pickle.dump((dataset_stub, data), f)

        df_masks = build_df_masks(data, num_nodes)
        torch.save(df_masks, os.path.join(out_dir, f"df_{seed}.pt"))
        print(f"  df_in={int(df_masks['in'].sum()):,}, "
              f"df_out={int(df_masks['out'].sum()):,}")

    print(f"\nDone. Files written to {out_dir}/")


if __name__ == "__main__":
    main()
