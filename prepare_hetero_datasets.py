"""Prepare NELL-995, Hetionet and MovieLens-1M for this GNNDelete fork.

Outputs per dataset under ./data/<dataset>/:
  train.txt, valid.txt, test.txt          # integer triples: head\trelation\ttail
  d_<seed>.pkl                            # (dataset_info, torch_geometric.data.Data)
  df_<seed>.pt                            # {'in': mask, 'out': mask} over train_pos_edge_index
  entity2id.txt, relation2id.txt          # generated/normalized dictionaries
"""
import argparse
import gzip
import os
import pickle
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import MovieLens1M
from torch_geometric.seed import seed_everything
from torch_geometric.utils import k_hop_subgraph

from framework.utils import negative_sampling_kg

DATA_DIR = Path("./data")
RAW_DIR = Path("./dataset")
SEEDS = [42, 21, 13, 87, 100]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_triples(path: Path, triples):
    with open(path, "w") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def write_mapping(path: Path, mapping):
    with open(path, "w") as f:
        for key, val in sorted(mapping.items(), key=lambda kv: kv[1]):
            f.write(f"{key}\t{val}\n")


def split_triples(triples, seed=42, val_ratio=0.05, test_ratio=0.10):
    triples = list(dict.fromkeys(triples))  # stable de-dup, preserve order
    rng = random.Random(seed)
    rng.shuffle(triples)
    n = len(triples)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    test = triples[:n_test]
    valid = triples[n_test:n_test + n_val]
    train = triples[n_test + n_val:]
    return train, valid, test


def make_data(name, train, valid, test, num_nodes, num_relations, seeds=SEEDS):
    out_dir = DATA_DIR / name
    ensure_dir(out_dir)
    write_triples(out_dir / "train.txt", train)
    write_triples(out_dir / "valid.txt", valid)
    write_triples(out_dir / "test.txt", test)

    train_ei = torch.tensor([(h, t) for h, _, t in train], dtype=torch.long).t().contiguous()
    train_et = torch.tensor([r for _, r, _ in train], dtype=torch.long)
    val_ei = torch.tensor([(h, t) for h, _, t in valid], dtype=torch.long).t().contiguous()
    val_et = torch.tensor([r for _, r, _ in valid], dtype=torch.long)
    test_ei = torch.tensor([(h, t) for h, _, t in test], dtype=torch.long).t().contiguous()
    test_et = torch.tensor([r for _, r, _ in test], dtype=torch.long)

    data = Data(
        x=torch.arange(num_nodes),
        edge_index=train_ei,
        edge_type=train_et,
        train_pos_edge_index=train_ei,
        train_edge_type=train_et,
        val_pos_edge_index=val_ei,
        val_edge_type=val_et,
        val_neg_edge_index=negative_sampling_kg(val_ei, val_et),
        test_pos_edge_index=test_ei,
        test_edge_type=test_et,
        test_neg_edge_index=negative_sampling_kg(test_ei, test_et),
        num_nodes=num_nodes,
    )
    dataset_info = SimpleNamespace(
        name=name, num_features=0, num_nodes=num_nodes, num_edge_type=num_relations
    )

    print(f"[{name}] nodes={num_nodes:,} relations={num_relations:,} "
          f"train={len(train):,} valid={len(valid):,} test={len(test):,}")

    for s in seeds:
        seed_everything(s)
        with open(out_dir / f"d_{s}.pkl", "wb") as f:
            pickle.dump((dataset_info, data), f)

        # Candidate deletion masks: train edges within/outside 2-hop enclosing area of test nodes.
        _, local_edges, _, mask = k_hop_subgraph(
            data.test_pos_edge_index.flatten().unique(),
            2,
            data.train_pos_edge_index,
            num_nodes=num_nodes,
        )
        torch.save({"out": ~mask, "in": mask}, out_dir / f"df_{s}.pt")
        print(f"  seed={s}: df_in={int(mask.sum()):,}, df_out={int((~mask).sum()):,}")


def parse_nell_line(line):
    parts = line.strip().split("#")
    if len(parts) < 3:
        return []
    rel = parts[0]
    head = parts[1]
    triples = []
    for obj in parts[2:]:
        if ":" not in obj:
            continue
        tail, label = obj.rsplit(":", 1)
        if label == "1":
            triples.append((head, rel, tail))
    return triples


def process_nell995():
    name = "NELL-995"
    out_dir = DATA_DIR / name
    ensure_dir(out_dir)
    zip_path = RAW_DIR / "NELL-995.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path) as z:
        # Load provided entity map; relation map from archive uses concept:xxx while task files use concept_xxx.
        ent_map = {}
        for raw in z.read("NELL-995/entity2id.txt").decode("utf-8").splitlines():
            if raw.strip():
                k, v = raw.split("\t")
                ent_map[k] = int(v)

        rel_map = {}
        for raw in z.read("NELL-995/relation2id.txt").decode("utf-8").splitlines():
            if raw.strip():
                k, v = raw.split("\t")
                rel_map[k] = int(v)
                if k.startswith("concept:"):
                    rel_map["concept_" + k[len("concept:"):]] = int(v)

        triples_by_file = {}
        for fname in ["whole_train_pos_neg.txt", "whole_sort_val.txt", "whole_sort_test.txt"]:
            triples = []
            text = z.read(f"NELL-995/{fname}").decode("utf-8", "replace")
            for line in text.splitlines():
                triples.extend(parse_nell_line(line))
            triples_by_file[fname] = triples

    # Add any missing IDs defensively.
    for triples in triples_by_file.values():
        for h, r, t in triples:
            if h not in ent_map:
                ent_map[h] = len(ent_map)
            if t not in ent_map:
                ent_map[t] = len(ent_map)
            if r not in rel_map:
                rel_map[r] = len(set(rel_map.values()))

    # Compress relation IDs to contiguous IDs because the archive includes inverse/unused relations.
    used_rel_keys = sorted({r for triples in triples_by_file.values() for _, r, _ in triples})
    rel_compact = {r: i for i, r in enumerate(used_rel_keys)}

    train = [(ent_map[h], rel_compact[r], ent_map[t]) for h, r, t in triples_by_file["whole_train_pos_neg.txt"]]
    valid = [(ent_map[h], rel_compact[r], ent_map[t]) for h, r, t in triples_by_file["whole_sort_val.txt"]]
    test = [(ent_map[h], rel_compact[r], ent_map[t]) for h, r, t in triples_by_file["whole_sort_test.txt"]]

    write_mapping(out_dir / "entity2id.txt", ent_map)
    write_mapping(out_dir / "relation2id.txt", rel_compact)
    make_data(name, train, valid, test, len(ent_map), len(rel_compact))


def download_hetionet_edges(dest_gz: Path):
    urls = [
        "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz",
        "https://media.githubusercontent.com/media/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz",
    ]
    for url in urls:
        try:
            print(f"  downloading Hetionet edges from {url}")
            urllib.request.urlretrieve(url, dest_gz)
            # Git LFS pointer starts with 'version ', not gzip magic.
            with open(dest_gz, "rb") as f:
                if f.read(2) == b"\x1f\x8b":
                    return True
        except Exception as e:
            print(f"  failed: {e}")
    return False


def process_hetionet():
    name = "Hetionet"
    out_dir = DATA_DIR / name
    raw_dir = out_dir / "raw"
    ensure_dir(raw_dir)
    zip_path = RAW_DIR / "hetionet-main.zip"
    nodes_path = raw_dir / "nodes.tsv"
    edges_path = raw_dir / "edges.sif"
    edges_gz = raw_dir / "edges.sif.gz"

    if zip_path.exists() and not nodes_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            with z.open("hetionet-main/hetnet/tsv/hetionet-v1.0-nodes.tsv") as src, open(nodes_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            # Source zip may contain only a Git-LFS pointer; save it only if real gzip.
            b = z.read("hetionet-main/hetnet/tsv/hetionet-v1.0-edges.sif.gz")
            if b[:2] == b"\x1f\x8b" and not edges_gz.exists():
                edges_gz.write_bytes(b)

    if not edges_path.exists():
        if not edges_gz.exists() or edges_gz.read_bytes()[:2] != b"\x1f\x8b":
            ok = download_hetionet_edges(edges_gz)
            if not ok:
                raise RuntimeError("Hetionet edges.sif.gz 不存在且自动下载失败。")
        with gzip.open(edges_gz, "rb") as src, open(edges_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    ent_map = {}
    with open(nodes_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            node_id = line.rstrip("\n").split("\t")[0]
            ent_map.setdefault(node_id, len(ent_map))

    rel_map = {}
    triples = []
    with open(edges_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                parts = line.strip().split()
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            ent_map.setdefault(h, len(ent_map))
            ent_map.setdefault(t, len(ent_map))
            rel_map.setdefault(r, len(rel_map))
            triples.append((ent_map[h], rel_map[r], ent_map[t]))

    train, valid, test = split_triples(triples, seed=42)
    write_mapping(out_dir / "entity2id.txt", ent_map)
    write_mapping(out_dir / "relation2id.txt", rel_map)
    make_data(name, train, valid, test, len(ent_map), len(rel_map))


def process_movielens1m():
    name = "MovieLens-1M"
    out_dir = DATA_DIR / name
    ensure_dir(out_dir)
    ds = MovieLens1M(root=str(out_dir))
    hdata = ds[0]
    num_movies = hdata["movie"].num_nodes
    num_users = hdata["user"].num_nodes
    movie_offset = num_users

    edge_index = hdata[("user", "rates", "movie")].edge_index
    ratings = hdata[("user", "rates", "movie")].rating
    triples = []
    rel_map = {}
    for i in range(edge_index.size(1)):
        h = int(edge_index[0, i])
        t = movie_offset + int(edge_index[1, i])
        rel = f"rates_{int(ratings[i])}"
        rel_map.setdefault(rel, len(rel_map))
        triples.append((h, rel_map[rel], t))

    ent_map = {f"user::{i}": i for i in range(num_users)}
    ent_map.update({f"movie::{i}": movie_offset + i for i in range(num_movies)})
    train, valid, test = split_triples(triples, seed=42)
    write_mapping(out_dir / "entity2id.txt", ent_map)
    write_mapping(out_dir / "relation2id.txt", rel_map)
    make_data(name, train, valid, test, num_users + num_movies, len(rel_map))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["NELL-995", "Hetionet", "MovieLens-1M"],
                        choices=["NELL-995", "Hetionet", "MovieLens-1M"])
    args = parser.parse_args()
    for d in args.datasets:
        if d == "NELL-995":
            process_nell995()
        elif d == "Hetionet":
            process_hetionet()
        elif d == "MovieLens-1M":
            process_movielens1m()


if __name__ == "__main__":
    main()
