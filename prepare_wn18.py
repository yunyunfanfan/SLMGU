import argparse
import os
import pickle

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import WordNet18, WordNet18RR
from torch_geometric.seed import seed_everything
from torch_geometric.utils import k_hop_subgraph

from framework.utils import negative_sampling_kg


DATASET_MAPPING = {
    "WordNet18": WordNet18,
    "WordNet18RR": WordNet18RR,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and preprocess WordNet18/WordNet18RR for this GNNDelete snapshot."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_MAPPING),
        default="WordNet18",
        help="Knowledge graph dataset to prepare.",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory where the dataset and processed files will be stored.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds for d_<seed>.pkl and df_<seed>.pt generation.",
    )
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="Ask PyG to re-download/re-process the raw dataset.",
    )
    parser.add_argument(
        "--skip_txt_export",
        action="store_true",
        help="Only write d_<seed>.pkl/df_<seed>.pt, not train.txt/valid.txt/test.txt.",
    )
    return parser.parse_args()


def build_project_data(data):
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    train_pos_edge_index = data.edge_index[:, train_mask]
    train_edge_type = data.edge_type[train_mask]

    val_pos_edge_index = data.edge_index[:, val_mask]
    val_edge_type = data.edge_type[val_mask]
    val_neg_edge_index = negative_sampling_kg(val_pos_edge_index, val_edge_type)

    test_pos_edge_index = data.edge_index[:, test_mask]
    test_edge_type = data.edge_type[test_mask]
    test_neg_edge_index = negative_sampling_kg(test_pos_edge_index, test_edge_type)

    row, col = train_pos_edge_index
    rev_edge_index = torch.stack([col, row], dim=0)
    num_edge_type = int(data.edge_type.max().item()) + 1
    rev_edge_type = train_edge_type + num_edge_type

    edge_index = torch.cat([train_pos_edge_index, rev_edge_index], dim=1)
    edge_type = torch.cat([train_edge_type, rev_edge_type], dim=0)

    return Data(
        x=torch.arange(data.num_nodes),
        edge_index=edge_index,
        edge_type=edge_type,
        train_pos_edge_index=train_pos_edge_index,
        train_edge_type=train_edge_type,
        val_pos_edge_index=val_pos_edge_index,
        val_edge_type=val_edge_type,
        val_neg_edge_index=val_neg_edge_index,
        test_pos_edge_index=test_pos_edge_index,
        test_edge_type=test_edge_type,
        test_neg_edge_index=test_neg_edge_index,
        num_nodes=data.num_nodes,
    )


def write_split_txt(path, edge_index, edge_type):
    with open(path, "w") as f:
        for head, relation, tail in zip(
            edge_index[0].tolist(), edge_type.tolist(), edge_index[1].tolist()
        ):
            f.write(f"{head}\t{relation}\t{tail}\n")


def export_txt_splits(output_dir, data):
    write_split_txt(
        os.path.join(output_dir, "train.txt"),
        data.train_pos_edge_index,
        data.train_edge_type,
    )
    write_split_txt(
        os.path.join(output_dir, "valid.txt"),
        data.val_pos_edge_index,
        data.val_edge_type,
    )
    write_split_txt(
        os.path.join(output_dir, "test.txt"),
        data.test_pos_edge_index,
        data.test_edge_type,
    )


def save_delete_masks(output_dir, data, seed):
    _, local_edges, _, in_mask = k_hop_subgraph(
        data.test_pos_edge_index.flatten().unique(),
        2,
        data.train_pos_edge_index,
        num_nodes=data.num_nodes,
    )
    out_mask = ~in_mask

    torch.save(
        {"out": out_mask, "in": in_mask},
        os.path.join(output_dir, f"df_{seed}.pt"),
    )

    print(
        "Delete candidates:",
        f"in={int(in_mask.sum()):,}",
        f"out={int(out_mask.sum()):,}",
        f"local_edges={local_edges.shape[1]:,}",
    )


def main():
    args = parse_args()
    dataset_cls = DATASET_MAPPING[args.dataset]
    output_dir = os.path.join(args.data_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {args.dataset} under {output_dir}")
    dataset = dataset_cls(output_dir, force_reload=args.force_reload)
    raw_data = dataset[0]
    data = build_project_data(raw_data)

    print(
        f"{args.dataset}:",
        f"nodes={data.num_nodes:,}",
        f"train={data.train_pos_edge_index.shape[1]:,}",
        f"valid={data.val_pos_edge_index.shape[1]:,}",
        f"test={data.test_pos_edge_index.shape[1]:,}",
        f"relations={int(raw_data.edge_type.max().item()) + 1}",
    )

    if not args.skip_txt_export:
        export_txt_splits(output_dir, data)
        print("Wrote train.txt, valid.txt, test.txt")

    for seed in args.seeds:
        seed_everything(seed)
        with open(os.path.join(output_dir, f"d_{seed}.pkl"), "wb") as f:
            pickle.dump((dataset, data), f)
        save_delete_masks(output_dir, data, seed)
        print(f"Wrote d_{seed}.pkl and df_{seed}.pt")


if __name__ == "__main__":
    main()
