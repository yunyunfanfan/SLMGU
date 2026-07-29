"""
Membership Inference (MI) Attack training script.

Trains two attack models against the *original* GNN:
  - member_infer_all : all training edges as "present"
  - member_infer_sub : Df edges only as "present"

Saved to:
  checkpoint/{dataset}/member_infer_all/{seed}/attack_model_best.pt
  checkpoint/{dataset}/member_infer_sub/{seed}/attack_model_best.pt

Usage:
  python train_mi.py --dataset WordNet18 --gnn rgcn --random_seed 42
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.seed import seed_everything
from torch_geometric.utils import to_undirected, k_hop_subgraph
from sklearn.metrics import roc_auc_score

from framework import get_model
from framework.training_args import parse_args


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLPAttacker(nn.Module):
    """2-class MLP attack model. Input: [P(absent), P(present)] per edge."""

    def __init__(self, args=None, input_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _edge_scores(model, edge_index, edge_type, context_edge_index, context_edge_type):
    """Return sigmoid scores for each edge using the given message-passing context."""
    dev = next(model.parameters()).device
    ei = edge_index.to(dev)

    if context_edge_type is not None:
        z = model(model.node_emb.weight.to(dev) if hasattr(model, 'node_emb') else None,
                  context_edge_index.to(dev), context_edge_type.to(dev))
        # Use actual node embeddings via standard forward
        # node_emb input for RGCN is the index tensor
        x = torch.arange(z.shape[0], device=dev)
        z = model(x, context_edge_index.to(dev), context_edge_type.to(dev))
        et = edge_type.to(dev)
        score = model.decode(z, ei, et).sigmoid()
    else:
        z = model(model._cached_x, context_edge_index.to(dev))
        score = model.decode(z, ei).sigmoid()

    return score.cpu()


@torch.no_grad()
def _edge_scores_simple(model, edge_index, edge_type, context_edge_index, context_edge_type, data):
    """Return sigmoid scores. Handles both GNN and KG models."""
    dev = next(model.parameters()).device
    ei = edge_index.to(dev)

    if context_edge_type is not None:
        x = data.x.to(dev)
        z = model(x, context_edge_index.to(dev), context_edge_type.to(dev))
        et = edge_type.to(dev)
        score = model.decode(z, ei, et).sigmoid()
    else:
        x = data.x.to(dev)
        z = model(x, context_edge_index.to(dev))
        score = model.decode(z, ei).sigmoid()

    return score.cpu()


def _build_attack_data(model, data, mode, is_kg):
    """
    Build (features, labels) for attack model training.

    Present (label=1): training edges the model was trained on.
    Absent  (label=0): validation edges the model never saw.

    mode='all' -> present = all dr edges
    mode='sub' -> present = df edges only
    """
    model.eval()

    if is_kg:
        directed_dr_mask = data.directed_dr_mask
        directed_df_mask = data.directed_df_mask
        ctx_ei = data.edge_index[:, data.dr_mask]
        ctx_et = data.edge_type[data.dr_mask]
        if mode == 'all':
            present_ei = data.train_pos_edge_index[:, directed_dr_mask]
            present_et = data.train_edge_type[directed_dr_mask]
        else:
            present_ei = data.train_pos_edge_index[:, directed_df_mask]
            present_et = data.train_edge_type[directed_df_mask]
        absent_ei = data.val_pos_edge_index
        absent_et = data.val_edge_type
    else:
        ctx_ei = data.train_pos_edge_index[:, data.dr_mask]
        ctx_et = None
        if mode == 'all':
            present_ei = data.train_pos_edge_index[:, data.dr_mask]
        else:
            present_ei = data.train_pos_edge_index[:, data.df_mask]
        present_et = None
        absent_ei = data.val_pos_edge_index
        absent_et = None

    # Balance
    n = min(present_ei.shape[1], absent_ei.shape[1])
    p_idx = torch.randperm(present_ei.shape[1])[:n]
    a_idx = torch.randperm(absent_ei.shape[1])[:n]
    present_ei = present_ei[:, p_idx]
    absent_ei = absent_ei[:, a_idx]
    if present_et is not None:
        present_et = present_et[p_idx]
        absent_et = absent_et[a_idx]

    s_present = _edge_scores_simple(model, present_ei, present_et, ctx_ei, ctx_et, data)
    s_absent = _edge_scores_simple(model, absent_ei, absent_et, ctx_ei, ctx_et, data)

    features = torch.stack(
        [torch.cat([1 - s_present, 1 - s_absent]),
         torch.cat([s_present, s_absent])], dim=1)
    labels = torch.cat([
        torch.ones(n, dtype=torch.long),
        torch.zeros(n, dtype=torch.long),
    ])
    return features, labels


def _train_attack(attack_model, features, labels, save_dir, epochs=50, lr=1e-3):
    os.makedirs(save_dir, exist_ok=True)

    n_val = max(1, int(0.2 * len(features)))
    n_train = len(features) - n_val
    perm = torch.randperm(len(features))
    train_feat, train_lbl = features[perm[:n_train]], labels[perm[:n_train]]
    val_feat, val_lbl = features[perm[n_train:]], labels[perm[n_train:]]

    train_loader = DataLoader(TensorDataset(train_feat, train_lbl), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_feat, val_lbl), batch_size=256)

    optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr)
    attack_model = attack_model.to(device)

    best_auc = 0.0
    for epoch in range(epochs):
        attack_model.train()
        for x, y in train_loader:
            logits = attack_model(x.to(device))
            loss = F.cross_entropy(logits, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        attack_model.eval()
        all_pred, all_lbl = [], []
        with torch.no_grad():
            for x, y in val_loader:
                probs = torch.softmax(attack_model(x.to(device)), dim=1)[:, 1].cpu()
                all_pred.append(probs)
                all_lbl.append(y)
        preds = torch.cat(all_pred).numpy()
        lbls = torch.cat(all_lbl).numpy()

        if len(np.unique(lbls)) < 2:
            continue
        auc = roc_auc_score(lbls, preds)
        if auc > best_auc:
            best_auc = auc
            torch.save({'model_state': attack_model.state_dict(), 'best_auc': best_auc},
                       os.path.join(save_dir, 'attack_model_best.pt'))

    print(f'  Best attack AUC: {best_auc:.4f}')
    return attack_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    seed_everything(args.random_seed)

    original_path = os.path.join(
        args.checkpoint_dir, args.dataset, args.gnn, 'original', str(args.random_seed))
    attack_path_all = os.path.join(
        args.checkpoint_dir, args.dataset, args.gnn, 'member_infer_all', str(args.random_seed))
    attack_path_sub = os.path.join(
        args.checkpoint_dir, args.dataset, args.gnn, 'member_infer_sub', str(args.random_seed))

    # Load dataset
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
        dataset, data = pickle.load(f)
    print('Dataset:', dataset, data)

    is_kg = args.gnn in ['rgcn', 'rgat']
    if not is_kg:
        args.in_dim = dataset.num_features

    # Build random df/dr masks (0.5% of training edges)
    n_edges = data.train_pos_edge_index.shape[1]
    df_size = max(1, int(0.005 * n_edges))
    perm = torch.randperm(n_edges)
    df_idx = perm[:df_size]

    dr_mask = torch.ones(n_edges, dtype=torch.bool)
    dr_mask[df_idx] = False
    df_mask = ~dr_mask

    # Subgraph masks required by model constructor
    _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(),
        2, data.train_pos_edge_index, num_nodes=data.num_nodes)
    _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(),
        1, data.train_pos_edge_index, num_nodes=data.num_nodes)

    sdf_node_1hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_2hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_1hop[one_hop_edge.flatten().unique()] = True
    sdf_node_2hop[two_hop_edge.flatten().unique()] = True

    # Save directed masks before doubling
    directed_dr_mask = dr_mask.clone()
    directed_df_mask = df_mask.clone()

    # To undirected / doubled for KG
    if is_kg:
        r, c = data.train_pos_edge_index
        rev_ei = torch.stack([c, r], dim=0)
        rev_et = data.train_edge_type + args.num_edge_type
        data.edge_index = torch.cat([data.train_pos_edge_index, rev_ei], dim=1)
        data.edge_type = torch.cat([data.train_edge_type, rev_et], dim=0)
        two_hop_mask = two_hop_mask.repeat(2)
        df_mask = df_mask.repeat(2)
        dr_mask = dr_mask.repeat(2)
    else:
        train_pos_edge_index, [df_mask_i, two_hop_mask_i] = to_undirected(
            data.train_pos_edge_index, [df_mask.int(), two_hop_mask.int()])
        two_hop_mask = two_hop_mask_i.bool()
        df_mask = df_mask_i.bool()
        dr_mask = ~df_mask
        data.train_pos_edge_index = train_pos_edge_index
        data.edge_index = train_pos_edge_index
        directed_dr_mask = dr_mask
        directed_df_mask = df_mask

    data.sdf_mask = two_hop_mask
    data.df_mask = df_mask
    data.dr_mask = dr_mask
    data.directed_dr_mask = directed_dr_mask
    data.directed_df_mask = directed_df_mask

    # Load original model
    model = get_model(args, sdf_node_1hop, sdf_node_2hop,
                      num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
    ckpt_path = os.path.join(original_path, 'model_best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'Original model not found: {ckpt_path}\n'
            f'Run train_gnn.py first.')
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'], strict=False)
    model = model.to(device).eval()
    print(f'Loaded original model from {ckpt_path}')

    # Train attack models
    print('\n--- Training member_infer_all ---')
    feat_all, lbl_all = _build_attack_data(model, data, mode='all', is_kg=is_kg)
    _train_attack(MLPAttacker(), feat_all, lbl_all, attack_path_all)

    print('\n--- Training member_infer_sub ---')
    feat_sub, lbl_sub = _build_attack_data(model, data, mode='sub', is_kg=is_kg)
    _train_attack(MLPAttacker(), feat_sub, lbl_sub, attack_path_sub)

    print(f'\nAttack models saved to:')
    print(f'  {attack_path_all}/attack_model_best.pt')
    print(f'  {attack_path_sub}/attack_model_best.pt')
    print(f'  (Trained on {args.gnn} original model, seed={args.random_seed})')


if __name__ == '__main__':
    main()
