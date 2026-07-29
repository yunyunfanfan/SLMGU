import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class CompGCNConv(nn.Module):
    """A lightweight CompGCN-style layer for single-entity-type KG data.

    The layer composes neighboring entity embeddings with relation embeddings
    before aggregation, then updates both entity and relation representations.
    It is intentionally dependency-light so it can run in the existing training
    pipeline without requiring an external CompGCN package.
    """

    def __init__(self, in_dim, out_dim, num_relations, opn='mult', dropout=0.0):
        super().__init__()
        self.num_relations = num_relations
        self.opn = opn
        self.w_msg = nn.Linear(in_dim, out_dim, bias=False)
        self.w_self = nn.Linear(in_dim, out_dim, bias=True)
        self.w_rel = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_dim)

    def compose(self, x, r):
        if self.opn == 'sub':
            return x - r
        if self.opn == 'corr':
            # Circular correlation implemented via FFT.  Kept for completeness;
            # default is multiplication for speed/stability.
            return torch.fft.irfft(
                torch.conj(torch.fft.rfft(x, dim=-1)) * torch.fft.rfft(r, dim=-1),
                n=x.size(-1),
                dim=-1,
            )
        return x * r

    def forward(self, x, edge_index, edge_type, rel_emb):
        row, col = edge_index
        msg = self.compose(x[row], rel_emb[edge_type])
        msg = self.w_msg(msg)
        out = scatter(msg, col, dim=0, dim_size=x.size(0), reduce='mean')
        out = out + self.w_self(x)
        out = self.bn(out)
        out = self.dropout(out)
        rel_emb = self.w_rel(rel_emb)
        return out, rel_emb


class CompGCN(nn.Module):
    def __init__(self, args, num_nodes, num_edge_type, **kwargs):
        super().__init__()
        self.args = args
        self.num_edge_type = num_edge_type
        self.num_total_rel = num_edge_type * 2
        opn = getattr(args, 'compgcn_opn', 'mult')
        dropout = float(getattr(args, 'compgcn_dropout', 0.0))

        self.node_emb = nn.Embedding(num_nodes, args.in_dim)
        self.rel_emb = nn.Embedding(self.num_total_rel, args.in_dim)
        self.conv1 = CompGCNConv(args.in_dim, args.hidden_dim, self.num_total_rel, opn=opn, dropout=dropout)
        self.conv2 = CompGCNConv(args.hidden_dim, args.out_dim, self.num_total_rel, opn=opn, dropout=dropout)
        self.relu = nn.ReLU()

        # DistMult decoder over original directed relation ids.
        self.W = nn.Parameter(torch.empty(num_edge_type, args.out_dim))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, edge, edge_type, return_all_emb=False):
        x = self.node_emb(x)
        rel = self.rel_emb.weight
        x1, rel1 = self.conv1(x, edge, edge_type, rel)
        h = self.relu(x1)
        x2, _ = self.conv2(h, edge, edge_type, rel1)

        if return_all_emb:
            return x1, x2
        return x2

    def decode(self, z, edge_index, edge_type):
        h = z[edge_index[0]]
        t = z[edge_index[1]]
        r = self.W[edge_type]
        return torch.sum(h * r * t, dim=1)
