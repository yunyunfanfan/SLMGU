import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, HANConv


class _HeteroKGBase(nn.Module):
    """KG link-prediction wrapper for heterogeneous PyG convs.

    The project stores WordNet18 as a single entity node type plus relation ids.
    This wrapper exposes the same interface as RGCN/RGAT:
      forward(x, edge_index, edge_type) -> entity embeddings
      decode(z, edge_index, edge_type) -> DistMult logits
    Internally, each (directed + reverse) relation id is treated as a hetero
    edge type ('entity', 'rel_i', 'entity').
    """

    conv_cls = None

    def __init__(self, args, num_nodes, num_edge_type, **kwargs):
        super().__init__()
        self.args = args
        self.num_edge_type = num_edge_type
        self.num_total_rel = num_edge_type * 2
        self.node_type = 'entity'
        self.edge_types = [(self.node_type, f'rel_{i}', self.node_type) for i in range(self.num_total_rel)]
        self.metadata = ([self.node_type], self.edge_types)

        self.node_emb = nn.Embedding(num_nodes, args.in_dim)
        heads = int(getattr(args, 'hetero_heads', 2))
        dropout = float(getattr(args, 'hetero_dropout', 0.0))

        self.conv1 = self._make_conv(args.in_dim, args.hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = self._make_conv(args.hidden_dim, args.out_dim, heads=heads, dropout=dropout)
        # HGT/HANConv do not add a DistMult-friendly identity path by default.
        # On single-node-type KG data this can collapse pos/neg logits around 0.
        # Residual projections keep entity identity information trainable.
        self.res1 = nn.Linear(args.in_dim, args.hidden_dim, bias=False)
        self.res2 = nn.Linear(args.hidden_dim, args.out_dim, bias=False)
        self.norm1 = nn.LayerNorm(args.hidden_dim)
        self.norm2 = nn.LayerNorm(args.out_dim)
        self.relu = nn.ReLU()

        # DistMult decoder over original directed relation ids.
        self.W = nn.Parameter(torch.empty(num_edge_type, args.out_dim))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))

    def _make_conv(self, in_channels, out_channels, heads=2, dropout=0.0):
        raise NotImplementedError

    def _edge_index_dict(self, edge_index, edge_type):
        # HGTConv/HANConv expect a dictionary keyed by metadata edge types.
        out = {}
        dev = edge_index.device
        empty = edge_index.new_empty((2, 0))
        for i, et in enumerate(self.edge_types):
            mask = edge_type == i
            out[et] = edge_index[:, mask] if bool(mask.any()) else empty
        return out

    def forward(self, x, edge, edge_type, return_all_emb=False):
        # x is node id tensor in this project.
        h0 = self.node_emb(x)
        x_dict = {self.node_type: h0}
        edge_index_dict = self._edge_index_dict(edge, edge_type)

        h1_dict = self.conv1(x_dict, edge_index_dict)
        h1 = h1_dict[self.node_type]
        if h1 is None:
            h1 = h0.new_zeros((h0.size(0), self.args.hidden_dim))
        h1 = self.relu(self.norm1(h1 + self.res1(h0)))

        h2_dict = self.conv2({self.node_type: h1}, edge_index_dict)
        h2 = h2_dict[self.node_type]
        if h2 is None:
            h2 = h1.new_zeros((h1.size(0), self.args.out_dim))
        h2 = self.norm2(h2 + self.res2(h1))

        if return_all_emb:
            return h1, h2
        return h2

    def decode(self, z, edge_index, edge_type):
        h = z[edge_index[0]]
        t = z[edge_index[1]]
        r = self.W[edge_type]
        return torch.sum(h * r * t, dim=1)


class HGT(_HeteroKGBase):
    def _make_conv(self, in_channels, out_channels, heads=2, dropout=0.0):
        return HGTConv(
            in_channels={self.node_type: in_channels},
            out_channels=out_channels,
            metadata=self.metadata,
            heads=heads,
        )


class HAN(_HeteroKGBase):
    def _make_conv(self, in_channels, out_channels, heads=2, dropout=0.0):
        return HANConv(
            in_channels={self.node_type: in_channels},
            out_channels=out_channels,
            metadata=self.metadata,
            heads=heads,
            dropout=dropout,
        )
