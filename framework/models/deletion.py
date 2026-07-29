import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from . import GCN, GAT, GIN, RGCN, RGAT
from .hetero_kg import HGT, HAN


class DeletionLayer(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
        # self.deletion_weight = nn.Parameter(torch.eye(dim, dim))
        # init.xavier_uniform_(self.deletion_weight)
    
    def forward(self, x, mask=None, delete_edge_type=None, delete_edge_index=None, message_edge_index=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask
        
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)

            return new_rep

        return x

class DeletionLayerKG(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
    
    def forward(self, x, mask=None, delete_edge_type=None, delete_edge_index=None, message_edge_index=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask
        
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)

            return new_rep

        return x


class SCGULowRankDeletionLayerKG(nn.Module):
    """SCGU subspace-constrained KG deletion operator.

    This implements the core operator from the SCGU paper/code as a drop-in
    KG deletion layer:

        W_r = I + A B_r,  A in R^{d x k}, B_r in R^{k x d}, k << d
        h'_v = h_v W_r, for nodes v in the affected subgraph mask.

    A is shared across relations and each relation has its own B_r.  For a
    delete request containing multiple relation types, SCGU uses their
    frequency-weighted average B_r, matching the official implementation's
    subgraph-level relation-aware low-rank update.  Reverse KG relation ids
    are mapped back with modulo num_relations.
    """

    def __init__(self, dim, mask, num_relations, rank=16, init_strategy='random'):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.num_relations = num_relations
        self.rank = rank

        # Prefix with "del_" so non-layerwise optimizers that select deletion
        # parameters by name keep working.  A/B aliases are exposed via
        # properties for readability and official-code parity.
        self.del_A = nn.Parameter(torch.randn(dim, rank) * 0.1)
        if init_strategy == 'zero':
            self.del_B = nn.Parameter(torch.zeros(num_relations, rank, dim))
        elif init_strategy == 'random':
            self.del_B = nn.Parameter(torch.randn(num_relations, rank, dim) * 0.1)
        else:
            raise ValueError(f'Unknown scgu_init: {init_strategy}')

    @property
    def A(self):
        return self.del_A

    @property
    def B(self):
        return self.del_B

    def _relation_weighted_B(self, delete_edge_type):
        if delete_edge_type is None:
            return self.del_B.mean(dim=0)

        rel_ids = delete_edge_type.to(self.del_B.device).view(-1)
        if rel_ids.numel() == 0:
            return self.del_B.mean(dim=0)

        # KG delete_gnn makes reverse edges as r + num_relations.  SCGU uses
        # the underlying forward relation to select B_r.
        rel_ids = rel_ids.remainder(self.num_relations).long()
        counts = torch.bincount(rel_ids, minlength=self.num_relations).to(self.del_B.dtype)
        weights = counts / (counts.sum() + 1e-8)
        return (weights.view(-1, 1, 1) * self.del_B).sum(dim=0)

    def forward(self, x, mask=None, delete_edge_type=None, delete_edge_index=None, message_edge_index=None):
        if mask is None:
            mask = self.mask

        if mask is None:
            return x

        new_rep = x.clone()
        local_x = new_rep[mask]
        if local_x.numel() == 0:
            return new_rep

        identity = torch.eye(self.dim, device=x.device, dtype=x.dtype)
        B = self._relation_weighted_B(delete_edge_type).to(device=x.device, dtype=x.dtype)
        deletion_weight = identity + torch.matmul(self.del_A.to(dtype=x.dtype), B)
        new_rep[mask] = torch.matmul(local_x, deletion_weight)
        return new_rep

    @torch.no_grad()
    def get_deletion_weight(self, relation_id=None):
        identity = torch.eye(self.dim, device=self.del_A.device, dtype=self.del_A.dtype)
        if relation_id is None:
            B = self.del_B.mean(dim=0)
        else:
            B = self.del_B[int(relation_id) % self.num_relations]
        return identity + torch.matmul(self.del_A, B)

    @torch.no_grad()
    def get_all_deletion_weights(self):
        identity = torch.eye(self.dim, device=self.del_A.device, dtype=self.del_A.dtype)
        return torch.stack([identity + torch.matmul(self.del_A, self.del_B[i])
                            for i in range(self.num_relations)])

    @torch.no_grad()
    def get_relation_diversity(self):
        B_flat = self.del_B.view(self.num_relations, -1)
        sim = torch.cosine_similarity(B_flat.unsqueeze(1), B_flat.unsqueeze(0), dim=2)
        off_diag = ~torch.eye(self.num_relations, dtype=torch.bool, device=sim.device)
        return sim[off_diag].mean()


class RelationLoRAMoEDeletionLayer(nn.Module):
    """Relation-driven LoRA-MoE deletion operator for heterogeneous graphs.

    This is a drop-in replacement for ``DeletionLayerKG``.  The local nodes
    still share one deletion operator in a forward pass, but the operator is
    selected by the relation type(s) of the edge(s) being deleted:

        DEL_r(h) = h W_base + scale * sum_k g_k(r) * h A^T B_k^T

    where A is shared by all experts and only B_k is expert-specific.  If a
    mini-batch contains several deleted relation types, their gates are
    averaged into one subgraph-level gate, preserving the Neighborhood
    Influence consistency assumption used by GNNDelete.
    """

    def __init__(
        self,
        dim,
        mask,
        num_relations,
        num_experts=4,
        rank=8,
        gate_emb_dim=16,
        lora_alpha=1.0,
        dropout=0.0,
        gate_mode='soft',
        gate_temperature=1.0,
        base_mode='deletion',
    ):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.num_relations = num_relations
        self.num_experts = num_experts
        self.rank = rank
        self.scaling = lora_alpha / max(rank, 1)
        self.gate_mode = gate_mode
        self.gate_temperature = gate_temperature
        self.base_mode = base_mode

        if base_mode not in ['deletion', 'identity']:
            raise ValueError(f'Unknown del_lora_base_mode: {base_mode}')

        # In 'deletion' mode, keep the original full-rank GNNDelete deletion
        # matrix. In 'identity' mode, use h as the frozen base and do NOT
        # register a full-rank deletion_weight parameter, yielding a genuinely
        # parameter-efficient LoRA-only deletion operator.
        if base_mode == 'deletion':
            self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
        else:
            self.register_parameter('deletion_weight', None)

        # Shared low-rank down projection A; expert-specific up projections B_k.
        self.del_lora_A = nn.Parameter(torch.empty(rank, dim))
        self.del_lora_B = nn.Parameter(torch.zeros(num_experts, dim, rank))

        self.del_relation_emb = nn.Embedding(num_relations, gate_emb_dim)
        self.del_gate = nn.Linear(gate_emb_dim, num_experts)
        self.del_dropout = nn.Dropout(dropout)

        init.kaiming_uniform_(self.del_lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.del_gate.bias)

    def _straight_through_hard_gate(self, probs):
        """One-hot in forward, softmax gradient in backward."""
        index = probs.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(probs).scatter_(-1, index, 1.0)
        return hard - probs.detach() + probs

    def _relation_gate(self, delete_edge_type, device):
        if delete_edge_type is None:
            # Safe fallback for evaluation/backward compatibility: use the
            # average gate over all relation types.
            rel = torch.arange(self.num_relations, device=device)
        else:
            rel = delete_edge_type.to(device).view(-1)
            if rel.numel() == 0:
                rel = torch.arange(self.num_relations, device=device)
            # Reverse KG edges are encoded as r + num_relations.
            rel = rel.remainder(self.num_relations).long()

        logits = self.del_gate(self.del_relation_emb(rel))
        probs = F.softmax(logits / self.gate_temperature, dim=-1)

        if self.gate_mode == 'soft':
            per_relation_gate = probs
        elif self.gate_mode == 'hard':
            per_relation_gate = self._straight_through_hard_gate(probs)
        else:
            raise ValueError(f'Unknown del_gate_mode: {self.gate_mode}')

        gate = per_relation_gate.mean(dim=0)
        return gate

    def forward(self, x, mask=None, delete_edge_type=None, delete_edge_index=None, message_edge_index=None):
        """Apply relation-gated LoRA-MoE deletion to local nodes only."""
        if mask is None:
            mask = self.mask

        if mask is None:
            return x

        new_rep = x.clone()
        local_x = new_rep[mask]
        if local_x.numel() == 0:
            return new_rep

        if self.base_mode == 'identity':
            base = local_x
        else:
            base = torch.matmul(local_x, self.deletion_weight)

        gate = self._relation_gate(delete_edge_type, x.device)
        low_rank = torch.matmul(self.del_dropout(local_x), self.del_lora_A.t())
        # [N, E, D], then gate-weighted sum over experts.
        expert_delta = torch.einsum('nr,edr->ned', low_rank, self.del_lora_B)
        delta = torch.einsum('e,ned->nd', gate, expert_delta)

        new_rep[mask] = base + self.scaling * delta
        return new_rep


class SCGUMoEDeletionLayerKG(nn.Module):
    """SCGU + MoE: relation-gated mixture of SCGU expert weights.

    Each expert e has its own B_e; the deletion weight is:
        W = sum_e g_e(r) * (I + A B_e)
    """

    def __init__(self, dim, mask, num_relations, rank=16, num_experts=4,
                 gate_emb_dim=16, gate_mode='soft', gate_temperature=1.0,
                 init_strategy='random'):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.num_relations = num_relations
        self.gate_mode = gate_mode
        self.gate_temperature = gate_temperature

        self.del_A = nn.Parameter(torch.randn(dim, rank) * 0.1)
        if init_strategy == 'zero':
            self.del_B = nn.Parameter(torch.zeros(num_experts, rank, dim))
        else:
            self.del_B = nn.Parameter(torch.randn(num_experts, rank, dim) * 0.1)

        self.del_relation_emb = nn.Embedding(num_relations, gate_emb_dim)
        self.del_gate = nn.Linear(gate_emb_dim, num_experts)
        nn.init.zeros_(self.del_gate.bias)

    def _relation_gate(self, delete_edge_type, device):
        if delete_edge_type is None:
            rel = torch.arange(self.num_relations, device=device)
        else:
            rel = delete_edge_type.to(device).view(-1)
            if rel.numel() == 0:
                rel = torch.arange(self.num_relations, device=device)
            rel = rel.remainder(self.num_relations).long()
        logits = self.del_gate(self.del_relation_emb(rel))
        probs = F.softmax(logits / self.gate_temperature, dim=-1)
        if self.gate_mode == 'hard':
            idx = probs.argmax(dim=-1, keepdim=True)
            hard = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
            probs = hard - probs.detach() + probs
        return probs.mean(dim=0)  # (num_experts,)

    def forward(self, x, mask=None, delete_edge_type=None, delete_edge_index=None, message_edge_index=None):
        if mask is None:
            mask = self.mask
        if mask is None:
            return x
        new_rep = x.clone()
        local_x = new_rep[mask]
        if local_x.numel() == 0:
            return new_rep

        # Original SCGU-MoE behavior: average gates over all deleted relation
        # types in the request, build one subgraph-level deletion matrix, and
        # apply it to all affected nodes.  delete_edge_index/message_edge_index
        # are accepted for API compatibility but intentionally unused here.
        gate = self._relation_gate(delete_edge_type, x.device)
        A = self.del_A.to(dtype=x.dtype)
        B = self.del_B.to(dtype=x.dtype)  # (E, r, d)

        # AB[e] = A @ B[e]: (d,r) x (r,d) -> (d,d), stacked: (E, d, d)
        AB = torch.matmul(A.unsqueeze(0), B)  # (E, d, d)
        identity = torch.eye(self.dim, device=x.device, dtype=x.dtype)
        # gate-weighted sum of (I + A B_e)
        deletion_weight = identity + torch.einsum('e,eij->ij', gate, AB)
        new_rep[mask] = torch.matmul(local_x, deletion_weight)
        return new_rep


class PlainMoEDeletionLayerKG(nn.Module):
    """Plain MoE deletion operator without SCGU shared-A constraint.

    Each expert has its own independent low-rank pair (A_e, B_e):
        W = I + sum_e g_e(r) * (A_e @ B_e)

    Unlike SCGUMoEDeletionLayerKG, A is NOT shared across experts, so there
    is no subspace constraint.  Gate network is identical to SCGU MoE.
    """

    def __init__(self, dim, mask, num_relations, rank=16, num_experts=4,
                 gate_emb_dim=16, gate_mode='soft', gate_temperature=1.0,
                 init_strategy='random'):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.num_relations = num_relations
        self.gate_mode = gate_mode
        self.gate_temperature = gate_temperature
        self.num_experts = num_experts

        # Each expert has its own A_e (d, r) and B_e (r, d) — no sharing
        if init_strategy == 'zero':
            self.del_A = nn.Parameter(torch.zeros(num_experts, dim, rank))
            self.del_B = nn.Parameter(torch.zeros(num_experts, rank, dim))
        else:
            self.del_A = nn.Parameter(torch.randn(num_experts, dim, rank) * 0.1)
            self.del_B = nn.Parameter(torch.randn(num_experts, rank, dim) * 0.1)

        self.del_relation_emb = nn.Embedding(num_relations, gate_emb_dim)
        self.del_gate = nn.Linear(gate_emb_dim, num_experts)
        nn.init.zeros_(self.del_gate.bias)

    def _relation_gate(self, delete_edge_type, device):
        if delete_edge_type is None:
            rel = torch.arange(self.num_relations, device=device)
        else:
            rel = delete_edge_type.to(device).view(-1)
            if rel.numel() == 0:
                rel = torch.arange(self.num_relations, device=device)
            rel = rel.remainder(self.num_relations).long()
        logits = self.del_gate(self.del_relation_emb(rel))
        probs = F.softmax(logits / self.gate_temperature, dim=-1)
        if self.gate_mode == 'hard':
            idx = probs.argmax(dim=-1, keepdim=True)
            hard = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
            probs = hard - probs.detach() + probs
        return probs.mean(dim=0)  # (num_experts,)

    def forward(self, x, mask=None, delete_edge_type=None, delete_edge_index=None, message_edge_index=None):
        if mask is None:
            mask = self.mask
        if mask is None:
            return x
        new_rep = x.clone()
        local_x = new_rep[mask]
        if local_x.numel() == 0:
            return new_rep

        gate = self._relation_gate(delete_edge_type, x.device)
        A = self.del_A.to(dtype=x.dtype)  # (E, d, r)
        B = self.del_B.to(dtype=x.dtype)  # (E, r, d)

        # AB[e] = A[e] @ B[e]: (d,r) x (r,d) -> (d,d), stacked: (E, d, d)
        AB = torch.bmm(A, B)  # (E, d, d)
        identity = torch.eye(self.dim, device=x.device, dtype=x.dtype)
        deletion_weight = identity + torch.einsum('e,eij->ij', gate, AB)
        new_rep[mask] = torch.matmul(local_x, deletion_weight)
        return new_rep


def build_kg_deletion_layer(args, dim, mask, num_edge_type):
    """Factory for pluggable KG deletion operators."""
    deletion_operator = getattr(args, 'deletion_operator', 'original')
    if deletion_operator == 'original':
        return DeletionLayerKG(dim, mask)
    if deletion_operator in ['scgu', 'scgu_lowrank', 'lowrank']:
        return SCGULowRankDeletionLayerKG(
            dim=dim,
            mask=mask,
            num_relations=num_edge_type,
            rank=getattr(args, 'scgu_rank', 16),
            init_strategy=getattr(args, 'scgu_init', 'random'),
        )
    if deletion_operator in ['relation_lora_moe', 'lora_moe']:
        return RelationLoRAMoEDeletionLayer(
            dim=dim,
            mask=mask,
            num_relations=num_edge_type,
            num_experts=getattr(args, 'del_moe_num_experts', 4),
            rank=getattr(args, 'del_lora_rank', 8),
            gate_emb_dim=getattr(args, 'del_gate_emb_dim', 16),
            lora_alpha=getattr(args, 'del_lora_alpha', 1.0),
            dropout=getattr(args, 'del_lora_dropout', 0.0),
            gate_mode=getattr(args, 'del_gate_mode', 'soft'),
            gate_temperature=getattr(args, 'del_gate_temperature', 1.0),
            base_mode=getattr(args, 'del_lora_base_mode', 'deletion'),
        )
    if deletion_operator == 'scgu_moe':
        return SCGUMoEDeletionLayerKG(
            dim=dim,
            mask=mask,
            num_relations=num_edge_type,
            rank=getattr(args, 'scgu_rank', 16),
            num_experts=getattr(args, 'del_moe_num_experts', 4),
            gate_emb_dim=getattr(args, 'del_gate_emb_dim', 16),
            gate_mode=getattr(args, 'del_gate_mode', 'soft'),
            gate_temperature=getattr(args, 'del_gate_temperature', 1.0),
            init_strategy=getattr(args, 'scgu_init', 'random'),
        )
    if deletion_operator == 'plain_moe':
        return PlainMoEDeletionLayerKG(
            dim=dim,
            mask=mask,
            num_relations=num_edge_type,
            rank=getattr(args, 'scgu_rank', 16),
            num_experts=getattr(args, 'del_moe_num_experts', 4),
            gate_emb_dim=getattr(args, 'del_gate_emb_dim', 16),
            gate_mode=getattr(args, 'del_gate_mode', 'soft'),
            gate_temperature=getattr(args, 'del_gate_temperature', 1.0),
            init_strategy=getattr(args, 'scgu_init', 'random'),
        )
    raise ValueError(f'Unknown deletion_operator: {deletion_operator}')


class GCNDelete(GCN):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        # with torch.no_grad():
        x1 = self.conv1(x, edge_index)
        
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.conv2(x, edge_index)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2
        
        return x2
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)

class GATDelete(GAT):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        with torch.no_grad():
            x1 = self.conv1(x, edge_index)
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.conv2(x, edge_index)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2
        
        return x2
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)

class GINDelete(GIN):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        with torch.no_grad():
            x1 = self.conv1(x, edge_index)
        
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.conv2(x, edge_index)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2

        return x2
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)

class RGCNDelete(RGCN):
    def __init__(self, args, num_nodes, num_edge_type, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args, num_nodes, num_edge_type)
        self.deletion1 = build_kg_deletion_layer(args, args.hidden_dim, mask_1hop, num_edge_type)
        self.deletion2 = build_kg_deletion_layer(args, args.out_dim, mask_2hop, num_edge_type)
        if hasattr(self.deletion1, 'hop'):
            self.deletion1.hop = 1
        if hasattr(self.deletion2, 'hop'):
            self.deletion2.hop = 2

        for module in [self.node_emb, self.conv1, self.conv2]:
            for param in module.parameters():
                param.requires_grad = False
        self.W.requires_grad = False

    def forward(self, x, edge_index, edge_type, mask_1hop=None, mask_2hop=None, return_all_emb=False, delete_edge_type=None, delete_edge_index=None):
        with torch.no_grad():
            x = self.node_emb(x)
            x1 = self.conv1(x, edge_index, edge_type)
        
        x1 = self.deletion1(
            x1, mask_1hop,
            delete_edge_type=delete_edge_type,
            delete_edge_index=delete_edge_index,
            message_edge_index=edge_index)

        x = F.relu(x1)
        
        x2 = self.conv2(x, edge_index, edge_type)
        x2 = self.deletion2(
            x2, mask_2hop,
            delete_edge_type=delete_edge_type,
            delete_edge_index=delete_edge_index,
            message_edge_index=edge_index)

        if return_all_emb:
            return x1, x2
        
        return x2
    
    def get_original_embeddings(self, x, edge_index, edge_type, return_all_emb=False):
        return super().forward(x, edge_index, edge_type, return_all_emb)

class RGATDelete(RGAT):
    def __init__(self, args, num_nodes, num_edge_type, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args, num_nodes, num_edge_type)
        self.deletion1 = build_kg_deletion_layer(args, args.hidden_dim, mask_1hop, num_edge_type)
        self.deletion2 = build_kg_deletion_layer(args, args.out_dim, mask_2hop, num_edge_type)
        if hasattr(self.deletion1, 'hop'):
            self.deletion1.hop = 1
        if hasattr(self.deletion2, 'hop'):
            self.deletion2.hop = 2

        for module in [self.node_emb, self.conv1, self.conv2]:
            for param in module.parameters():
                param.requires_grad = False
        self.W.requires_grad = False

    def forward(self, x, edge_index, edge_type, mask_1hop=None, mask_2hop=None, return_all_emb=False, delete_edge_type=None, delete_edge_index=None):
        with torch.no_grad():
            x = self.node_emb(x)
            x1 = self.conv1(x, edge_index, edge_type)
        
        x1 = self.deletion1(
            x1, mask_1hop,
            delete_edge_type=delete_edge_type,
            delete_edge_index=delete_edge_index,
            message_edge_index=edge_index)

        x = F.relu(x1)
        
        x2 = self.conv2(x, edge_index, edge_type)
        x2 = self.deletion2(
            x2, mask_2hop,
            delete_edge_type=delete_edge_type,
            delete_edge_index=delete_edge_index,
            message_edge_index=edge_index)

        if return_all_emb:
            return x1, x2
        
        return x2
    
    def get_original_embeddings(self, x, edge_index, edge_type, return_all_emb=False):
        return super().forward(x, edge_index, edge_type, return_all_emb)


class HGTDelete(HGT):
    def __init__(self, args, num_nodes, num_edge_type, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args, num_nodes, num_edge_type)
        self.deletion1 = build_kg_deletion_layer(args, args.hidden_dim, mask_1hop, num_edge_type)
        self.deletion2 = build_kg_deletion_layer(args, args.out_dim, mask_2hop, num_edge_type)
        if hasattr(self.deletion1, 'hop'):
            self.deletion1.hop = 1
        if hasattr(self.deletion2, 'hop'):
            self.deletion2.hop = 2

        for module in [self.node_emb, self.conv1, self.conv2, self.res1, self.res2, self.norm1, self.norm2]:
            for param in module.parameters():
                param.requires_grad = False
        self.W.requires_grad = False

    def forward(self, x, edge_index, edge_type, mask_1hop=None, mask_2hop=None, return_all_emb=False, delete_edge_type=None, delete_edge_index=None):
        with torch.no_grad():
            h0 = self.node_emb(x)
            edge_index_dict = self._edge_index_dict(edge_index, edge_type)
            h1_dict = self.conv1({self.node_type: h0}, edge_index_dict)
            h1 = h1_dict[self.node_type]
            if h1 is None:
                h1 = h0.new_zeros((h0.size(0), self.args.hidden_dim))
            h1 = self.relu(self.norm1(h1 + self.res1(h0)))

        h1 = self.deletion1(
            h1, mask_1hop,
            delete_edge_type=delete_edge_type,
            delete_edge_index=delete_edge_index,
            message_edge_index=edge_index)

        h2_dict = self.conv2({self.node_type: h1}, edge_index_dict)
        h2 = h2_dict[self.node_type]
        if h2 is None:
            h2 = h1.new_zeros((h1.size(0), self.args.out_dim))
        h2 = self.norm2(h2 + self.res2(h1))
        h2 = self.deletion2(
            h2, mask_2hop,
            delete_edge_type=delete_edge_type,
            delete_edge_index=delete_edge_index,
            message_edge_index=edge_index)

        if return_all_emb:
            return h1, h2
        return h2

    def get_original_embeddings(self, x, edge_index, edge_type, return_all_emb=False):
        return super().forward(x, edge_index, edge_type, return_all_emb)


class HANDelete(HAN):
    def __init__(self, args, num_nodes, num_edge_type, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args, num_nodes, num_edge_type)
        self.deletion1 = build_kg_deletion_layer(args, args.hidden_dim, mask_1hop, num_edge_type)
        self.deletion2 = build_kg_deletion_layer(args, args.out_dim, mask_2hop, num_edge_type)
        if hasattr(self.deletion1, 'hop'):
            self.deletion1.hop = 1
        if hasattr(self.deletion2, 'hop'):
            self.deletion2.hop = 2

        for module in [self.node_emb, self.conv1, self.conv2, self.res1, self.res2, self.norm1, self.norm2]:
            for param in module.parameters():
                param.requires_grad = False
        self.W.requires_grad = False

    def forward(self, x, edge_index, edge_type, mask_1hop=None, mask_2hop=None, return_all_emb=False, delete_edge_type=None, delete_edge_index=None):
        with torch.no_grad():
            h0 = self.node_emb(x)
            edge_index_dict = self._edge_index_dict(edge_index, edge_type)
            h1_dict = self.conv1({self.node_type: h0}, edge_index_dict)
            h1 = h1_dict[self.node_type]
            if h1 is None:
                h1 = h0.new_zeros((h0.size(0), self.args.hidden_dim))
            h1 = self.relu(self.norm1(h1 + self.res1(h0)))

        h1 = self.deletion1(
            h1, mask_1hop,
            delete_edge_type=delete_edge_type,
            delete_edge_index=delete_edge_index,
            message_edge_index=edge_index)

        h2_dict = self.conv2({self.node_type: h1}, edge_index_dict)
        h2 = h2_dict[self.node_type]
        if h2 is None:
            h2 = h1.new_zeros((h1.size(0), self.args.out_dim))
        h2 = self.norm2(h2 + self.res2(h1))
        h2 = self.deletion2(
            h2, mask_2hop,
            delete_edge_type=delete_edge_type,
            delete_edge_index=delete_edge_index,
            message_edge_index=edge_index)

        if return_all_emb:
            return h1, h2
        return h2

    def get_original_embeddings(self, x, edge_index, edge_type, return_all_emb=False):
        return super().forward(x, edge_index, edge_type, return_all_emb)
