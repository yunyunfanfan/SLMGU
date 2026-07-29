import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from .utils import get_link_labels


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def eval_lp(model, stage, data=None, loader=None):
    model.eval()
    
    # For full batch
    if data is not None:
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']
        
        if hasattr(data, 'dtrain_mask') and data.dtrain_mask is not None:
            embedding = model(data.x.to(device), data.train_pos_edge_index[:, data.dtrain_mask].to(device))
        else:
            embedding = model(data.x.to(device), data.train_pos_edge_index.to(device))

        logits = model.decode(embedding, pos_edge_index, neg_edge_index).sigmoid()
        label = get_link_labels(pos_edge_index, neg_edge_index)

    # For mini batch
    if loader is not None:
        logits = []
        label = []
        for batch in loader:
            edge_index = batch.edge_index.to(device)

            if hasattr(batch, 'edge_type'):
                edge_type = batch.edge_type.to(device)
            
                embedding1 = model1(edge_index, edge_type)
                embedding2 = model2(edge_index, edge_type)

                s1 = model.decode(embedding1, edge_index, edge_type)
                s2 = model.decode(embedding2, edge_index, edge_type)

            else:
                embedding1 = model1(edge_index)
                embedding2 = model2(edge_index)

                s1 = model.decode(embedding1, edge_index)
                s2 = model.decode(embedding2, edge_index)

        embedding = model(data.train_pos_edge_index.to(device))

        lg = model.decode(embedding, pos_edge_index, neg_edge_index).sigmoid()
        lb = get_link_labels(pos_edge_index, neg_edge_index)

        logits.append(lg)
        label.append(lb)

    loss = F.binary_cross_entropy_with_logits(logits, label)
    auc = roc_auc_score(label.cpu(), logits.cpu())
    aup = average_precision_score(label.cpu(), logits.cpu())

    return loss, auc, aup

@torch.no_grad()
def verification_error(model1, model2):
    '''L2 distance between aproximate model and re-trained model'''

    model1 = model1.to('cpu')
    model2 = model2.to('cpu')

    modules1 = {n: p for n, p in model1.named_parameters()}
    modules2 = {n: p for n, p in model2.named_parameters()}

    all_names = set(modules1.keys()) & set(modules2.keys())

    print(all_names)

    diff = torch.tensor(0.0).float()
    for n in all_names:
        diff += torch.norm(modules1[n] - modules2[n])
    
    return diff

@torch.no_grad()
def member_infer_attack(
        target_model,
        attack_model,
        data,
        logits=None,
        use_full_graph=False,
        use_all_df=False):
    '''Membership inference attack. Works for both graph and KG models.

    use_full_graph/use_all_df are intended for the pre-unlearning attack
    baseline: when sweeping different deletion ratios, the initial model and
    dataset should be evaluated on the same full graph and same candidate Df
    pool, so mi_*_before is comparable and ratio-independent.
    '''
    dev = next(target_model.parameters()).device
    target_was_training = target_model.training
    attack_was_training = attack_model.training
    target_model.eval()
    attack_model.eval()
    is_kg = hasattr(data, 'directed_df_edge_type')

    if is_kg:
        if use_all_df and hasattr(data, 'directed_df_edge_index_all'):
            df_edge_index = data.directed_df_edge_index_all.to(dev)
            df_edge_type = data.directed_df_edge_type_all.to(dev)
        else:
            df_edge_index = data.directed_df_edge_index.to(dev)
            df_edge_type = data.directed_df_edge_type.to(dev)
        if use_full_graph:
            mp_edge_index = data.edge_index.to(dev)
            mp_edge_type = data.edge_type.to(dev)
        else:
            mp_edge_index = data.edge_index[:, data.dr_mask].to(dev)
            mp_edge_type = data.edge_type[data.dr_mask].to(dev)
        if use_full_graph and hasattr(target_model, 'get_original_embeddings'):
            # Pre-unlearning MI baselines must be computed on the exact
            # original GNN, not through method-specific deletion layers
            # (GNNDelete/SCGU/SCGUMoE initialize different deletion operators).
            z = target_model.get_original_embeddings(
                data.x.to(dev),
                mp_edge_index,
                mp_edge_type)
        else:
            z = target_model(
                data.x.to(dev),
                mp_edge_index,
                mp_edge_type)
        feature1 = target_model.decode(z, df_edge_index, df_edge_type).sigmoid()
    else:
        if (use_all_df and hasattr(data, 'df_mask_all')
                and data.df_mask_all.numel() == data.train_pos_edge_index.shape[1]):
            edge = data.train_pos_edge_index[:, data.df_mask_all].to(dev)
        else:
            edge = data.train_pos_edge_index[:, data.df_mask].to(dev)
        mp_edge_index = data.train_pos_edge_index.to(dev) if use_full_graph else data.train_pos_edge_index[:, data.dr_mask].to(dev)
        if use_full_graph and hasattr(target_model, 'get_original_embeddings'):
            z = target_model.get_original_embeddings(
                data.x.to(dev),
                mp_edge_index)
        else:
            z = target_model(
                data.x.to(dev),
                mp_edge_index)
        feature1 = target_model.decode(z, edge).sigmoid()

    feature0 = 1 - feature1
    feature = torch.stack([feature0, feature1], dim=1).cpu()
    logits = attack_model(feature.to(next(attack_model.parameters()).device))
    _, pred = torch.max(logits, 1)
    suc_rate = 1 - pred.float().mean()

    if target_was_training:
        target_model.train()
    if attack_was_training:
        attack_model.train()

    return torch.softmax(logits, dim=-1).squeeze().tolist(), suc_rate.cpu().item()

@torch.no_grad()
def member_infer_attack_node(target_model, attack_model, data, logits=None):
    '''Membership inference attack'''

    edge = data.train_pos_edge_index[:, data.df_mask]
    z = target_model(data.x, data.train_pos_edge_index[:, data.dr_mask])
    feature = torch.cat([z[edge[0]], z[edge][1]], dim=-1)
    logits = attack_model(feature)
    _, pred = torch.max(logits, 1)
    suc_rate = 1 - pred.float().mean()

    return torch.softmax(logits, dim=-1).squeeze().tolist(), suc_rate.cpu().item()

@torch.no_grad()
def get_node_embedding_data(model, data):
    model.eval()
    
    if hasattr(data, 'dtrain_mask') and data.dtrain_mask is not None:
        node_embedding = model(data.x.to(device), data.train_pos_edge_index[:, data.dtrain_mask].to(device))
    else:
        node_embedding = model(data.x.to(device), data.train_pos_edge_index.to(device))

    return node_embedding

@torch.no_grad()
def output_kldiv(model1, model2, data=None, loader=None):
    '''KL-Divergence between output distribution of model and re-trained model'''

    model1.eval()
    model2.eval()

    # For full batch
    if data is not None:
        embedding1 = get_node_embedding_data(model1, data).to(device)
        embedding2 = get_node_embedding_data(model2, data).to(device)

        if data.edge_index is not None:
            edge_index = data.edge_index.to(device)
        if data.train_pos_edge_index is not None:
            edge_index = data.train_pos_edge_index.to(device)


        if hasattr(data, 'edge_type'):
            edge_type = data.edge_type.to(device)
            score1 = model1.decode(embedding1, edge_index, edge_type)
            score2 = model2.decode(embedding2, edge_index, edge_type)
        else:
            score1 = model1.decode(embedding1, edge_index)
            score2 = model2.decode(embedding2, edge_index)

    # For mini batch
    if loader is not None:
        score1 = []
        score2 = []
        for batch in loader:
            edge_index = batch.edge_index.to(device)

            if hasattr(batch, 'edge_type'):
                edge_type = batch.edge_type.to(device)
            
                embedding1 = model1(edge, edge_type)
                embedding2 = model2(edge, edge_type)

                s1 = model.decode(embedding1, edge, edge_type)
                s2 = model.decode(embedding2, edge, edge_type)

            else:
                embedding1 = model1(edge)
                embedding2 = model2(edge)

                s1 = model.decode(embedding1, edge)
                s2 = model.decode(embedding2, edge)

            score1.append(s1)
            score2.append(s2)

        score1 = torch.hstack(score1)
        score2 = torch.hstack(score2)
    
    kldiv = F.kl_div(
        F.log_softmax(score1, dim=-1),
        F.softmax(score2, dim=-1)
    )

    return kldiv

