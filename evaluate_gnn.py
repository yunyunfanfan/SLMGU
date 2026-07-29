import json
import math
import os
import pickle

import torch
from torch_geometric.seed import seed_everything
from torch_geometric.utils import is_undirected

from framework import get_model, get_trainer
from framework.training_args import parse_args


def _json_safe(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _resolve_checkpoint_dir(args):
    if os.path.exists(os.path.join(args.checkpoint_dir, 'model_best.pt')):
        return args.checkpoint_dir
    if args.unlearning_model == 'retrain':
        return os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'retrain',
                            '-'.join(map(str, [args.df, args.df_size, args.random_seed])))
    if 'gnndelete' in args.unlearning_model:
        raise ValueError('For gnndelete evaluation, set --checkpoint_dir to the exact directory containing model_best.pt.')
    return os.path.join(args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, str(args.random_seed))


def _prepare_original_kg_data(args):
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
        dataset, data = pickle.load(f)
    if args.gnn not in ['rgcn', 'rgat', 'compgcn', 'hgt', 'han']:
        args.in_dim = dataset.num_features
    if args.gnn in ['rgcn', 'rgat', 'compgcn', 'hgt', 'han']:
        if not hasattr(data, 'train_mask'):
            data.train_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
        r, c = data.train_pos_edge_index
        rev_edge_index = torch.stack([c, r], dim=0)
        rev_edge_type = data.train_edge_type + args.num_edge_type
        data.edge_index = torch.cat((data.train_pos_edge_index, rev_edge_index), dim=1)
        data.edge_type = torch.cat([data.train_edge_type, rev_edge_type], dim=0)
        data.dr_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
        assert is_undirected(data.edge_index)
    return dataset, data


@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(args.random_seed)
    ckpt_dir = _resolve_checkpoint_dir(args)
    args.checkpoint_dir = ckpt_dir
    print(f'Loading checkpoint from: {ckpt_dir}')

    _, data = _prepare_original_kg_data(args)
    model = get_model(args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
    ckpt = torch.load(os.path.join(ckpt_dir, 'model_best.pt'), map_location='cpu')
    model.load_state_dict(ckpt['model_state'], strict=False)

    trainer = get_trainer(args)
    loss, dt_auc, dt_aup, df_auc, df_aup, _, _, test_log = trainer.test(model, data, ckpt='ckpt')
    result = {
        'dataset': args.dataset,
        'gnn': args.gnn,
        'seed': args.random_seed,
        'checkpoint_dir': ckpt_dir,
        'dt_loss': loss,
        'dt_auc': dt_auc,
        'dt_aup': dt_aup,
        'df_auc': df_auc,
        'df_aup': df_aup,
        'test_log': test_log,
    }
    out_dir = getattr(args, 'output_dir', './outputs/results')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'{args.dataset}_{args.gnn}_{args.unlearning_model}_seed{args.random_seed}_eval.json')
    result = _json_safe(result)
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f'Saved result to: {out_file}')


if __name__ == '__main__':
    main()
