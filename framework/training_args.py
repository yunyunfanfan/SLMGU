import argparse
import sys


num_edge_type_mapping = {
    'FB15k-237': 237,
    'WordNet18': 18,
    'WordNet18RR': 11,
    'YAGO3-10': 37,
    'codex-s': 42,
    'codex-m': 51,
    'codex-l': 69,
    'ogbl-biokg': 51,
    # DBLP heterogeneous graph from node.dat/link.dat: 6 directed relation types.
    'DBLP': 6,
    # Locally prepared heterogeneous/KG-style datasets.
    'NELL-995': 12,
    'Hetionet': 25,
    'MovieLens-1M': 5,
    # LastFM: user-artist(0), user-user(1), artist-tag(2)
    'LastFM': 3,
    # OGB-MAG: author-institution(0), author-paper(1), paper-paper(2), paper-fos(3)
    'OGB-MAG': 4
}

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument('--unlearning_model', type=str, default='retrain',
                        help='unlearning method')
    parser.add_argument('--gnn', type=str, default='gcn', 
                        help='GNN architecture')
    parser.add_argument('--in_dim', type=int, default=128, 
                        help='input dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, 
                        help='hidden dimension')
    parser.add_argument('--out_dim', type=int, default=64, 
                        help='output dimension')
    parser.add_argument('--hetero_heads', type=int, default=2,
                        help='attention heads for HGT/HAN KG backbones')
    parser.add_argument('--hetero_dropout', type=float, default=0.0,
                        help='dropout for HAN KG backbone')
    parser.add_argument('--compgcn_opn', type=str, default='mult', choices=['mult', 'sub', 'corr'],
                        help='composition operator for CompGCN KG backbone')
    parser.add_argument('--compgcn_dropout', type=float, default=0.0,
                        help='dropout for CompGCN KG backbone')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='data dir')
    parser.add_argument('--df', type=str, default='none',
                        help='Df set to use')
    parser.add_argument('--df_idx', type=str, default='none',
                        help='indices of data to be deleted')
    parser.add_argument('--df_size', type=float, default=0.5,
                        help='Df size')
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='dataset')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=8192, 
                        help='batch size for GraphSAINTRandomWalk sampler')
    parser.add_argument('--walk_length', type=int, default=2,
                        help='random walk length for GraphSAINTRandomWalk sampler')
    parser.add_argument('--num_steps', type=int, default=32,
                        help='number of steps for GraphSAINTRandomWalk sampler')

    # Training
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, 
                        help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', 
                        help='optimizer to use')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of epochs to train')
    parser.add_argument('--valid_freq', type=int, default=100,
                        help='# of epochs to do validation')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                        help='checkpoint folder')
    parser.add_argument('--output_dir', type=str, default='./outputs/results',
                        help='directory for evaluation result json files')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha in loss function')
    parser.add_argument('--neg_sample_random', type=str, default='non_connected',
                        help='type of negative samples for randomness')
    parser.add_argument('--loss_fct', type=str, default='mse_mean',
                        choices=['mse', 'mse_mean', 'mse_sum',
                                 'kld', 'kld_mean', 'kld_sum',
                                 'cosine', 'cosine_mean', 'cosine_sum',
                                 'linear_cka', 'rbf_cka'],
                        help='loss function; mse/kld/cosine are aliases for *_mean')
    parser.add_argument('--loss_type', type=str, default='both_layerwise',
                        help='type of loss. one of {both_all, both_layerwise, only2_layerwise, only2_all, only1}')
    parser.add_argument('--loss_r_type', type=str, default='embedding_mse',
                        choices=['embedding_mse', 'preference', 'preference_l2', 'bce_l2', 'rank_l2', 'bce_rank_l2', 'dist_l2', 'sorted_dist_l2'],
                        help='forget-side loss for GNNDelete randomness objective')
    parser.add_argument('--loss_r_margin', type=float, default=0.0,
                        help='margin for preference loss: deleted edge score should be lower than negative edge score by this value')
    parser.add_argument('--loss_r_beta', type=float, default=1.0,
                        help='temperature scale for preference loss')
    parser.add_argument('--rank_same_relation', action='store_true', default=False,
                        help='rank loss only compares deleted edges against retained edges of the same relation type')

    # Pluggable deletion operator for KG/heterogeneous GNNDelete.
    parser.add_argument('--deletion_operator', type=str, default='original',
                        choices=['original', 'scgu', 'scgu_lowrank', 'lowrank', 'relation_lora_moe', 'lora_moe', 'scgu_moe', 'plain_moe'],
                        help='deletion operator: original shared matrix, SCGU low-rank subspace operator, or relation-driven LoRA-MoE')
    parser.add_argument('--scgu_rank', type=int, default=16,
                        help='rank k of the SCGU low-rank deletion subspace')
    parser.add_argument('--scgu_init', type=str, default='random',
                        choices=['random', 'zero'],
                        help='initialization for SCGU relation-specific B matrices')
    parser.add_argument('--del_moe_num_experts', type=int, default=4,
                        help='number of LoRA experts for relation_lora_moe deletion')
    parser.add_argument('--del_lora_rank', type=int, default=8,
                        help='shared LoRA rank for relation_lora_moe deletion')
    parser.add_argument('--del_gate_emb_dim', type=int, default=16,
                        help='relation embedding dimension used by the deletion gate')
    parser.add_argument('--del_lora_alpha', type=float, default=1.0,
                        help='LoRA scaling alpha; effective scale is alpha / rank')
    parser.add_argument('--del_lora_dropout', type=float, default=0.0,
                        help='dropout before the shared LoRA A projection')
    parser.add_argument('--del_lora_base_mode', type=str, default='deletion',
                        choices=['deletion', 'identity'],
                        help='relation_lora_moe base: deletion trains full deletion_weight; identity uses h + LoRA delta and has no full deletion_weight parameter')
    parser.add_argument('--del_gate_mode', type=str, default='soft',
                        choices=['soft', 'hard'],
                        help='soft: relation-weighted expert mixture; hard: straight-through one-hot expert routing')
    parser.add_argument('--del_gate_temperature', type=float, default=1.0,
                        help='temperature for relation gate softmax')
    parser.add_argument('--plot_expert_heatmap', action='store_true', default=False,
                        help='save relation x expert gate heatmaps for MoE deletion operators')
    parser.add_argument('--expert_heatmap_freq', type=int, default=0,
                        help='if >0, save expert heatmaps every N epochs during training; final heatmap is always saved when --plot_expert_heatmap is set')
    parser.add_argument('--log_train_every_epoch', action='store_true', default=False,
                        help='append train loss to trainer_log every epoch without requiring validation')
    parser.add_argument('--log_train_steps', action='store_true', default=False,
                        help='append every mini-batch train loss with global_step to trainer_log')

    # GraphEraser
    parser.add_argument('--num_clusters', type=int, default=10, 
                        help='top k for evaluation')
    parser.add_argument('--kmeans_max_iters', type=int, default=1, 
                        help='top k for evaluation')
    parser.add_argument('--shard_size_delta', type=float, default=0.005)
    parser.add_argument('--terminate_delta', type=int, default=0)

    # GraphEditor
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)

    parser.add_argument('--num_remove_links', type=int, default=11)
    parser.add_argument('--parallel_unlearning', type=int, default=4)

    parser.add_argument('--lam', type=float, default=0)
    parser.add_argument('--regen_feats', action='store_true')
    parser.add_argument('--regen_neighbors', action='store_true')
    parser.add_argument('--regen_links', action='store_true')
    parser.add_argument('--regen_subgraphs', action='store_true')
    parser.add_argument('--hop_neighbors', type=int, default=20)


    # Evaluation
    parser.add_argument('--topk', type=int, default=500, 
                        help='top k for evaluation')
    parser.add_argument('--eval_on_cpu', type=bool, default=False, 
                        help='whether to evaluate on CPU')
    parser.add_argument('--sample_eval_edges', type=int, default=0,
                        help='for KG eval, sample this many val/test positive edges and use their induced edges for message passing; 0 means full eval')
    parser.add_argument('--sample_eval_hops', type=int, default=1,
                        help='number of message-passing hops for KG sampled eval subgraph')
    parser.add_argument('--sample_eval_msg_edges', type=int, default=200000,
                        help='cap message-passing edges in ogbl-biokg sampled eval; 0 means no cap')

    # KG
    parser.add_argument('--num_edge_type', type=int, default=None, 
                        help='number of edges types')

    args = parser.parse_args()
    user_set_epochs = any(
        arg == '--epochs' or arg.startswith('--epochs=')
        for arg in sys.argv[1:]
    )
    user_set_batch_size = any(
        arg == '--batch_size' or arg.startswith('--batch_size=')
        for arg in sys.argv[1:]
    )
    user_set_lr = any(
        arg == '--lr' or arg.startswith('--lr=')
        for arg in sys.argv[1:]
    )
    user_set_weight_decay = any(
        arg == '--weight_decay' or arg.startswith('--weight_decay=')
        for arg in sys.argv[1:]
    )
    user_set_valid_freq = any(
        arg == '--valid_freq' or arg.startswith('--valid_freq=')
        for arg in sys.argv[1:]
    )
    user_set_eval_steps = any(
        arg == '--eval_steps' or arg.startswith('--eval_steps=')
        for arg in sys.argv[1:]
    )

    def set_epochs(value):
        if not user_set_epochs:
            args.epochs = value

    def set_batch_size(value):
        if not user_set_batch_size:
            args.batch_size = value

    def set_lr(value):
        if not user_set_lr:
            args.lr = value

    def set_weight_decay(value):
        if not user_set_weight_decay:
            args.weight_decay = value

    def set_valid_freq(value):
        if not user_set_valid_freq and not user_set_eval_steps:
            args.valid_freq = value

    if 'ogbl' in args.dataset:
        args.eval_on_cpu = True

    # For KG
    if args.gnn in ['rgcn', 'rgat', 'compgcn', 'hgt', 'han']:
        set_lr(1e-3)
        set_epochs(3000)
        set_valid_freq(500)
        set_batch_size(args.batch_size // 2)
        args.num_edge_type = num_edge_type_mapping[args.dataset]
        args.eval_on_cpu = True
        # args.in_dim = 512
        # args.hidden_dim = 256
        # args.out_dim = 128

    if args.unlearning_model in ['original', 'retrain']:
        set_epochs(2000)
        set_valid_freq(500)
        
        # For large graphs
        if args.gnn not in ['rgcn', 'rgat', 'compgcn', 'hgt', 'han'] and 'ogbl' in args.dataset:
            set_epochs(600)
            set_valid_freq(200)
        if args.gnn in ['rgcn', 'rgat', 'compgcn', 'hgt', 'han'] and 'ogbl' in args.dataset:
            set_batch_size(1024)

    if 'gnndelete' in args.unlearning_model:
        if args.gnn not in ['rgcn', 'rgat', 'compgcn', 'hgt', 'han'] and 'ogbl' in args.dataset:
            set_epochs(600)
            set_valid_freq(100)
        if args.gnn in ['rgcn', 'rgat', 'compgcn', 'hgt', 'han']:
            set_epochs(50)
            if args.dataset == 'WordNet18':
                set_valid_freq(2)
                set_batch_size(1024)
            if args.dataset == 'ogbl-biokg':
                set_valid_freq(10)
                set_batch_size(64)

    elif args.unlearning_model == 'gradient_ascent':
        set_epochs(10)
        set_valid_freq(1)
    
    elif args.unlearning_model == 'descent_to_delete':
        set_epochs(1)

    elif args.unlearning_model == 'graph_editor':
        set_epochs(400)
        set_valid_freq(200)


    if args.dataset == 'ogbg-molhiv':
        set_epochs(100)
        set_valid_freq(5)

    if 'gnndelete' in args.unlearning_model and user_set_eval_steps and not user_set_valid_freq:
        args.valid_freq = args.eval_steps

    return args
