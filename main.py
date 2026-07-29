import argparse
import runpy
import sys
from pathlib import Path

import yaml


def _config_to_argv(config):
    argv = []
    for key, value in config.items():
        if value is None:
            continue
        flag = '--' + key
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv.extend([flag, ','.join(map(str, value))])
        else:
            argv.extend([flag, str(value)])
    return argv


def main():
    parser = argparse.ArgumentParser(description='Unified entry point for SLMGU/GNNDelete experiments.')
    parser.add_argument('--config', type=str, required=True, help='Path to a YAML config file.')
    parser.add_argument('--mode', type=str, default=None, choices=['train', 'delete', 'evaluate'],
                        help='Override mode in the config file.')
    args, extra = parser.parse_known_args()

    cfg_path = Path(args.config)
    with cfg_path.open('r') as f:
        cfg = yaml.safe_load(f) or {}
    cfg_mode = cfg.pop('mode', 'train')
    mode = args.mode or cfg_mode

    module = {
        'train': 'train_gnn',
        'delete': 'delete_gnn',
        'evaluate': 'evaluate_gnn',
    }[mode]

    sys.argv = [module + '.py'] + _config_to_argv(cfg) + extra
    runpy.run_module(module, run_name='__main__')


if __name__ == '__main__':
    main()
