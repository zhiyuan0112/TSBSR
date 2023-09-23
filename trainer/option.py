import argparse

import models

model_names = sorted(name for name in models.__dict__ if name.islower()
                     and not name.startswith("__") and callable(models.__dict__[name]))


def basic_opt(description=''):
    parser = argparse.ArgumentParser(description=description)

    # Model specifications
    parser.add_argument('--arch', '-a', required=True, choices=model_names,
                        help='Model architecture: ' + ' | '.join(model_names))
    parser.add_argument('--prefix', '-p', type=str, default='icvl_100', help='Distinguish checkpoint.')

    # Training specifications
    parser.add_argument('--batchSize', '-b', type=int, default=16, help='Training batch size.')
    parser.add_argument('--nEpochs', '-n', type=int, default=100, help='Training epoch.')
    parser.add_argument('--bandwise', action='store_true', help='Able bandwise?')
    parser.add_argument('--downsample', '-ds', default='gaussian', help='Type of downsmaple')
    parser.add_argument('--noise', type=float, default=10, help='Sigma')
    parser.add_argument('--sf', type=int, default=4, help='Scale Factor')

    # Optimization specifications
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--min-lr', '-mlr', type=float, default=1e-5, help='Minimal learning rate.')

    # Checkpoint specifications
    parser.add_argument('--ri', type=int, default=50, help='Record interval.')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint.')
    parser.add_argument('--resumePath', '-rp', type=str, help='Checkpoint to use.')

    # Hardware specifications
    parser.add_argument('--threads', type=int, default=8, help='Number of threads for data loader.')
    parser.add_argument('--no-cuda', action='store_true', help='Disable cuda?')
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids.')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed to use.')    
    
    # Log specifications
    parser.add_argument('--no-log', action='store_true', help='Disable logger?')    

    # Data specifications
    parser.add_argument('--dir', type=str, default='/data1/liangzhiyuan/data/RGB/DIV2K_64_1.db', help='Training data path.')


    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)
    return opt


def _parse_str_args(args):
    str_args = args.split(',')
    parsed_args = []
    for str_arg in str_args:
        arg = int(str_arg)
        if arg >= 0:
            parsed_args.append(arg)
    return parsed_args
