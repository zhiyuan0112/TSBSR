import os
import socket
from datetime import datetime

import torch.nn as nn
import torch.nn.init as init
from qqdm import qqdm
from tensorboardX import SummaryWriter
from tqdm import tqdm


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate"""
    print('Adjust Learning Rate => %.4e' % lr)
    for i, param_group in enumerate(optimizer.param_groups):
        if i == 0:
            param_group['lr'] = lr
        elif i == 1:
            param_group['lr'] = lr * 0.1
        else:
            param_group['lr'] = lr


def display_learning_rate(optimizer):
    lrs = []
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        print('learning rate of group %d: %.4e' % (i, lr))
        lrs.append(lr)
    return lrs


def get_summary_writer(arch, prefix):
    log_dir = 'logs/runs/%s/%s/' % (arch, prefix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    return writer


def progress_bar(type, total):
    if type == 'qqdm':
        bar = qqdm
    elif type == 'tqdm':
        bar = tqdm
    else:
        raise ValueError('choice of progressbar [qqdm, tqdm]')
    return bar(total=total, dynamic_ncols=True)


def format_num(n, fmt='{0:.3g}'):
    f = fmt.format(n).replace('+0', '+').replace('-0', '-')
    n = str(n)
    return f if len(f) < len(n) else n


def format_nums(d):
    return {k: format_num(v, fmt='{:.4f}') for k, v in d.items()}


def init_params(net, bn_frozen=False):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant(m.bias, 0)
        # elif isinstance(m, nn.Conv1d):
        #     init.normal_(m.weight)
        #     if m.bias is not None:
        #         init.constant(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            if bn_frozen:
                init.constant(m.weight, 0)
            else:
                init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant(m.bias, 0)
