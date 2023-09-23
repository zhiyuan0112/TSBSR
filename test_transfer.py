import argparse
import os
import sys

import torch
from scipy.io.matlab.mio import savemat
from torch.utils.data import DataLoader
from torchlight.logging import Logger
from torchvision.transforms.transforms import Compose

import models
from utils.helper import format_nums, progress_bar
from utils.metric import PSNR, SAM, SSIM, MetricTracker
from utils.transforms._util import SingleImageDataset
from utils.transforms.degrade import (GaussianDownsample, K1Downsample,
                                      K2Downsample, K3Downsample, K4Downsample,
                                      UniformDownsample)
from utils.transforms.noise import GaussianNoise


def Opt():    
    parser = argparse.ArgumentParser(description='Test Mode.')
    
    model_names = sorted(name for name in models.__dict__ if name.islower()
                     and not name.startswith("__") and callable(models.__dict__[name]))

    # Model specifications
    parser.add_argument('--arch', '-a', required=True, choices=model_names,
                        help='Model architecture: ' + ' | '.join(model_names))
    parser.add_argument('--prefix', '-p', default='bicubic', help='Distingduish methods.')
    parser.add_argument('--dir', required=True, help='Data Dir')
    parser.add_argument('--datasetName', '-dn', default='', help='Data set name.')
    parser.add_argument('--fileName', '-fn', default='', help='Data set name.')
    parser.add_argument('--downsample', '-ds', default='gaussian', help='Choose Different Kernels in Downsampling.')
    parser.add_argument('--noise', type=float, default=10, help='Choose Different Noise Level in Downsampling.')
    parser.add_argument('--resumePath', '-rp', type=str, required=True, help='Checkpoint to use.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads for data loader.')
    parser.add_argument('--no-cuda', action='store_true', help='Disable cuda?')
    parser.add_argument('--bandwise', action='store_true', help='Test in a bandwise manner.')
    parser.add_argument('--no-save', action='store_true', help='Save Result?')
    parser.add_argument('--no-log', action='store_true', help='Save Log?')
    parser.add_argument('--sf', type=int, default=4, help='Scale Factor')

    opt = parser.parse_args()
    return opt


DownsampleType = {
    'gaussian': GaussianDownsample,
    'uniform': UniformDownsample,
    'k1': K1Downsample,
    'k2': K2Downsample,
    'k3': K3Downsample,
    'k4': K4Downsample,
}

def get_loader(opt):
    degrade = Compose([
        DownsampleType[opt.downsample](sf=opt.sf),
        GaussianNoise(sigma=opt.noise/255., is_test=True),
    ])
    datadir = os.path.join(opt.dir, opt.fileName)

    dataset = SingleImageDataset(datadir, degrade=degrade, is_test=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=opt.threads, pin_memory=not opt.no_cuda)

    return loader


def main():
    opt = Opt()
    device = torch.device('cpu') if opt.no_cuda else torch.device('cuda')
    use_2dconv = True
    
    # ---------------- Define Test Logger ---------------- #
    if not opt.no_log:
        cmd = ': python ' + ' '.join(sys.argv)
        log = Logger('logs/log/' + opt.arch)
        log.info(cmd)
        log.info(opt)

    # ---------------- Test Data ---------------- #
    test_loader = get_loader(opt)

    # ---------------- Define Model ---------------- #
    net = models.__dict__[opt.arch]().to(device)
    net.eval()

    # ---------------- Load Checkpoint ---------------- #
    checkpoint = torch.load(opt.resumePath)
    net.load_state_dict(checkpoint['net'])


    metric_fns = {'PSNR': PSNR, 'SSIM': SSIM, 'SAM': SAM}
    tracker = MetricTracker()

    pbar = progress_bar('tqdm', len(test_loader))
    for i, (lr, sr, gt) in enumerate(test_loader):
        lr, sr, gt = lr.to(device), sr.to(device), gt.to(device)

        with torch.no_grad():
            output_lr, output_sr, _ = net(lr, sr, 1)

        metrics = {}
        for name, fn in metric_fns.items():
            metrics[name] = fn(output_sr, gt)
            tracker.update(name, metrics[name])
        
        pbar.set_postfix(format_nums(tracker.result()))
        pbar.update()
        
        if not opt.no_log:
            log.debug(opt.fileName + str(format_nums(metrics)))

        # ---------------- Save Results ---------------- #
        if not opt.no_save:  
            def to_numpy(data):
                # choose 1 sample from batch
                data = data[0] 
                # squeeze channel dim for 3D net (BS,C_in,C,H,W)
                if not use_2dconv:
                    data = data[0]
                return data.cpu().numpy().transpose(1,2,0)  
            
            savedir = os.path.join('logs/result',
                                    opt.arch, opt.datasetName, opt.prefix)
            os.makedirs(savedir, exist_ok=True)
            obj = {'pred': to_numpy(output_sr), 'gt': to_numpy(gt)}
            obj.update(metrics)
            savemat(os.path.join(savedir, opt.fileName), obj)

    pbar.close()

    if not opt.no_log:
        log.info(format_nums(tracker.result()))


if __name__ == '__main__':
    main()
