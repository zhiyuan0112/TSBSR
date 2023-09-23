import argparse
import os
import sys

import torch
from scipy.io.matlab.mio import savemat
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchlight.logging import Logger
from torchnet.dataset import TransformDataset
from torchvision.transforms.transforms import Compose

import models
from utils.helper import format_nums, progress_bar
from utils.metric import PSNR, SAM, SSIM, MetricTracker
from utils.transforms._util import (HSI2Tensor, ImageTransformDataset,
                                    LoadMatKey, MatDataFromFolder)
from utils.transforms.degrade import (GaussianDownsample, K1Downsample,
                                      K2Downsample, K3Downsample, K4Downsample,
                                      Resize, UniformDownsample)
from utils.transforms.noise import GaussianNoise


def Opt():    
    parser = argparse.ArgumentParser(description='Test Mode.')
    
    model_names = sorted(name for name in models.__dict__ if name.islower()
                     and not name.startswith("__") and callable(models.__dict__[name]))

    # Model specifications
    parser.add_argument('--arch', '-a', required=True, choices=model_names,
                        help='Model architecture: ' + ' | '.join(model_names))
    parser.add_argument('--prefix', '-p', default='bicubic', help='Distingduish methods.')
    parser.add_argument('--datasetName', '-dn', default='delete', help='Distingduish Dataset.')
    parser.add_argument('--downsample', '-ds', default='gaussian', help='Choose Different Kernels in Downsampling.')
    parser.add_argument('--noise', type=float, default=10, help='Choose Different Noise Level in Downsampling.')
    parser.add_argument('--resumePath', '-rp', type=str, required=True, help='Checkpoint to use.')
    parser.add_argument('--threads', type=int, default=8, help='Number of threads for data loader.')
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

def get_loader(opt, use_2dconv):
    totensor = HSI2Tensor(use_2dconv)
    sr_degrade = Compose([
        DownsampleType[opt.downsample](sf=opt.sf),
        GaussianNoise(sigma=opt.noise/255., is_test=True),
        Resize(sf=opt.sf),
        lambda x:x.transpose(2,0,1),
        totensor
    ]) if ('bi3dqrnn' in opt.arch or 'bicubic' in opt.prefix) else Compose([
        DownsampleType[opt.downsample](sf=opt.sf),
        GaussianNoise(sigma=opt.noise/255., is_test=True),
        lambda x:x.transpose(2,0,1),
        totensor
    ])
    target_transform = Compose([
        lambda x:x.transpose(2,0,1),
        totensor
    ])

    if 'paviau' in opt.datasetName:
        test_dir = '/media/exthdd/datasets/hsi/lzy_data/remote/test_norm_paviau'
    elif 'harvard' in opt.datasetName:
        test_dir = '/media/exthdd/datasets/hsi/lzy_data/Harvard/harvard_512_10'
    elif 'wdc' in opt.datasetName:
        test_dir = '/media/exthdd/datasets/hsi/lzy_data/remote/washingtondc/test'
    elif 'salinas' in opt.datasetName:
        test_dir = '/media/exthdd/datasets/hsi/lzy_data/remote/salinas/test_128_norm'
    elif 'houston13' in opt.datasetName:
        test_dir = '/media/exthdd/datasets/hsi/lzy_data/remote/houston13/test/norm'
    elif 'houston18' in opt.datasetName:
        test_dir = '/media/exthdd/datasets/hsi/lzy_data/remote/houston18/test'
    else:
        test_dir = ''
        
    fns = os.listdir(test_dir)
    dataset = MatDataFromFolder(test_dir, size=None)  # (340,340,103)
    dataset = TransformDataset(dataset, LoadMatKey(key='gt'))
    dataset = ImageTransformDataset(dataset, sr_degrade, target_transform)

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=opt.threads, pin_memory=not opt.no_cuda)
    
    return loader, fns


def main():
    opt = Opt()
    device = torch.device('cpu') if opt.no_cuda else torch.device('cuda')
    use_2dconv = True
    
    # ---------------- Define Test Logger ---------------- #
    if not opt.no_log:
        cmd = ': python ' + ' '.join(sys.argv)
        os.makedirs('logs/log/test', exist_ok=True)
        log = Logger('logs/log/test/' + opt.datasetName)
        log.info(cmd)
        log.info(opt)

    # ---------------- Test Data ---------------- #
    test_loader, fns = get_loader(opt, use_2dconv)

    # ---------------- Define Model ---------------- #
    net = models.__dict__[opt.arch]().to(device)
    net.eval()

    # ---------------- Load Checkpoint ---------------- #
    checkpoint = torch.load(opt.resumePath)
    net.load_state_dict(checkpoint['net'])


    metric_fns = {'PSNR': PSNR, 'SSIM': SSIM, 'SAM': SAM}
    tracker = MetricTracker()

    pbar = progress_bar('tqdm', len(test_loader))
    for i, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)
            
        if 'bicubic' in opt.prefix:
            output = input
        else:
            if opt.bandwise:
                outs = []
                dim = 1 if net.use_2dconv else 2
                
                localFeats = []
                idx = 0
                for _, (inp, tar) in enumerate(zip(input.split(1, dim=dim), target.split(1, dim=dim))):
                    with torch.no_grad():
                        out = net(inp)
                    outs.append(out)
                output = torch.cat(outs, dim=dim)
            else:
                with torch.no_grad():
                    output = net(input)

        metrics = {}
        for name, fn in metric_fns.items():
            metrics[name] = fn(output, target)
            tracker.update(name, metrics[name])
        
        pbar.set_postfix(format_nums(tracker.result()))
        pbar.update()
        
        if not opt.no_log:
            log.debug(fns[i] + str(format_nums(metrics)))

        # ---------------- Save Results ---------------- #
        if not opt.no_save:  
            def to_numpy(data):
                # choose 1 sample from batch
                data = data[0] 
                # squeeze channel dim for 3D net (BS,C_in,C,H,W)
                if not use_2dconv:
                    data = data[0]
                return data.cpu().numpy().transpose(1,2,0)  
            
            if 'bicubic' in opt.prefix:
                savedir = os.path.join('logs/result', 'bicubic',
                                    opt.datasetName, opt.prefix)
            elif 'swinir' in opt.arch and opt.bandwise:
                savedir = os.path.join('logs/result/swinirY',
                                    opt.datasetName, opt.prefix)
            else:
                savedir = os.path.join('logs/result',
                                    opt.arch, opt.datasetName, opt.prefix)
            os.makedirs(savedir, exist_ok=True)
            obj = {'pred': to_numpy(output), 'gt': to_numpy(target)}
            obj.update(metrics)
            savemat(os.path.join(savedir, fns[i]), obj)

    pbar.close()

    if not opt.no_log:
        log.info(format_nums(tracker.result()))


if __name__ == '__main__':
    main()
