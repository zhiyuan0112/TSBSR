import os
import sys

from torch.utils.data import DataLoader
from torchlight.logging import Logger
from torchvision.transforms.transforms import Compose

from trainer.engine_transfer import Engine
from trainer.option import basic_opt
from utils.helper import adjust_learning_rate, display_learning_rate
from utils.transforms._util import SingleImageDataset
from utils.transforms.degrade import (GaussianDownsample, K1Downsample,
                                      K2Downsample, K3Downsample, K4Downsample, UniformDownsample)
from utils.transforms.noise import GaussianNoise

opt = basic_opt()
print(opt)
engine = Engine(opt)

# ---------------- Prepare Train Data ---------------- #
print('==> Preparing data..')

DownsampleType = {
    'gaussian': GaussianDownsample,
    'uniform': UniformDownsample,
    'k1': K1Downsample,
    'k2': K2Downsample,
    'k3': K3Downsample,
    'k4': K4Downsample,
}

degrade = Compose([
    DownsampleType[opt.downsample](sf=opt.sf),
    GaussianNoise(sigma=opt.noise/255., is_test=True),
])

dataset = SingleImageDataset(opt.dir, degrade=degrade)
loader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False,
                    num_workers=opt.threads, pin_memory=not opt.no_cuda)
val_dataset = SingleImageDataset(opt.dir, degrade=degrade, is_test=True)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False,
                    num_workers=opt.threads, pin_memory=not opt.no_cuda)


# ---------------- Training ---------------- #
adjust_learning_rate(engine.optimizer, opt.lr)
if not opt.no_log:
    cmd = ': python ' + ' '.join(sys.argv)
    os.makedirs(os.path.join('logs/log/train', opt.arch, opt.prefix), exist_ok=True)
    log = Logger(os.path.join('logs/log/train', opt.arch, opt.prefix))
    log.info(cmd)
    log.info(opt)

for i in range(opt.nEpochs):
    engine.train(loader)
    psnr, loss = engine.validate(val_loader) if opt.no_log else engine.validate(val_loader, log)
    engine.scheduler.step(loss)

    display_learning_rate(engine.optimizer)
    if engine.epoch % opt.ri == 0:
        engine.save_checkpoint(psnr, loss)
