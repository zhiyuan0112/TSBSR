from functools import partial

from torch.utils.data import DataLoader
from torchnet.dataset import TransformDataset
from torchvision.transforms.transforms import Compose

from trainer.engine import Engine
from trainer.option import basic_opt
from utils.helper import display_learning_rate, adjust_learning_rate
from utils.transforms._util import (HSI2Tensor, ImageTransformDataset,
                                    LMDBDataset, get_train_valid_dataset)
from utils.transforms.degrade import GaussianDownsample
from utils.transforms.noise import GaussianNoiseBlindv2

opt = basic_opt()
print(opt)
engine = Engine(opt)

# ---------------- Train and Validate Data ---------------- #
print('==> Preparing data..')
train_data = LMDBDataset(opt.dir)
train_data = TransformDataset(train_data, lambda x: x)
print("Length of train and val set: {}.".format(len(train_data)))
# ---- Split patches dataset into training, validation parts ---- #
n_val = round(len(train_data)/10)
train_data, val_data = get_train_valid_dataset(train_data, n_val)        # (c,h,w)

use_2dconv = engine.net.module.use_2dconv if len(opt.gpu_ids) > 1 else engine.net.use_2dconv
HSI2Tensor = partial(HSI2Tensor, use_2dconv=use_2dconv)

sigmas = [0, 2.5, 5, 10]
sigmas = [i/255. for i in sigmas]
sr_degrade = Compose([
    lambda x: x.transpose(1, 2, 0),  # (h,w,c)
    GaussianDownsample(sf=opt.sf),
    GaussianNoiseBlindv2(sigmas=sigmas),
    lambda x: x.transpose(2, 0, 1),  # (c,h,w)
    HSI2Tensor()
])

ImageTransformDataset = partial(ImageTransformDataset, target_transform=HSI2Tensor())
train_dataset = ImageTransformDataset(train_data, sr_degrade)
val_dataset = ImageTransformDataset(val_data, sr_degrade)

train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True,
                          num_workers=opt.threads, pin_memory=not opt.no_cuda)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                        num_workers=opt.threads, pin_memory=not opt.no_cuda)



adjust_learning_rate(engine.optimizer, opt.lr)
for i in range(opt.nEpochs):
    engine.train(train_loader)
    psnr, loss = engine.validate(val_loader)

    engine.scheduler.step(loss)
    display_learning_rate(engine.optimizer)
    if engine.epoch % opt.ri == 0:
        engine.save_checkpoint(psnr, loss)
