import os
from os.path import join

import models
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid
from utils.helper import (format_nums, get_summary_writer, init_params,
                          progress_bar)
from utils.metric import PSNR, SAM, SSIM, MetricTracker


class Engine():
    def __init__(self, opt) -> None:
        self.opt = opt
        self._setup()

    def _setup(self):
        self.epoch = 0
        self.iteration = 0
        self.best_loss = 1e6
        self.best_psnr = 0
        self.basedir = join('logs/checkpoint', self.opt.arch, self.opt.prefix)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        #-------------------- Define Logger --------------------#
        self.log = not self.opt.no_log
        if self.log:
            self.writer = get_summary_writer(self.opt.arch, self.opt.prefix)

        self.metric_fns = {'PSNR': PSNR, 'SSIM': SSIM, 'SAM': SAM}
        self.tracker = MetricTracker()

        #-------------------- Define CUDA --------------------#
        self.cuda = not self.opt.no_cuda
        print('==> Cuda Acess: {}'.format(self.cuda))
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(self.opt.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.opt.seed)

        #-------------------- Define Model --------------------#
        print("==> Creating model '{}'".format(self.opt.arch))
        self.net = models.__dict__[self.opt.arch]()
        # print(self.net)
        # init_params(self.net)

        # TO_DO: varify
        if len(self.opt.gpu_ids) > 1:
            import torch.nn.parallel as parallel
            self.net = parallel.data_parallel(self.net, device_ids=self.opt.gpu_ids)
        if self.cuda:
            self.net.cuda()
        #-------------------- Define Loss Function --------------------#
        self.criterion = nn.L1Loss()

        #-------------------- Define Optimizer --------------------#
        if 'blind0' in self.opt.prefix:
            kernel_params = list(map(id, self.net.degrade.parameters()))
            base_params = filter(lambda x: id(x) not in kernel_params, self.net.parameters())
            self.optimizer = optim.AdamW([
                {'params': base_params, 'lr': self.opt.lr},
                {'params': self.net.degrade.parameters(), 'lr': self.opt.lr*0.1, 'weight_decay': 5e-4},
            ])
        else:
            self.optimizer = optim.AdamW(self.net.parameters(), lr=self.opt.lr)

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.1,
                                           patience=20, min_lr=self.opt.min_lr, verbose=True)

        #-------------------- Resume Previous Model --------------------#
        if self.opt.resume:
            self.load_checkpoint(self.opt.resumePath)
        else:
            print("==> Building model..")

    def load_checkpoint(self, resumePath):
        model_best_path = join(self.basedir, 'model_best.pth')
        if os.path.exists(model_best_path):
            best_model = torch.load(model_best_path)
            self.best_psnr = best_model['psnr']
            self.best_loss = best_model['loss']

        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('logs/checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath or model_best_path)
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_checkpoint(self, psnr, loss, model_out_path=None):
        if not os.path.isdir(join(self.basedir)):
            os.makedirs(join(self.basedir))
        if not model_out_path:
            model_out_path = join(self.basedir, "model_epoch_%d_%d.pth" % (self.epoch, self.iteration))
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'psnr': psnr,
            'loss': loss,
            'epoch': self.epoch,
            'iteration': self.iteration,
        }

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self, loader):
        print('\nEpoch:', self.epoch + 1)
        self.net.train()

        pbar = progress_bar('tqdm', len(loader))
        self.tracker.reset()

        if self.epoch < 50:
            t = 2
        else:
            t = 1

        for batch_idx, (lr, sr, gt) in enumerate(loader):
            if self.cuda:
                lr, sr, gt = lr.cuda(), sr.cuda(), gt.cuda()

            self.optimizer.zero_grad()
            output_lr, output_sr, loss_criterion = self.net(lr, sr, t)

            loss = self.criterion(lr, output_lr) + self.criterion(sr, output_sr) + loss_criterion

            loss.backward()
            
            self.optimizer.step()
            self.iteration += 1
            
            for name, fn in self.metric_fns.items():
                self.tracker.update(name, fn(output_sr, gt))
            
            self.tracker.update('Loss', loss.detach().cpu().item())

            pbar.set_postfix(format_nums(self.tracker.result()))
            pbar.update()

        pbar.close()

        # avg_psnr = self.tracker.avg('PSNR')
        # avg_ssim = self.tracker.avg('SSIM')
        avg_sam = self.tracker.avg('SAM')
        if self.log:
            self.writer.add_scalar(join(self.opt.prefix, 'SAM_epoch'), avg_sam, self.epoch)
            # self.writer.add_scalar(join(self.opt.prefix, 'train_psnr_epoch'), avg_psnr, self.epoch)


    def validate(self, loader, log=None):
        self.net.eval()

        pbar = progress_bar('tqdm', len(loader))
        self.tracker.reset()
        
        t = 1

        for batch_idx, (lr, sr, gt) in enumerate(loader):
            if self.cuda:
                lr, sr, gt = lr.cuda(), sr.cuda(), gt.cuda()

            with torch.no_grad():
                output_lr, output_sr, loss_criterion = self.net(lr, sr, t)

            loss = self.criterion(lr, output_lr) + self.criterion(sr, output_sr) + loss_criterion
            
            for name, fn in self.metric_fns.items():
                self.tracker.update(name, fn(output_sr, gt))
            
            self.tracker.update('Loss', loss.detach().cpu().item())

            pbar.set_postfix(format_nums(self.tracker.result()))
            pbar.update()
        
        if self.log:
            log.info(format_nums({'epoch:': self.epoch+1}))
            log.info(format_nums(self.tracker.result()))

        pbar.close()
        
        #-------------------- Save Checkpoint --------------------#
        avg_psnr = self.tracker.avg('PSNR')
        avg_loss = self.tracker.avg('Loss')
        if avg_loss < self.best_loss:
            print('Best Result Saving...')
            model_best_path = join(self.basedir, 'model_best.pth')
            self.save_checkpoint(psnr=avg_psnr, loss=avg_loss, model_out_path=model_best_path)
            self.best_psnr = avg_psnr
            self.best_loss = avg_loss

        self.epoch += 1
        if self.log:
            if self.epoch % self.opt.ri == 0:
                if not self.net.use_2dconv:
                    sr = torch.squeeze(sr, dim=1)
                    output_sr = torch.squeeze(output_sr, dim=1)
                show_channel = 0 if sr.shape[1] == 1 else 20
                self.writer.add_image(join(self.opt.prefix,'input_image'), make_grid(sr[:, show_channel:show_channel+1,:,:].clamp(0,1)), self.iteration, dataformats='CHW')
                self.writer.add_image(join(self.opt.prefix,'output_image'), make_grid(output_sr[:, show_channel:show_channel+1,:,:].clamp(0,1)), self.iteration, dataformats='CHW')

        if self.log:
            self.writer.add_scalar(join(self.opt.prefix, 'train_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(join(self.opt.prefix, 'train_psnr_epoch'), avg_psnr, self.epoch)
        
        return avg_loss, avg_psnr
