import os
from os.path import join

import models
import torch
from torch import nn, optim
from torch.autograd import Variable
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

        self.metric_fns = {'PSNR': PSNR}
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

        # TO_DO: verify
        if len(self.opt.gpu_ids) > 1:
            import torch.nn.parallel as parallel
            self.net = parallel.data_parallel(self.net, device_ids=self.opt.gpu_ids)
        if self.cuda:
            self.net.cuda()
        #-------------------- Define Loss Function --------------------#
        self.criterion = nn.L1Loss()

        #-------------------- Define Optimizer --------------------#
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.opt.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5,
                                           patience=10, min_lr=self.opt.min_lr, verbose=True)

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

    def train(self, train_loader):
        print('\nEpoch:', self.epoch + 1)
        self.net.train()
        avg_psnr, avg_loss = self._step(train=True, data_loader=train_loader)

        if self.log:
            self.writer.add_scalar(join(self.opt.prefix, 'train_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(join(self.opt.prefix, 'train_psnr_epoch'), avg_psnr, self.epoch)

    def validate(self, val_loader):
        self.net.eval()
        avg_psnr, avg_loss = self._step(train=False, data_loader=val_loader)

        if self.log:
            self.writer.add_scalar(join(self.opt.prefix, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(join(self.opt.prefix,'val_psnr_epoch'), avg_psnr, self.epoch)

        #-------------------- Save Checkpoint --------------------#
        if avg_loss < self.best_loss:
            print('Best Result Saving...')
            model_best_path = join(self.basedir, 'model_best.pth')
            self.save_checkpoint(psnr=avg_psnr, loss=avg_loss, model_out_path=model_best_path)
            self.best_psnr = avg_psnr
            self.best_loss = avg_loss

        return avg_psnr, avg_loss

    def test(self, test_loader):
        self.net.eval()
        avg_psnr, avg_loss = self._step(train=False, data_loader=test_loader)

        if self.log:
            self.writer.add_scalar(join(self.opt.prefix, 'test_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(join(self.opt.prefix,'test_psnr_epoch'), avg_psnr, self.epoch)

    def _step(self, train, data_loader):
        pbar = progress_bar('tqdm', len(data_loader))
        self.tracker.reset()

        for batch_idx, (input, target) in enumerate(data_loader):
            if self.cuda:
                input, target = input.cuda(), target.cuda()

            if train:
                self.optimizer.zero_grad()

            # Bandwise
            if self.opt.bandwise:
                outs = []
                dim = 1 if self.net.use_2dconv else 2

                localFeats = []
                i = 0
                for inp, tar in zip(input.split(1, dim=dim), target.split(1, dim=dim)):
                    tar = Variable(tar, requires_grad=False)
                    if 'sfcsr' in self.opt.arch:
                        if i == 0:
                            x = input[:,0:3,:,:]         
                            y = input[:,0,:,:]
                        elif i == input.shape[1]-1:                 
                            x = input[:,i-2:i+1,:,:]                	   
                            y = input[:,i,:,:]
                        else:
                            x = input[:,i-1:i+2,:,:]                	
                            y = input[:,i,:,:]

                        if train:
                            out, localFeats = self.net(x, y, localFeats, i)
                        else:
                            with torch.no_grad():
                                out, localFeats = self.net(x, y, localFeats, i)
                        i += 1
                        localFeats.detach_()
                        localFeats = localFeats.detach()
                        localFeats = Variable(localFeats.data, requires_grad=False)
                    else:
                        if train:
                            out = self.net(inp)
                        else:
                            with torch.no_grad():
                                out = self.net(inp)
                    outs.append(out)
                    loss = self.criterion(out, tar)
                    if train:
                        loss.backward()
                output = torch.cat(outs, dim=dim)
            else:
                if train:
                    output = self.net(input)
                else:
                    with torch.no_grad():
                        output = self.net(input)

                loss = self.criterion(output, target)
                if train:
                    loss.backward()
            
            if train:
                self.optimizer.step()
                self.iteration += 1
            
            
            for name, fn in self.metric_fns.items():
                self.tracker.update(name, fn(output, target))
            
            self.tracker.update('Loss', loss.detach().cpu().item())
            
            pbar.set_postfix(format_nums(self.tracker.result()))
            pbar.update()

        pbar.close()

        if train:
            self.epoch += 1
            if self.log:
                if self.epoch % self.opt.ri == 0:
                    if not self.net.use_2dconv:
                        input = torch.squeeze(input, dim=1)
                        output = torch.squeeze(output, dim=1)
                    show_channel = 0 if input.shape[1] == 1 else 20
                    self.writer.add_image(join(self.opt.prefix,'input_image'), make_grid(input[:,show_channel:show_channel+1,:,:].clamp(0,1)), self.iteration, dataformats='CHW')
                    self.writer.add_image(join(self.opt.prefix,'output_image'), make_grid(output[:,show_channel:show_channel+1,:,:].clamp(0,1)), self.iteration, dataformats='CHW')

        avg_psnr = self.tracker.avg('PSNR')
        avg_loss = self.tracker.avg('Loss')
        
        return avg_psnr, avg_loss
