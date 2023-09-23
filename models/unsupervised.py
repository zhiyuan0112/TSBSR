import torch
import torch.nn.functional as F
from torch import nn

from utils.transforms.degrade import fspecial_gaussian
from .unet_parts import *


class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))
        self.loss = nn.L1Loss()

    def get_target_tensor(self, input):
        target_tensor = self.one

        return target_tensor.expand_as(input)

    def __call__(self, input):
        input = torch.sum(input, 1)
        target_tensor = self.get_target_tensor(input)
        loss = self.loss(input, target_tensor)

        return loss


def kl_divergence(p, q):
    p = F.softmax(p)
    q = F.softmax(q)
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))

    return s1 + s2


class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()
        self.register_buffer('zero', torch.tensor(0.01, dtype=torch.float))

    def __call__(self, input):
        input = torch.sum(input, 0, keepdim=True)
        target_zero = self.zero.expand_as(input)
        loss = kl_divergence(target_zero, input)
        return loss


class Blind_Degrade(nn.Module):
    def __init__(self):
        super(Blind_Degrade, self).__init__()
        kernel = fspecial_gaussian(hsize=8, sigma=3).astype('float32')
        kernel = torch.from_numpy(kernel).cuda()
        self.kernel = nn.Parameter(kernel, requires_grad=True)
        self.relu = nn.ReLU()

    def blur(self, img):
        # ===== Padding ===== #
        pad_img = F.pad(img, pad=(3, 4, 3, 4), mode='replicate')    # (bs,c,h,w)
        # ===== Conv ===== #
        c = img.shape[1]
        weights = self.relu(self.kernel.repeat(c, 1, 1, 1))
        img_blur = F.conv2d(pad_img, weights, groups=c)
        return img_blur

    def downsmapler(self, img, sf=4):
        return F.interpolate(img, scale_factor=1. / sf, mode='bicubic', align_corners=True)

    def forward(self, x):
        return self.downsmapler(self.blur(x))


class Conv_ReLU_Block(nn.Module):
    def __init__(self, n_feats):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(n_feats, n_feats, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.conv(x)) + x


class my_Lr2Delta_1(nn.Module):
    def __init__(self, input_c, ngf=64):
        super(my_Lr2Delta_1, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, ngf * 2, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 2, 1, 1, 0),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, num_channels, num_endmember, n_feats):
        super(CNN, self).__init__()
        self.num_channels = num_channels
        self.num_endmember = num_endmember
        self.n_feats = n_feats

        self.input = nn.Conv2d(self.num_channels, self.n_feats, 3, stride=1, padding=1)
        self.cnn_layer = self.make_layer(Conv_ReLU_Block, 3)
        self.output = my_Lr2Delta_1(input_c=self.n_feats)

        self.U = nn.Parameter(torch.rand(
            [self.num_endmember, self.num_channels]), requires_grad=True)
        self.relu = nn.ReLU(False)

        self.criterionSumToOne = SumToOneLoss()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(self.n_feats))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x = self.input(x)
        x = self.cnn_layer(x)
        w_lr = self.output(x)    # bs,num_endmember,h/4,w/4
        out_lr = torch.einsum('ijkl,jm->imkl', w_lr, self.relu(self.U))

        y = self.input(y)
        y = self.cnn_layer(y)
        w_hr = self.output(y)    # bs,num_endmember,h,w
        out_hr = torch.einsum('ijkl,jm->imkl', w_hr, self.relu(self.U))

        loss = self.criterionSumToOne(w_lr) + self.criterionSumToOne(w_hr)
        return out_lr, out_hr, loss


class CNN_Blind(CNN):
    def __init__(self, num_channels, num_endmember, n_feats):
        super(CNN_Blind, self).__init__(num_channels, num_endmember, n_feats)

        # self.degrade = Degrade()
        self.degrade = Blind_Degrade()
        self.degradeLoss = nn.L1Loss()

    def forward(self, x, y, t=0):
        input = x

        x = self.input(x)
        x = self.cnn_layer(x)
        w_lr = self.output(x)    # bs,num_endmember,h/4,w/4
        out_lr = torch.einsum('ijkl,jm->imkl', w_lr, self.relu(self.U))

        y = self.input(y)
        y = self.cnn_layer(y)
        w_hr = self.output(y)    # bs,num_endmember,h,w
        out_hr = torch.einsum('ijkl,jm->imkl', w_hr, self.relu(self.U))

        degrade_input = self.degrade(out_hr)

        loss = self.criterionSumToOne(w_lr) +\
            self.criterionSumToOne(w_hr) +\
            self.degradeLoss(input, degrade_input)

        return out_lr, out_hr, loss


class CNN_Blind_Sparse(CNN_Blind):
    def __init__(self, num_channels, num_endmember, n_feats):
        super(CNN_Blind_Sparse, self).__init__(num_channels, num_endmember, n_feats)
        self.sparse = SparseKLloss()

    def forward(self, x, y, t):
        input = x

        x = self.input(x)
        x = self.cnn_layer(x)
        w_lr = self.output(x)    # bs,num_endmember,h/4,w/4
        out_lr = torch.einsum('ijkl,jm->imkl', w_lr, self.relu(self.U))

        y = self.input(y)
        y = self.cnn_layer(y)
        w_hr = self.output(y)    # bs,num_endmember,h,w
        out_hr = torch.einsum('ijkl,jm->imkl', w_hr, self.relu(self.U))

        degrade_input = self.degrade(out_hr)

        loss = self.criterionSumToOne(w_lr) +\
            self.criterionSumToOne(w_hr) +\
            t * self.degradeLoss(input, degrade_input) +\
            1e-3 * self.sparse(w_lr) + 1e-3 * self.sparse(w_hr)
        
        return out_lr, out_hr, loss



if __name__ == '__main__':
    from torch.autograd import Variable
    lr_data = Variable(torch.randn(1, 3, 4, 4)).cuda()
    hr_data = Variable(torch.randn(1, 3, 16, 16)).cuda()
    net = CNN(num_channels=3, num_endmember=10, n_feats=8)
    net.cuda()
    out = net(lr_data, hr_data)
    # print(out.shape)
