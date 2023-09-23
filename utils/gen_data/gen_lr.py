import argparse
import os
from os.path import join

import cv2
import numpy as np
import scipy
from scipy import ndimage
from scipy.io import loadmat, savemat


def crop_center(img, size):
    cropx = size[0]
    cropy = size[1]
    x, y = img.shape[-2], img.shape[-1]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[..., startx:startx + cropx, starty:starty + cropy]

def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)

def downsampling(img, sf, mode='cubic'):
    mode_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA
    }
    mode = mode_map[mode]
    sf = 1 / sf
    img = cv2.resize(img, (int(img.shape[1] * sf), int(img.shape[0] * sf)), interpolation=mode)
    return img


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1),
                         np.arange(-siz[0], siz[0] + 1))
    arg = -(x * x + y * y) / (2 * std * std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


def blur(img, kernel_type='gaussian'):
    downsample_type = {
        'k1': 0,
        'k2': 1,
        'k3': 2,
        'k4': 3,
        'gaussian': fspecial_gaussian(hsize=8, sigma=3),
    }
    kernel = loadmat('utils/transforms/kernels/f_set.mat')
    kernel = kernel['f_set']
    kernel = kernel[0, downsample_type[kernel_type]] if 'gaussian' not in kernel_type else downsample_type[kernel_type]
    img_blur = ndimage.filters.convolve(img, np.expand_dims(kernel, axis=2), mode='nearest')
    return img_blur


def add_noise(img, sigma, sf=4, datasetName='cave'):
    noise_level = {
        1 / 255.: 1,
        2.5 / 255.: 2.5,
        5 / 255.: 5,
        10 / 255.: 10,
        20 / 255.: 20,
    }

    if datasetName == 'paviau':
        if sf == 4:
            load_path = 'utils/transforms/noise/test_85_103'
        elif sf == 8:
            load_path = 'utils/transforms/noise/test_42_103'
    elif datasetName == 'wdc':
        if sf == 4:
            load_path = 'utils/transforms/noise/test_76_191'
        elif sf == 8:
            load_path = 'utils/transforms/noise/test_38_191'
    elif datasetName == 'salinas':
        if sf == 4:
            load_path = 'utils/transforms/noise/test_32_224'
        elif sf == 8:
            load_path = 'utils/transforms/noise/test_16_224'
    elif datasetName == 'ksc':
        if sf == 4:
            load_path = 'utils/transforms/noise/test_60_176'
        elif sf == 8:
            load_path = 'utils/transforms/noise/test_60_176'
    elif datasetName == 'botswana':
        if sf == 4:
            load_path = 'utils/transforms/noise/test_60_145'
        elif sf == 8:
            load_path = 'utils/transforms/noise/test_60_145'
    elif datasetName == 'houston18':
        if sf == 4:
            load_path = 'utils/transforms/noise/test_128_48'
        elif sf == 8:
            load_path = 'utils/transforms/noise/test_85_144'
    elif datasetName == 'houston13':
        if sf == 4:
            load_path = 'utils/transforms/noise/test_85_144'
        elif sf == 8:
            load_path = 'utils/transforms/noise/test_85_144'
    else:
        if sf == 4:
            load_path = 'utils/transforms/noise/test_128_31'
        elif sf == 8:
            load_path = 'utils/transforms/noise/test_64_31'

    print(os.path.join(load_path, 'sigma' + str(noise_level[sigma]) + '.npy'))
    noise = np.load(os.path.join(load_path, 'sigma' + str(noise_level[sigma]) + '.npy'))
    img = img + noise
    img = np.clip(img, 0, 1)

    return img


def get_option(description=''):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--sigma', type=float, default=10, help='Noise Level.')
    parser.add_argument('--kernel', '-k', type=str, default='k1', help='Kernel Type.')
    parser.add_argument('--sf', type=int, default=4, help='Scale Factor.')
    parser.add_argument('--datasetName', '-dn', type=str, default='cave', help='Dataset Name.')

    opt = parser.parse_args()
    return opt


def main():
    opt = get_option()
    # paviau path, please change
    path = '/home/liangzhiyuan/Data/remote/paviau/test_512_norm'
    fns = os.listdir(path)
    # save LR paviau path, please change
    save_path = '/home/liangzhiyuan/Data/remote/paviau/lr/sigma'+str(int(opt.sigma))+'_'+opt.kernel+'_x'+str(opt.sf)  # paviau
    os.makedirs(save_path, exist_ok=True)
    for fn in fns:
        gt = loadmat(join(path, fn))['gt']
        lr = gt
        if opt.datasetName == 'paviau' and opt.sf == 8:  # for paviau and x8 case
            lr = crop_center(lr, (336, 336))
        lr = blur(gt, opt.kernel)
        lr = downsampling(lr, opt.sf)
        lr = add_noise(lr, opt.sigma / 255, opt.sf, opt.datasetName)
        print('lr =', lr.shape)
        savemat(join(save_path, fn), {'gt': gt, 'lr': lr})


if __name__ == '__main__':
    main()
