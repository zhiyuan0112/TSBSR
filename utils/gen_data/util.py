import math
import os
import sys

import matplotlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from transforms.functional.general import minmax_normalize


def mat2gray(path, fns, matkey, save_path):
    for fn in fns:
        file = os.path.join(path, fn)
        data = loadmat(file)[matkey]
        for i, band in enumerate(np.split(data, data.shape[-1], -1)):
            save_name = fn[:-4] + '_' + str(i) + '.mat'
            savemat(os.path.join(save_path, save_name), {matkey: band.astype(np.float64)})


def plot_errormaps(pred_path, gt_path, fn):
    pred = loadmat(os.path.join(pred_path, fn))['pred']
    gt = loadmat(os.path.join(gt_path, fn))['gt']
    h, w, c = pred.shape
    print(h, w, c)
    err = np.zeros((h, w), dtype='float32')
    for i in range(c):
        err += abs(pred[:, :, i] - gt[:, :, i])
    err = err*255/c
    err = err[::-1, :]
    # err = np.rot90(np.rot90(np.rot90(err)))

    gci = plt.imshow(err, origin='lower',
                     cmap=matplotlib.cm.jet,
                     norm=matplotlib.colors.Normalize(vmin=0, vmax=5))
    # cbar = plt.colorbar(gci, orientation='horizontal')
    plt.imsave(os.path.join(pred_path, fn[:-3]+'png'), err, origin='lower',
               cmap=matplotlib.cm.jet,
               vmin=0, vmax=5)


def gen_rbg_CRF():
    u = [600, 550, 500]
    sigma = [50*0.8, 50*1.2, 50]
    xs = []
    ys = []
    for i in range(len(u)):
        x = np.linspace(400, 700, 102)
        y = np.exp(-(x - u[i]) ** 2 / (2 * sigma[i] ** 2)) / (math.sqrt(2*math.pi)*sigma[i])
        xs.append(x)
        ys.append(y)
        color = {0: 'r', 1: 'g', 2: 'b'}
        plt.plot(x, y, color[i])
    plt.xlim((400, 700))
    plt.savefig('RGB_CRF.png')
    print(xs[0].shape, ys[0].shape)
    print(sum(ys[0]), sum(ys[1]), sum(ys[2]))
    np.savetxt('RGB_CRF', ys)


def hsi2rgb(hsi_path, fn, matkey, save_path):
    rgb_CRF = np.loadtxt('RGB_CRF')  # (3,102)
    hsi = loadmat(os.path.join(hsi_path, fn))[matkey]  # (h,w,c)
    hsi = minmax_normalize(hsi)
    print(hsi.shape)
    rgb = np.einsum('ijk,lk->ijl', hsi, rgb_CRF) * 2
    rgb = np.clip(255 * rgb, 0, 255).astype('uint8')
    print(rgb.shape)
    print(rgb[0:4, 0:4, 0])
    plt.imsave('rgb.png', rgb)

def rgb2hsi(rgb_path, hsi_path):
    # --------- Inverse HSI2RGB matrix --------- #
    matrix = np.loadtxt('RGB_CRF')
    print(matrix.shape)
    matrix_inv = np.linalg.pinv(matrix)
    print(matrix_inv.shape)
    # --------- Load RGB image --------- #
    fns = ['0677.png', '0684.png']
    fn = fns[0]
    rgb = cv2.imread(os.path.join(rgb_path, fn))  # (h,w,c=3)
    print(rgb.shape)
    hsi = np.einsum('ijk,lk->ijl', rgb, matrix_inv)
    hsi = minmax_normalize(hsi).astype('float32')
    print(hsi.dtype, hsi.shape)
    if not os.path.exists(hsi_path):
        os.makedirs(hsi_path)
    savemat(os.path.join(hsi_path, fn[:-3]+'mat'), {'pavia': hsi})  # (h,w,c=102)
    # plt.imsave('hsi.png', hsi[:,:,80], cmap='gray')

def test_3channel(hsi_path, fn, matkey):
    hsi = loadmat(os.path.join(hsi_path, fn))[matkey]
    hsi = minmax_normalize(hsi)
    index = [30, 50, 80]
    hsi_3 = hsi[:,:,index]
    hsi_3 = np.clip(255 * hsi_3, 0, 255).astype('uint8')
    print(hsi_3.shape)
    plt.imsave('hsi_3.png', hsi_3)

if __name__ == '__main__':
    pass