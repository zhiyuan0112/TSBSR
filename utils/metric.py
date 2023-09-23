from functools import partial

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-3]
        bwindex = []
        for ch in range(C):
            x = torch.squeeze(X[..., ch, :, :].data).cpu().numpy()
            y = torch.squeeze(Y[..., ch, :, :].data).cpu().numpy()
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex


cal_bwpsnr = Bandwise(partial(peak_signal_noise_ratio, data_range=1))
cal_bwssim = Bandwise(structural_similarity)


def cal_sam(X, Y, eps=1e-8):
    X = torch.squeeze(X.data).cpu().numpy()
    Y = torch.squeeze(Y.data).cpu().numpy()
    tmp = (np.sum(X*Y, axis=0) + eps) / (np.sqrt(np.sum(X**2, axis=0)) + eps) / (np.sqrt(np.sum(Y**2, axis=0)) + eps)
    return np.mean(np.real(np.arccos(tmp.clip(-1,1))))

# def MSIQA(X, Y):
#     psnr = np.mean(cal_bwpsnr(X, Y))
#     ssim = np.mean(cal_bwssim(X, Y))
#     sam = cal_sam(X, Y)
#     return psnr, ssim, sam


def PSNR(outputs, targets):
    return np.mean(cal_bwpsnr(outputs, targets))


def SSIM(outputs, targets):
    return np.mean(cal_bwssim(outputs, targets))


def SAM(outputs, targets):
    return cal_sam(outputs, targets)


class MetricTracker:
    def __init__(self):
        self._data = {}
        self.reset()

    def reset(self):
        self._data = {}

    def update(self, key, value, n=1):
        if key not in self._data.keys():
            self._data[key] = {'total': 0, 'count': 0}
        self._data[key]['total'] += value * n
        self._data[key]['count'] += n

    def avg(self, key):
        return self._data[key]['total'] / self._data[key]['count']

    def result(self):
        return {k: self._data[k]['total'] / self._data[k]['count'] for k in self._data.keys()}

    def summary(self):
        items = ['{}: {:.8f}'.format(k, v) for k, v in self.result().items()]
        return ' '.join(items)
