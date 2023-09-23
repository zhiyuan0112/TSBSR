import os

import numpy as np
from numpy.core.fromnumeric import resize
from scipy import ndimage
import scipy
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1),
                         np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h


"""
# --------------------------------------------
# anisotropic Gaussian kernels
# --------------------------------------------
"""
def analytic_kernel(k):
    """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()


class AbstractBlur:
    def __call__(self, img):
        img_L = ndimage.filters.convolve(
            img, np.expand_dims(self.kernel, axis=2), mode='nearest')
        return img_L


class GaussianBlur(AbstractBlur):
    """ Expect input'shape = [W,H,C]
    """

    def __init__(self, ksize=8, sigma=3):
        self.kernel = fspecial_gaussian(ksize, sigma)


class AnisotropicGaussianBlur(AbstractBlur):
    """ Expect input'shape = [W,H,C]
    """

    def __init__(self, ksize=8):
        self.kernel = analytic_kernel(ksize)

class UniformBlur(AbstractBlur):
    """ Expect input'shape = [W,H,C]
    """

    def __init__(self, ksize):
        self.kernel = np.ones((ksize, ksize)) / (ksize*ksize)


class KnBlur(AbstractBlur):
    """ Expect input'shape = [W,H,C]
    """

    def __init__(self, idx):
        self.kernel = loadmat('utils/transforms/kernels/f_set.mat')
        self.kernel = self.kernel['f_set']
        self.kernel = self.kernel[0,idx]
        # plt.imsave('logs/kernel/test_kernel/f_set_4.png', self.kernel, cmap='gray')
        # cv2.imwrite('kernel_real.png', self.kernel*255)
        # print(self.kernel*255)
        print(self.kernel.shape)
    


## -------------------- Downsample -------------------- ##

class KFoldDownsample:
    ''' k-fold downsampler:
        Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

        Expect input'shape = [W,H,C]
    '''

    def __init__(self, sf):
        self.sf = sf

    def __call__(self, img):
        """ input: [w,h,c] """
        st = 0
        return img[st::self.sf, st::self.sf, :]


class AbstractDownsample:
    def __call__(self, img):
        img = self.blur(img)
        img = self.downsampler(img)
        return img


class UniformDownsample(AbstractDownsample):
    """ Expect input'shape = [W,H,C]
    """

    def __init__(self, sf):
        self.sf = sf
        self.blur = UniformBlur(sf)
        self.downsampler = Resize(1/sf)


class K1Downsample(AbstractDownsample):
    """ Expect input'shape = [W,H,C]
    """

    def __init__(self, sf):
        self.sf = sf
        self.blur = KnBlur(0)
        self.downsampler = Resize(1/sf)

class K2Downsample(AbstractDownsample):
    """ Expect input'shape = [W,H,C]
    """

    def __init__(self, sf):
        self.sf = sf
        self.blur = KnBlur(1)
        self.downsampler = Resize(1/sf)

class K3Downsample(AbstractDownsample):
    """ Expect input'shape = [W,H,C]
    """

    def __init__(self, sf):
        self.sf = sf
        self.blur = KnBlur(2)
        self.downsampler = Resize(1/sf)

class K4Downsample(AbstractDownsample):
    """ Expect input'shape = [W,H,C]
    """

    def __init__(self, sf):
        self.sf = sf
        self.blur = KnBlur(3)
        self.downsampler = Resize(1/sf)

class K11Downsample(AbstractDownsample):
    """ Expect input'shape = [W,H,C]
    """

    def __init__(self, sf):
        self.sf = sf
        self.blur = KnBlur(10)
        self.downsampler = Resize(1/sf)

class K12Downsample(AbstractDownsample):
    """ Expect input'shape = [W,H,C]
    """

    def __init__(self, sf):
        self.sf = sf
        self.blur = KnBlur(11)
        self.downsampler = Resize(1/sf)

class GaussianDownsample(AbstractDownsample):
    """ Expect input'shape = [W,H,C]
    """

    def __init__(self, sf, ksize=8, sigma=3):
        self.sf = sf
        self.blur = GaussianBlur(ksize, sigma)
        self.downsampler = Resize(1/sf)


class Resize:
    """ Expect input'shape = [H,W,C]
    """

    def __init__(self, sf, mode='cubic'):
        self.sf = sf
        self.mode = self.mode_map[mode]

    def __call__(self, img):
        [h,w,c] = img.shape
        new_img = np.zeros([int(w*self.sf), int(h*self.sf), c])
        # print(new_img.shape)
        for i in range(c):
            new_img[:,:,i] = cv2.resize(img[:,:,i], (int(w*self.sf), int(h*self.sf)), interpolation=self.mode)
        return new_img

    mode_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA
    }


class HSI2RGB(object):
    def __init__(self, spe=None):
        if spe is None:
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
            self.SPE = loadmat(os.path.join(CURRENT_DIR, 'kernels', 'misr_spe_p.mat'))['P']  # (3,31)
        else:
            self.SPE = spe

    def __call__(self, img):
        return img @ self.SPE.transpose()
