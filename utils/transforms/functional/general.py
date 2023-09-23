import random
from itertools import product

import numpy as np


def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)


def crop_center(img, cropx, cropy):
    x, y = img.shape[-2], img.shape[-1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[..., startx:startx+cropx, starty:starty+cropy]


def rand_crop(img, cropx, cropy):
    x, y = img.shape[-2], img.shape[-1]
    x1 = random.randint(0, x - cropx)
    y1 = random.randint(0, y - cropy)
    return img[..., x1:x1+cropx, y1:y1+cropy]


def mod_crop(img, modulo):
    _, ih, iw = img.shape
    ih = ih - (ih % modulo)
    iw = iw - (iw % modulo)
    img = img[:, 0:ih, 0:iw]
    return img


def Data2Volume(data, ksizes, strides):
    """
    Construct Volumes from Original High Dimensional (D) Data
    """
    dshape = data.shape
    PatNum = lambda l, k, s: (np.floor( (l - k) / s ) + 1)    

    TotalPatNum = 1
    for i in range(len(ksizes)):
        TotalPatNum = TotalPatNum * PatNum(dshape[i], ksizes[i], strides[i])
    
    V = np.zeros([int(TotalPatNum)]+ksizes); # create D+1 dimension volume

    args = [range(kz) for kz in ksizes]
    for s in product(*args):
        s1 = (slice(None),) + s
        s2 = tuple([slice(key, -ksizes[i]+key+1 or None, strides[i]) for i, key in enumerate(s)])
        V[s1] = np.reshape(data[s2],-1)
        
    return V


def data_augmentation(image, mode=None):
    """
    Warning: this function is not available for pytorch DataLoader now,
    since it only return a view of original array 
    which is currently not supported by DataLoader.

    To use data augmentation in data with type of numpy.ndarray, 
    you need first transform the numpy array into PIL.Image, then 
    use torchvision.transforms to augment data.

    Data augmentation in numpy level rather than PIL.Image level
    """
    axes = (-2, -1)
    if mode is None:
        mode = random.randint(0, 7)
    if mode == 0:
        # original
        image = image
    elif mode == 1:
        # flip up and down
        image = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = np.rot90(image, axes=axes)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image, axes=axes)
        image = np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = np.rot90(image, k=2, axes=axes)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes=axes)
        image = np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = np.rot90(image, k=3, axes=axes)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=axes)
        image = np.flipud(image)
    
    return image