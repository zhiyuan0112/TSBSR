import os
from os.path import join

import matplotlib.pyplot as plt
from scipy.io import loadmat


def get_mat(path, key):
    return loadmat(path)[key]


dir = 'utils/transforms/kernels/'
filename = 'f_set.mat'
file = join(dir, filename)
key = 'f_set'


x = get_mat(file, key)
kernel = x[0]
print(len(kernel))

for i in range(len(kernel)):
    savedir = join(dir, key)
    os.makedirs(savedir, exist_ok=True)
    plt.imsave(join(savedir, 'k'+str(i+1)+'.pdf'), kernel[i], cmap='gray')
