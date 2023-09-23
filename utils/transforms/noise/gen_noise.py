import argparse
import os

import numpy as np


def basic_opt(description=''):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--sigma', type=float, default=10, help='Sigma of Noise.')
    parser.add_argument('--scaleFactor', '-sf', type=int, default=4, help='Scale Factor.')
    parser.add_argument('--height', '-ht', type=int, default=512, help='Height.')
    parser.add_argument('--width', '-w', type=int, default=512, help='Width.')
    parser.add_argument('--band', '-b', type=int, default=31, help='Number of Band.')

    opt = parser.parse_args()
    return opt


opt = basic_opt()
print(opt)

sigma = opt.sigma / 255.
scale_factor = opt.scaleFactor
shape = (opt.height//scale_factor, opt.width//scale_factor, opt.band)
h,w,c = shape

noise = np.random.normal(0, sigma, shape).astype('float32')
save_path = 'utils/transforms/noise/test_' + str(h) + '_' + str(c)
os.makedirs(save_path, exist_ok=True)
np.save(os.path.join(save_path, 'sigma' + str(opt.sigma) + '.npy'), noise)
