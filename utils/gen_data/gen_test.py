import os
import sys

import h5py
from scipy.io import loadmat, savemat

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from transforms.functional.general import crop_center, minmax_normalize


def create_normed_test(
        datadir, fns, matkey, crop_sizes,
        load, save):
    for fn in fns:
        try:
            data = load(datadir + fn)[matkey]                      # h,w,c
        except:
            print('loading', datadir+fn, 'fail')
            continue
        data = minmax_normalize(data.transpose((2, 0, 1)))         # c,h,w
        data = crop_center(data, crop_sizes[0], crop_sizes[1])     # c,512,512

        data = data.transpose((1, 2, 0))                           # 512,512,c
        save_data = {'gt': data}

        save_path = os.path.join(datadir, 'test_' + str(crop_sizes[0]) + '_norm')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save(os.path.join(save_path, fn), save_data)


if __name__ == '__main__':
    create_normed_test(
        datadir='/data1/liangzhiyuan/data/Remote/',
        fns=['PaviaU.mat'],
        matkey='paviaU',
        crop_sizes=(340, 340),
        load=loadmat,
        save=savemat)
    
    # create_normed_test(
    #     datadir='/home/liangzhiyuan/Data/remote/salinas/',
    #     fns=['Salinas.mat'],
    #     matkey='salinas',
    #     crop_sizes=(128, 128),
    #     load=loadmat,
    #     save=savemat)

    # create_normed_test(
    #     datadir='/home/liangzhiyuan/Data/remote/botswana/test/',
    #     fns=['test_240_norm.mat'],
    #     matkey='salinas',
    #     crop_sizes=(128, 128),
    #     load=loadmat,
    #     save=savemat)
