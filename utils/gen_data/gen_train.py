import os
import sys

import caffe
import cv2
import h5py
import lmdb
import numpy as np
from scipy.ndimage import zoom

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from transforms.functional.general import (Data2Volume, crop_center,
                                           data_augmentation, minmax_normalize)


def create_lmdb_train(
        datadir, fns, name, matkey,
        crop_sizes, scales, ksizes, strides,
        load=h5py.File, augment=True,
        seed=2021):
    """
    Create Augmented Dataset
    """
    def preprocess(data):                                                           # h,w,c
        new_data = []
        # data = minmax_normalize(data)
        data = minmax_normalize(data.transpose((2, 0, 1)))                          # c,h,w
        # Visualize3D(data)
        if crop_sizes is not None:
            if data.shape[-1] > crop_sizes[0] and data.shape[-2] > crop_sizes[0]:
                data = crop_center(data, crop_sizes[0], crop_sizes[1])
        for i in range(len(scales)):
            temp = zoom(data, zoom=(1, scales[i], scales[i])) if scales[i] != 1 else data
            temp = Data2Volume(temp, ksizes=ksizes, strides=list(strides[i]))
            new_data.append(temp)
        new_data = np.concatenate(new_data, axis=0)
        if augment:
            for i in range(new_data.shape[0]):
                new_data[i, ...] = data_augmentation(new_data[i, ...])

        return new_data.astype(np.float32)                                           # c,h,w

    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)
    assert len(scales) == len(strides)
    # calculate the shape of dataset
    if load == cv2.imread:
        data = load(datadir + fns[0], 0)
        data = np.expand_dims(data, 2)
    else:
        data = load(datadir + fns[0])[matkey]
    
    print('init:', data.shape)
    data = preprocess(data)
    N = data.shape[0]
    print('processed:', data.shape)

    # We need to prepare the database for the size. We'll set it 1.5 times
    # greater than what we theoretically need.
    map_size = data.nbytes * len(fns) * 1.5
    print('map size (GB):', map_size / 1024 / 1024 / 1024)

    if os.path.exists(name+'.db'):
        raise Exception('database already exist!')
    env = lmdb.open(name+'.db', map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        k = 0
        for i, fn in enumerate(fns):
            try:
                if load == cv2.imread:
                    X = load(datadir + fn, 0)
                    X = np.expand_dims(X, 2)
                else:
                    X = load(datadir + fn)[matkey]
            except:
                print('loading', datadir+fn, 'fail')
                continue
            X = preprocess(X)
            N = X.shape[0]
            for j in range(N):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = X.shape[1]
                datum.height = X.shape[2]
                datum.width = X.shape[3]
                datum.data = X[j].tobytes()
                str_id = '{:08}'.format(k)
                k += 1
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
            print('load mat (%d/%d): %s' % (i, len(fns), fn))

        print('done')


def create_rgb64_1_y():
    print('create rgb_64_1...')
    datadir = '/media/exthdd/datasets/hsi/lzy_data/AID/'  # path of your AID dataset, please change
    out_datadir = '/media/exthdd/datasets/hsi/lzy_data/AID_64_Y'  # path for saving AID_64_Y.db, please change
    fns = os.listdir(datadir)

    create_lmdb_train(
        datadir, fns, out_datadir, None,
        crop_sizes=None,
        scales=(1, ),
        ksizes=(1, 64, 64),
        strides=[(1, 32, 32)],
        load=cv2.imread, augment=True,
    )



if __name__ == '__main__':
    create_rgb64_1_y()
