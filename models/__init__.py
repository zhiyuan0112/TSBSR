from .unsupervised import CNN_Blind_Sparse
from .swinir import SwinIR


def swinir():
    net = SwinIR()
    net.use_2dconv = True
    return net

def cnn_103_128_64_blind_sparse():  # paviau
    net = CNN_Blind_Sparse(num_channels=103, num_endmember=128, n_feats=64)
    net.use_2dconv = True
    return net


def cnn_191_128_64_blind_sparse():  # wdc
    net = CNN_Blind_Sparse(num_channels=191, num_endmember=128, n_feats=64)
    net.use_2dconv = True
    return net


def cnn_224_128_64_blind_sparse():  # salinas
    net = CNN_Blind_Sparse(num_channels=224, num_endmember=128, n_feats=64)
    net.use_2dconv = True
    return net


def cnn_176_128_64_blind_sparse():  # ksc
    net = CNN_Blind_Sparse(num_channels=176, num_endmember=128, n_feats=64)
    net.use_2dconv = True
    return net


def cnn_145_128_64_blind_sparse():  # botswana
    net = CNN_Blind_Sparse(num_channels=145, num_endmember=128, n_feats=64)
    net.use_2dconv = True
    return net


def cnn_144_128_64_blind_sparse():  # houston13
    net = CNN_Blind_Sparse(num_channels=144, num_endmember=128, n_feats=64)
    net.use_2dconv = True
    return net


def cnn_48_128_64_blind_sparse():  # houston18
    net = CNN_Blind_Sparse(num_channels=48, num_endmember=128, n_feats=64)
    net.use_2dconv = True
    return net
