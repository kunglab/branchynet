from __future__ import absolute_import

from branchynet.links.links import *
from branchynet.links import resnet
from branchynet.net import BranchyNet

import chainer.links as L
import chainer.functions as F
import math

def norm():
    return [FL(F.relu), 
            FL(F.local_response_normalization,n=3, alpha=5e-05, beta=0.75)]

def get_network(percentTrainKeeps=1):
    conv = lambda n: [L.Convolution2D(n, 32,  3, pad=1, stride=1), FL(F.relu)]
    # cap2 = lambda n: [FL(F.max_pooling_2d, 3, 2), L.Linear(n, int(n/2)), L.Linear(int(n/2), 10)]

    cap = lambda n: [L.Linear(n, 10)]

    # ResNet-110 for CiFAR10 as presented in the ResNet paper
    w = math.sqrt(2)
    n = 18
    network = [
        L.Convolution2D(3, 16, 3, 1, 0, w),
        L.BatchNormalization(16),
        FL(F.relu),
    ]
    network += [Branch([L.Convolution2D(16, 64,  5, pad=2, stride=1)] + norm() +  conv(64) + conv(32) + cap(28800))]
    # network += [resnet.ResBlock(16, 16)]
    # network += [Branch(resnet.ResBlock(16, 16) + resnet.ResBlock(16, 16) + cap2(7200))]

    for i in range(n):
        network += [resnet.ResBlock(16, 16)]

    network += [Branch([resnet.ResBlock(16, 16), L.Linear(14400, 10)])]

    for i in range(n):
        network += [resnet.ResBlock(32 if i > 0 else 16, 32,
                                       1 if i > 0 else 2)]
    for i in range(n):
        network += [resnet.ResBlock(64 if i > 0 else 32, 64,
                                       1 if i > 0 else 2)]

    network += [FL(F.average_pooling_2d, 6, 1)]
    network += [Branch([L.Linear(576, 10)])]


    net = BranchyNet(network, percentTrainKeeps=percentTrainKeeps)
    return net
