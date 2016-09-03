from __future__ import absolute_import

from branchynet.links.links import *
from branchynet.net import BranchyNet

import chainer.functions as F
import chainer.links as L

def norm():
    return [FL(F.relu), FL(F.max_pooling_2d, 3, 2),
            FL(F.local_response_normalization,n=3, alpha=5e-05, beta=0.75)]

conv = lambda n: [L.Convolution2D(n, 32,  3, pad=1, stride=1), FL(F.relu)]
cap =  lambda n: [FL(F.max_pooling_2d, 3, 2), L.Linear(n, 10)]

def gen_2b(branch1, branch2):
    network = [
        L.Convolution2D(3, 32, 5, pad=2, stride=1),
        FL(F.relu),
        FL(F.max_pooling_2d, 3, 2),
        FL(F.local_response_normalization,n=3, alpha=5e-05, beta=0.75),
        L.Convolution2D(32, 64,  5, pad=2, stride=1),
        Branch(branch1),
        FL(F.relu),
        FL(F.max_pooling_2d, 3, 2),
        FL(F.local_response_normalization,n=3, alpha=5e-05, beta=0.75),
        L.Convolution2D(64, 96,  3, pad=1, stride=1),
        Branch(branch2),
        FL(F.relu),
        L.Convolution2D(96, 96,  3, pad=1, stride=1),
        FL(F.relu),
        L.Convolution2D(96, 64,  3, pad=1, stride=1),
        FL(F.relu),
        FL(F.max_pooling_2d, 3, 2),
        L.Linear(1024, 256),
        FL(F.relu),
        SL(FL(F.dropout,0.5,train=True)),
        L.Linear(256, 128),
        FL(F.relu),
        SL(FL(F.dropout,0.5,train=True)),
        Branch([L.Linear(128, 10)])
    ]
    
    return network

def get_network(percentTrainKeeps=1):
    network = gen_2b(branch1=norm() + conv(64) + conv(32) + cap(512),
                              branch2=norm() + conv(96) + cap(128))
    net = BranchyNet(network, percentTrainKeeps=percentTrainKeeps)
    return net
