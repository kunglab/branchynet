from __future__ import absolute_import

from branchynet.links.links import *
from branchynet.net import BranchyNet

import chainer.links as L
import chainer.functions as F

def get_network(percentTrainKeeps=1):
    network = [
        L.Convolution2D(1, 5, 5,stride=1, pad=3),
        Branch([FL(F.max_pooling_2d, 2, 2), FL(F.ReLU()), L.Convolution2D(5, 10,  3, pad=1, stride=1),
              FL(F.max_pooling_2d, 2, 2), FL(F.ReLU()), L.Linear(640, 10)]),
        FL(F.max_pooling_2d, 2, 2),
        FL(F.ReLU()),
        L.Convolution2D(5, 10, 5,stride=1, pad=3),
        FL(F.max_pooling_2d, 2, 2),
        FL(F.ReLU()),
        L.Convolution2D(10, 20, 5,stride=1, pad=3),
        FL(F.max_pooling_2d, 2, 2),
        FL(F.ReLU()),
        L.Linear(720, 84),
        Branch([L.Linear(84, 10)])
    ]
    net = BranchyNet(network, percentTrainKeeps=percentTrainKeeps)
    return net
