import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import inspect
from scipy.stats import entropy
import types
import copy
import inspect

# NOTE: NEED TO IMPLEMENT 
# zerograds
# __deepcopy__
# to_gpu
# to_cpu

# Deprecated use SL instead
class SL2(Link):
    pass

# Deprecated use FL(F.max_pooling_2d, 2) instead
class Max(Link):
    pass

# Deprecated use FL(F.relu, 2) instead
class ReLU(Link):
    pass

import inspect
import copy
class X(chainer.ChainList):
    def __init__(self, main):
        super(X, self).__init__()
        for link in main:
            self.add_link(link)
        self.weights = [1.,1.]
    def __deepcopy__(self, memo):
        newmain = []
        for link in self:
            newlink = copy.deepcopy(link)
            newlink.name = None
            newmain.append(newlink)            
        new = type(self)(newmain)
        return new
    def set_weights(self, weights):
        self.weights = weights
        return
    def __call__(self, x, test):
        h = x
        for link in self:
            if len(inspect.getargspec(link.__call__)[0]) == 2:
                h = link(h)
            else:
                h = link(h,test)
        if x.data.shape != h.data.shape:
            xp = chainer.cuda.get_array_module(x.data)
            n, c, hh, ww = x.data.shape
            pad_c = h.data.shape[1] - c
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = chainer.Variable(p, volatile=test)
            x = F.concat((p, x))
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x, 1, 2)
        return self.weights[0]*h + self.weights[1]*x

class SL(Link):
    # Switch Layer
    def __init__(self, fnTrain, fnTest=None):
        super(SL, self).__init__()
        self.fnTrain = fnTrain
        self.fnTest = fnTest
    def zerograds(self):
        super(SL, self).zerograds()
        if self.fnTrain is not None:
            self.fnTrain.zerograds()
        if self.fnTest is not None:
            self.fnTest.zerograds()
    def __deepcopy__(self, memo):
        fnTrain = copy.deepcopy(self.fnTrain,memo)
        fnTest = copy.deepcopy(self.fnTest,memo)
        new = type(self)(fnTrain,fnTest)
        return new
    def to_gpu(self):
        if self.fnTrain is not None:
            self.fnTrain.to_gpu()
        if self.fnTest is not None:
            self.fnTest.to_gpu()
    def to_cpu(self):
        if self.fnTrain is not None:
            self.fnTrain.to_cpu()
        if self.fnTest is not None:
            self.fnTest.to_cpu()
    def __call__(self, x, test=False):
        if not test:
            return self.fnTrain(x,test)
        else:
            if self.fnTest is None:
                return x
            return self.fnTest(x,test)
        return x

class FL(Link):
    # Function Layer
    def __init__(self, fn, *arguments, **keywords):
        super(FL, self).__init__()
        self.fn = fn
        self.arguments = arguments
        self.keywords = keywords
    def __call__(self, x, test=False):
        return self.fn(x, *self.arguments, **self.keywords)

class Net(ChainList):
    def __init__(self,weight=1.):
        super(Net, self).__init__()
        self.weight = weight
        self.starti = 0
        self.endi = 0
    def __call__(self, x, test=False, starti=0, endi=None):
        h = x
        for link in self[starti:endi]:
            if len(inspect.getargspec(link.__call__)[0]) == 2:
                h = link(h)
            else:
                h = link(h,test)
        self.h = h
        return h
    def train(self, x, t, starti=0, endi=None):
        h = self(x,False,starti,endi)
        self.accuracy = F.accuracy(h,t)
        self.loss = F.softmax_cross_entropy(h,t)
        return self.loss
    def test(self, x, starti=0, endi=None):
        h = self(x,True,starti,endi)
        #if not isinstance(t,types.NoneType) and endi == None:
        #    self.accuracy = F.accuracy(h,t)
        return h
    
class Branch(ChainList):
    def __init__(self, branch, weight=1.):
        super(Branch, self).__init__()
        self.branch = branch
        self.weight = 1.
        for link in branch:
            self.add_link(link)
            
    def zerograds(self):
        super(SL, self).zerograds()
        for link in self.branch:
            link.zerograds()
            
    def __deepcopy__(self, memo):
        newbranches = []
        for link in self.branch:
            newbranches.append(copy.deepcopy(link,memo))
        newbranch = newbranches
        new = type(self)(newbranches,self.weight)
        return new
    
    def __call__(self, x, test=False):
        h = x
        for link in self[starti:endi]:
            if len(inspect.getargspec(link.__call__)[0]) == 2:
                h = link(h)
            else:
                h = link(h,test)
        return h
    