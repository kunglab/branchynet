import math
import chainer
import chainer.functions as F
import chainer.links as L

import copy

# class ResBlockBranch(chainer.Chain):
#     def __init__(self, in_size, out_size, lin, lout, stride=1, ksize=1):
#         w = math.sqrt(2)
#         super(ResBlockBranch, self).__init__(
#             conv1=L.Convolution2D(in_size, out_size, 3, stride, 1, w),
#             bn1=L.BatchNormalization(out_size),
#             conv2=L.Convolution2D(out_size, out_size, 3, 1, 1, w),
#             bn2=L.BatchNormalization(out_size),
#             b_conv1=L.Convolution2D(in_size, out_size, 3, stride, 1, w),
#             b_bn1=L.BatchNormalization(out_size),
#             b_conv2=L.Convolution2D(out_size, out_size, 3, 1, 1, w),
#             b_bn2=L.BatchNormalization(out_size),
#             b_fc=L.Linear(lin,lout)
#         )
#         self.lin = lin
#         self.lout = lout
#         self.in_size = in_size
#         self.out_size = out_size
#         self.stride = stride
#         self.ksize = ksize
#     def __deepcopy__(self, memo):
#         new = type(self)(self.in_size, self.out_size, self.lin, self.lout, self.stride, self.ksize)
#         return new
#     def __call__(self, x, test):
#         train = not test
#         h = F.relu(self.bn1(self.conv1(x), test=not train))
#         h = self.bn2(self.conv2(h), test=not train)
#         # branch
        
#         return F.relu(h)
    
class ResBlock(chainer.Chain):

    def __init__(self, in_size, out_size, stride=1, ksize=1):
        w = math.sqrt(2)
        super(ResBlock, self).__init__(
            conv1=L.Convolution2D(in_size, out_size, 3, stride, 1, w),
            bn1=L.BatchNormalization(out_size),
            conv2=L.Convolution2D(out_size, out_size, 3, 1, 1, w),
            bn2=L.BatchNormalization(out_size),
        )
        self.in_size = in_size
        self.out_size = out_size
        self.stride = stride
        self.ksize = ksize
    def __deepcopy__(self, memo):
        new = type(self)(self.in_size, self.out_size, self.stride, self.ksize)
        return new
    def __call__(self, x, test):
        train = not test
        h = F.relu(self.bn1(self.conv1(x), test=not train))
        h = self.bn2(self.conv2(h), test=not train)
        if x.data.shape != h.data.shape:
            xp = chainer.cuda.get_array_module(x.data)
            n, c, hh, ww = x.data.shape
            pad_c = h.data.shape[1] - c
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = chainer.Variable(p, volatile=not train)
            x = F.concat((p, x))
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x, 1, 2)
        return F.relu(h + x)

class BottleNeckA(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride=2):
        w = math.sqrt(2)
        super(BottleNeckA, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, w, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, w, nobias=True),
            bn3=L.BatchNormalization(out_size),

            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, w, nobias=True),
            bn4=L.BatchNormalization(out_size),
        )
        self.in_size = in_size
        self.ch = ch
        self.out_size = out_size
        self.stride = stride        
    def __deepcopy__(self, memo):
        new = type(self)(self.in_size, self.ch, self.out_size, self.stride)
        return new
    def __call__(self, x, test):
        h1 = F.relu(self.bn1(self.conv1(x), test=test))
        h1 = F.relu(self.bn2(self.conv2(h1), test=test))
        h1 = self.bn3(self.conv3(h1), test=test)
        h2 = self.bn4(self.conv4(x), test=test)

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):
    def __init__(self, in_size, ch):
        w = math.sqrt(2)
        super(BottleNeckB, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0, w, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, w, nobias=True),
            bn3=L.BatchNormalization(in_size),
        )
        self.in_size = in_size
        self.ch = ch
    def __deepcopy__(self, memo):
        new = type(self)(self.in_size, self.ch)
        return new
    def __call__(self, x, test):
        h = F.relu(self.bn1(self.conv1(x), test=test))
        h = F.relu(self.bn2(self.conv2(h), test=test))
        h = self.bn3(self.conv3(h), test=test)

        return F.relu(h + x)


class Block(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        links = [('a', BottleNeckA(in_size, ch, out_size, stride))]
        for i in range(layer-1):
            links += [('b{}'.format(i+1), BottleNeckB(out_size, ch))]

        for link in links:
            self.add_link(*link)
        self.forward = links
        
        self.layer = layer
        self.in_size = in_size
        self.ch = ch
        self.out_size = out_size
        self.stride = stride
        
    def __deepcopy__(self, memo):
        new = type(self)(self.layer, self.in_size, self.ch, self.out_size, self.stride)
        return new

    def __call__(self, x, test):
        for name,_ in self.forward:
            f = getattr(self, name)
            h = f(x if name == 'a' else h, test)

        return h

