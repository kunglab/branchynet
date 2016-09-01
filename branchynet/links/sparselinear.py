from chainer.functions.connection import linear
import chainer.functions as F
import chainer.links as L
from chainer import link,Variable,cuda,optimizers,Chain

from chainer import function
from chainer.utils import type_check

import numpy
import numpy as np
from scipy.sparse import csr_matrix

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)

from itertools import chain, repeat

class SparseLinearFunction(function.Function):
    
    def __init__(self, indices, indptr, shape, mask):
        super(SparseLinearFunction, self).__init__()
        self.indices = indices
        self.indptr = indptr
        self.shape = shape
        self.mask = mask
        
        self.rowcols = zip(
          chain(*(
                repeat(i, r)
                for (i,r) in enumerate(indptr[1:] - indptr[:-1])
          )),
          indices
        )
        
    def iter_csr(self):
        for (row, col) in self.rowcols:
            yield (row, col)

    def check_type_forward(self, in_types):
        return

    def forward(self, inputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        W = csr_matrix((W, self.indices, self.indptr), shape=self.shape, dtype=x.dtype)
        y = W.dot(x.T).T
        if len(inputs) == 3:
            b = inputs[2]
            y += b        
        return y,

    def backward(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        W = csr_matrix((W, self.indices, self.indptr), shape=self.shape, dtype=x.dtype)
        gy = grad_outputs[0]
        gx = W.T.dot(gy.T).T.reshape(inputs[0].shape)
        gW = np.array([gy.T[row,:].dot(x[:,col]) for row, col in self.iter_csr()])
        # Alternatively
        # gW = gy.T.dot(x)[self.mask]
        if len(inputs) == 3:
            gb = gy.sum(0)
            return gx, gW, gb
        else:
            return gx, gW
        
class SparseLinear(link.Link):
    def __init__(self, in_size, out_size, sparseness=.99, wscale=1, bias=0, nobias=False,
                 initialW=None, initial_bias=None):

        if initialW is None:
            initialW = numpy.random.normal(
                0, wscale * numpy.sqrt(1. / in_size), (out_size, in_size))

        mask = numpy.abs(initialW) < numpy.percentile(numpy.abs(initialW),sparseness*100)
        initialW[mask] = 0
                
        self.spW = csr_matrix(initialW,dtype=np.float32)
        
        self.mask = ~mask
        
        super(SparseLinear, self).__init__(W=(self.spW.data.size))
        
        self.W.data = self.spW.data
        
        if nobias:
            self.b = None
        else:
            self.add_param('b', out_size)
            if initial_bias is None:
                initial_bias = bias
            self.b.data[...] = initial_bias
        
        self.slf = SparseLinearFunction(self.spW.indices,self.spW.indptr,self.spW.shape,self.mask)

    def __call__(self, x):
        if self.b is None:
            return self.slf(x, self.W)
        else:
            return self.slf(x, self.W, self.b)    