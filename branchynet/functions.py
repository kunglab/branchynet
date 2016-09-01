from chainer import cuda

def entropy_gpu(x):    
    vec = cuda.elementwise(
        'T x',
        'T y',
        '''
           y = (x == 0) ? 0 : -x*log(x);
        ''',
        'entropy')(x.data)
    return cuda.cupy.sum(vec,1)