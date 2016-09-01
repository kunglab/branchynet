import os
import sys
import glob
import numpy
import numpy as np
from six.moves import cPickle as pickle
from scipy import linalg
from skimage.color import rgb2luv
from skimage import img_as_float

dirname = os.path.dirname(os.path.realpath(__file__))

def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data

def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=False,
                              sqrt_bias=0., min_divisor=1e-8):
    """
    Global contrast normalizes by (optionally) subtracting the mean
    across features and then normalizes by either the vector norm
    or the standard deviation (across features, for each example).
    Parameters
    ----------
    X : ndarray, 2-dimensional
        Design matrix with examples indexed on the first axis and \
        features indexed on the second.
    scale : float, optional
        Multiply features by this const.
    subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing. \
        Defaults to `True`.
    use_std : bool, optional
        Normalize by the per-example standard deviation across features \
        instead of the vector norm. Defaults to `False`.
    sqrt_bias : float, optional
        Fudge factor added inside the square root. Defaults to 0.
    min_divisor : float, optional
        If the divisor for an example is less than this value, \
        do not apply it. Defaults to `1e-8`.
    Returns
    -------
    Xp : ndarray, 2-dimensional
        The contrast-normalized features.
    Notes
    -----
    `sqrt_bias` = 10 and `use_std = True` (and defaults for all other
    parameters) corresponds to the preprocessing used in [1].
    References
    ----------
    .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the current
    # object is the train, valid, or test set.
    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, numpy.newaxis]  # Makes a copy.
    else:
        X = X.copy()

    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = numpy.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = numpy.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, numpy.newaxis]  # Does not make a copy.
    return X

def preprocessing(data):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S))), U.T)
    whiten = np.dot(mdata, components.T)

    return components, mean, whiten

def get_data(gcn=1,whitening=1):
    train = unpickle(dirname+'/data/cifar100/train')
    data = np.asarray(train['data'], dtype=np.float32)
    labels = np.asarray(train['fine_labels'], dtype=np.int32)
    
    data /= 255
    mean = data.mean(axis=0)
    data -= mean
    
    if gcn==1:
        data = global_contrast_normalize(data,use_std=True)
        
    # if whitening == 1:
    #     components, meanw, data = preprocessing(data)

    data = data.reshape((-1, 3, 32, 32))
    # for i,image in enumerate(data):
    #     data[i] = rgb2luv(img_as_float(image).transpose((1,2,0))).transpose((2,0,1))


    # for i in range(50000):
    #     d = data[i]
    #     d -= d.min()
    #     d /= d.max()
    #     data[i] = d.astype(np.float32)
        
    train_data = data
    train_labels = np.asarray(labels, dtype=np.int32)

    test = unpickle(dirname+'/data/cifar100/test')
    data = np.asarray(test['data'], dtype=np.float32)
    
    data /= 255
    data -= mean

    if gcn==1:
        data = global_contrast_normalize(data,use_std=True)

    # if whitening == 1:
    #     mdata = data - meanw
    #     data = np.dot(mdata, components.T)

    data = data.reshape((-1, 3, 32, 32))
    # for i,image in enumerate(data):
    #     data[i] = rgb2luv(img_as_float(image).transpose((1,2,0))).transpose((2,0,1))


    # for i in range(10000):
    #     d = data[i]
    #     d -= d.min()
    #     d /= d.max()
    #     data[i] = d.astype(np.float32)
        
    test_data = data
    test_labels = np.asarray(test['fine_labels'], dtype=np.int32)

    return train_data, train_labels, test_data, test_labels

def get_data_dev(numclasses=2):
    train_data, train_labels, test_data, test_labels = get_data()
    idx = (train_labels == 0)
    for i in range(1,numclasses):
        idx |= (train_labels == i)
    train_data = train_data[idx]
    train_labels = train_labels[idx]
    idx = (test_labels == 0)
    for i in range(1,numclasses):
        idx |= (test_labels == i)
    test_data = test_data[idx]
    test_labels = test_labels[idx]
    
    return train_data, train_labels, test_data, test_labels
