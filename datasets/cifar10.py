import os
import sys
import glob
import numpy as np
from six.moves import cPickle as pickle
from scipy import linalg

dirname = os.path.dirname(os.path.realpath(__file__))

def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data

def preprocessing(data):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S))), U.T)
    whiten = np.dot(mdata, components.T)

    return components, mean, whiten

def get_data(whitening=1):
    data = np.zeros((50000, 3 * 32 * 32), dtype=np.float32)
    labels = []
    for i, data_fn in enumerate(
            sorted(glob.glob(dirname+'/data/cifar10/data_batch*'))):
        batch = unpickle(data_fn)
        data[i * 10000:(i + 1) * 10000] = batch['data']
        labels.extend(batch['labels'])
    
    if whitening == 1:
        components, mean, data = preprocessing(data)

    for i in range(50000):
        d = data[i]
        d -= d.min()
        d /= d.max()
        data[i] = d.astype(np.float32)
        
    train_data = data.reshape((-1, 3, 32, 32))
    train_labels = np.asarray(labels, dtype=np.int32)

    test = unpickle(dirname+'/data/cifar10/test_batch')
    data = np.asarray(test['data'], dtype=np.float32)
    
    if whitening == 1:
        mdata = data - mean
        data = np.dot(mdata, components.T)

    for i in range(10000):
        d = data[i]
        d -= d.min()
        d /= d.max()
        data[i] = d.astype(np.float32)
        
    test_data = data.reshape((-1, 3, 32, 32))
    test_labels = np.asarray(test['labels'], dtype=np.int32)

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
