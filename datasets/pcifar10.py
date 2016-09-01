import os
import sys
import glob
import numpy as np
from six.moves import cPickle as pickle
from scipy import linalg

dirname = os.path.dirname(os.path.realpath(__file__))

def get_data(redir=''):
    datasets = np.load(redir + 'datasets/data/pcifar10/data.npz')
    train_data = datasets['train_x']
    train_labels = datasets['train_y']
    test_data = datasets['test_x']
    test_labels = datasets['test_y']
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

