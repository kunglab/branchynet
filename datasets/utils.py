import numpy as np

def compute_mean(train_data):
    return np.mean(train_data,0)

def subtract_mean(train_data,mean):
    return train_data - mean