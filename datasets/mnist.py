from sklearn.datasets import fetch_mldata
import numpy as np

def get_data():
    mnist = fetch_mldata('MNIST original')
    x_all = mnist['data'].astype(np.float32) / 255
    y_all = mnist['target'].astype(np.int32)
    x_train, x_test = np.split(x_all, [60000])
    y_train, y_test = np.split(y_all, [60000])

    x_train = x_train.reshape([-1,1,28,28])
    x_test = x_test.reshape([-1,1,28,28])
    return x_train,y_train,x_test,y_test
