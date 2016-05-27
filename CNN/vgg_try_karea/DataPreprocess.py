# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:56:17 2016

@author: WuPeng
"""

import numpy as np
from numpy.random import permutation
from keras.utils import np_utils

def preprocess_train(X_train, y_train):
    X_train = np.array(X_train, dtype=np.uint8)
    y_train = np.array(y_train, dtype=np.uint8)

    X_train = X_train.transpose((0, 3, 1, 2))

    y_train = np_utils.to_categorical(y_train, 10)
    X_train = X_train.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        X_train[:, c, :, :] = X_train[:, c, :, :] - mean_pixel[c]

    perm = permutation(len(y_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    
    return X_train, y_train

def preprocess_test(X_test):  
    X_test = np.array(X_test, dtype=np.uint8)
    X_test = X_test.transpose((0, 3, 1, 2))
    X_test = X_test.astype('float32')
    mean_pixel = [103.939, 116.779, 123.68]
    for c in range(3):
        X_test[:, c, :, :] = X_test[:, c, :, :] - mean_pixel[c]
        
    return X_test