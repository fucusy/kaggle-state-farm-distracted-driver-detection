# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:36:37 2016

@author: liuzheng
"""

import skimage.io as skio
from keras.utils import np_utils
import numpy as np
import os

def readTrainingImages_generator(datapath):
    folder = os.listdir(datapath)
    X_train = np.zeros((len(folder), 3, 64, 64))
    Y_train = np.zeros((len(folder)), dtype=int)
    totalNum = len(folder)
    idx = -1
    for f in folder:
        idx += 1
        print('reading data %s, %d / %d'%(f, idx+1, totalNum))
        label = np.int(f[0])
        img = skio.imread(datapath+f)
        img = img.swapaxes(1, 2)
        img = img.swapaxes(0, 1)
        X_train[idx, ...] = img
        Y_train[idx] = label
    
    classNum = np.max(Y_train) + 1
    Y_train = np_utils.to_categorical(Y_train, classNum)

    return X_train, Y_train


def readTestingImages_generator_noLabel_nameList(datapath, classNum=10):
    folder_all = os.listdir(datapath)
    folder = []
    for f in folder_all:
        if f[-10:] == 'resize.jpg':
            folder.append(f)
    folder.sort()
    X_test = np.zeros((len(folder), 3, 64, 64))
    Y_test = np.zeros((len(folder)), dtype=int)
    totalNum = len(folder)
    idx = -1
    nameList = []
    for f in folder:
        idx += 1
        print('reading testing data %s, %d / %d'%(f, idx+1, totalNum))
        img = skio.imread(datapath+f)
        img = img.swapaxes(1, 2)
        img = img.swapaxes(0, 1)
        X_test[idx, ...] = img
        Y_test[idx] = idx % 9
        nameList.append(f[:-11]+'.jpg')
    
    Y_test = np_utils.to_categorical(Y_test, classNum)

    return X_test, Y_test, nameList
