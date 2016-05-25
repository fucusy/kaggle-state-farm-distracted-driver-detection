# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:36:37 2016

@author: liuzheng
"""

import skimage.io as skio
import skimage
from keras.utils import np_utils
import numpy as np
import os

def readTrainingImages(datapath, imgSize=(1,224,224), meanNorm=True, stdNorm=True, validSplit=0.3):
    ch, ih, iw = imgSize
    folder = os.listdir(datapath)
    folder = np.random.shuffle(folder)
    if ch == 1:
        X_train = np.zeros((len(folder), 1, ih, iw))
        Y_train = np.zeros((len(folder)), dtype=int)
        totalNum = len(folder)
        idx = -1
        for f in folder:
            idx += 1
            print('reading data %s, %d / %d'%(f, idx+1, totalNum))
            label = np.int(f[0])
            img = skio.imread(datapath+f)
            X_train[idx, 0, ...] = img
            Y_train[idx] = label
    elif ch == 3:
        X_train = np.zeros((len(folder), 3, ih, iw))
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
    
    if meanNorm:
        X_train = featurewiseMeanNormalization(X_train)
    if stdNorm:
        X_train = featurewiseStdNormalization(X_train)
    
    trainNum = np.int( len(folder) * (1-validSplit) )
    X_valid = X_train[trainNum:, ...]
    Y_valid = Y_train[trainNum:, :]
    X_train = X_train[:trainNum, ...]
    Y_train = Y_train[:trainNum, :]

    return X_train, Y_train, X_valid, Y_valid


def readTestingImages_noLabel_nameList(datapath, imgSize=(224,224), classNum=10, colorMode='gray', meanNorm=True, stdNorm=True):
    ih, iw = imgSize
    folder_all = os.listdir(datapath)
    folder = []
    for f in folder_all:
        if f[-10:] == 'resize.jpg':
            folder.append(f)
    folder.sort()
    if colorMode == 'gray':
        X_test = np.zeros((len(folder), 1, ih, iw))
#        Y_test = np.zeros((len(folder)), dtype=int)
        totalNum = len(folder)
        idx = -1
        nameList = []
        for f in folder:
            idx += 1
            print('reading testing data %s, %d / %d'%(f, idx+1, totalNum))
            img = skio.imread(datapath+f)
            X_test[idx, 0, ...] = img
#            Y_test[idx] = idx % 9
            nameList.append(f[:f.rfind('_')]+'.jpg')
    elif colorMode == 'rgb':
        X_test = np.zeros((len(folder), 3, ih, iw))
#        Y_test = np.zeros((len(folder)), dtype=int)
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
#            Y_test[idx] = idx % 9
            nameList.append(f[:f.rfind('_')]+'.jpg')
    
#    Y_test = np_utils.to_categorical(Y_test, classNum)
    
    if meanNorm:
        X_test = featurewiseMeanNormalization(X_test)
    if stdNorm:
        X_test = featurewiseStdNormalization(X_test)
    
    return X_test, nameList

def featurewiseMeanNormalization(data):
    data -= np.mean(data, axis=0)
    return data

def featurewiseStdNormalization(data):
    data /= np.std(data, axis=0)
    return data

def computeMeanImage(trainingPath, testingPath, savePath, imgSize):
    ch, ih, iw = imgSize
    meanImage = np.zeros((ch, ih, iw))
    print('computing mean image')
    folder = os.listdir(trainingPath)
    trainNum = 0
    for f in folder:
        if not f[-4:] == '.jpg':
            continue
        img = skimage.img_as_float( skio.imread(trainingPath+f) )
        trainNum += 1
        if ch == 3:
            img = img.swapaxes(1, 2)
            img = img.swapaxes(0, 1)
        meanImage += img
    
    folder = os.listdir(testingPath)
    testNum = 0
    for f in folder:
        if not f[-4:] == '.jpg':
            continue
        img = skimage.img_as_float( skio.imread(testingPath+f) )
        testNum += 1
        if ch == 3:
            img = img.swapaxes(1, 2)
            img = img.swapaxes(0, 1)
        meanImage += img
    meanImage /= (trainNum + testNum)
    with open(savePath, 'wb') as f:
        np.save(f, meanImage)

def prepareTrainingDataFragment(datapath, imgSize=(1,224,224), validSplit=0.3,\
                                fragSize=5000, batchSize=32, meanImagePath='', testingPath=''):
    ch, ih, iw = imgSize
    if not os.path.exists(meanImagePath):
        computeMeanImage(datapath, testingPath, meanImagePath, imgSize)
    else:
        print('mean images exists.')
    folder = os.listdir(datapath)
    np.random.shuffle(folder)
    trainNum = np.int( len(folder) * (1-validSplit) )
    
    trainingFile = folder[:trainNum]
    validationFile = folder[trainNum:]
    
#    num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#    tindex = []
#    for i in range(len(folder)):
#        f = folder[i]
#        if ( not f[0] in num ) or (not f[-4:] == '.jpg'):
#            tindex.append(i)
    
    print('start preparing fragment')
    trainingList = []
    fragList = []
    fragCount = 0
    i = 0
    for tf in trainingFile:
        if not tf[-4:] == '.jpg':
            continue
        
        fragList.append(tf)
        fragCount += 1
        if fragCount == fragSize:
            i += 1
            print('training fragment %d finished'%(i))
            trainingList.append(fragList)
            fragList = []
            fragCount = 0
#            break
    if len(fragList) < batchSize:
        trainingList[-1] += fragList
    else:
        trainingList.append(fragList)
    
    validationList = []
    fragList = []
    fragCount = 0
    i = 0
    for vf in validationFile:
        if not tf[-4:] == '.jpg':
            continue
        fragList.append(vf)
        fragCount += 1
        if fragCount == fragSize:
            i += 1
            print('validation fragment %d finished'%(i))
            validationList.append(fragList)
            fragList = []
            fragCount = 0
#            break
    if len(fragList) < batchSize:
        validationList[-1] += fragList
    else:
        validationList.append(fragList)
    
    return trainingList, validationList

def prepareTestingDataFragment(datapath, imgSize=(1,224,224),\
                                fragSize=5000, batchSize=32, meanImagePath=''):
    ch, ih, iw = imgSize
    
    folder = os.listdir(datapath)
    folder.sort()
    
#    num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#    tindex = []
#    for i in range(len(folder)):
#        f = folder[i]
#        if ( not f[0] in num ) or (not f[-4:] == '.jpg'):
#            tindex.append(i)
    
    print('start preparing testing fragment')
    testingList = []
    fragList = []
    fragCount = 0
    i = 0
    for f in folder:
        if not f[-4:] == '.jpg':
            continue
        
        fragList.append(f)
        fragCount += 1
        if fragCount == fragSize:
            i += 1
            print('testing fragment %d finished'%(i))
            testingList.append(fragList)
            fragList = []
            fragCount = 0
#            break
    if len(fragList) < batchSize:
        testingList[-1] += fragList
    else:
        testingList.append(fragList)
    
    return testingList

def readTrainingFragment(datapath, fragList, imgSize=(1,224,224), meanImage=[], classNum=10):
    ch, ih, iw = imgSize
    fragLen = len(fragList)
    if ch == 1:
        X = np.zeros((fragLen, 1, ih, iw))
        Y = np.zeros((fragLen), dtype=int)
        idx = -1
        print('reading data')
        for f in fragList:
            idx += 1
            # print(f)
            label = np.int(f[0])
            img = skimage.img_as_float(skio.imread(datapath+f) )
#            img -= meanImage
            X[idx, 0, ...] = img
            Y[idx] = label
    elif ch == 3:
        X = np.zeros((fragLen, 3, ih, iw))
        Y = np.zeros((fragLen), dtype=int)
        idx = -1
        print('reading data')
        for f in fragList:
            idx += 1
            label = np.int(f[0])
            img = skimage.img_as_float(skio.imread(datapath+f) )
            img = img.swapaxes(1, 2)
            img = img.swapaxes(0, 1)
#            img -= meanImage
            X[idx, ...] = img
            Y[idx] = label
    X -= np.tile(meanImage, [fragLen, 1, 1, 1])
    Y = np_utils.to_categorical(Y, classNum)
    return X, Y
    
def readTestingFragment(datapath, fragList, imgSize=(1,224,224), meanImage=[]):
    ch, ih, iw = imgSize
    fragLen = len(fragList)
    if ch == 1:
        X = np.zeros((fragLen, 1, ih, iw))
        idx = -1
        print('reading data')
        for f in fragList:
            idx += 1
            # print(f)
            img = skimage.img_as_float(skio.imread(datapath+f) )
#            img -= meanImage
            X[idx, 0, ...] = img
    elif ch == 3:
        X = np.zeros((fragLen, 3, ih, iw))
        idx = -1
        print('reading data')
        for f in fragList:
            idx += 1
            img = skimage.img_as_float(skio.imread(datapath+f) )
            img = img.swapaxes(1, 2)
            img = img.swapaxes(0, 1)
#            img -= meanImage
            X[idx, ...] = img
    X -= np.tile(meanImage, [fragLen, 1, 1, 1])
    return X
























