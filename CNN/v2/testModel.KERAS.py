# -*- coding: utf-8 -*-
"""
Created on Wed May 25 10:30:08 2016

@author: liuzheng
"""

from keras.models import model_from_json
from keras.optimizers import SGD, Adadelta
import numpy as np
import os
import csv
import DataReader_KERAS as dr_keras

def loadModel(archFilename, weightFilename, loss='categorical_crossentropy', optimizer='sgd'):
    model = model_from_json(open(archFilename).read())
    model.load_weights(weightFilename)
    print('compiling model.....')
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#def savePrediction_CSV(nameList, prediction, prefix):
#    fileObj = open(prefix+'prediction.csv', 'wb')
#    writer = csv.writer(fileObj)
#    writer.writerow(['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
#    for idx in range(len(nameList)):
#        line = [nameList[idx]] + list(prediction[idx, :])
#        writer.writerow(line)
#    fileObj.close()

def savePrediction_CSV(resultDict, prefix):
    fileObj = open(prefix+'prediction.csv', 'wb')
    writer = csv.writer(fileObj)
    writer.writerow(['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
    keys = resultDict.keys()
    keys.sort()
    for k in keys:
        line = [k+'.jpg']
        for i in range(10):
            line.append('%.3f'%(resultDict[k][i]))
        writer.writerow(line)
    fileObj.close()

def statPrediction(resultDict, fragList, fragmentPrediction):
    for i in range(fragmentPrediction.shape[0]):
        imgName = fragList[i][:fragList[i].rfind('_')]
        if imgName in resultDict.keys():
            resultDict[imgName].append(fragmentPrediction[i, :])
        else:
            resultDict[imgName] = [fragmentPrediction[i, :]]
    return resultDict

def fuseCropResult(resultDict):
    finalDict = {}
    for k in resultDict.keys():
        finalDict[k] = np.mean(np.array(resultDict[k]), axis=0)
    return finalDict

'''====================================================================================================='''
#isServer = False
#if not isServer:
#    pcProjectpath = '/home/liuzheng/competition/kaggle/distractedDrivers/'
#cropMode = 'manualCrop'
#colorMode = 'gray'
#savePrefix=pcProjectpath+colorMode+'_'+cropMode+'/'
#meanImagePath = savePrefix + 'meanImage.npy'
#
#archFilename = '/home/liuzheng/competition/kaggle/distractedDrivers/gray_manualCrop/vgg_self_exp1_keras_arch.json'
#weightFilename = '/home/liuzheng/competition/kaggle/distractedDrivers/gray_manualCrop/vgg_self_exp1_keras_weights_best_vLoss0.00967_vAcc0.997.h5'
#
#testDatapath = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/testAugmentation_gray_manualCrop/'
#
#model = loadModel(archFilename, weightFilename, loss='categorical_crossentropy', optimizer='sgd')
#
#testingList = dr_keras.prepareTestingDataFragment(datapath=testDatapath, imgSize=(1,224,224),\
#                                fragSize=5000, batchSize=32, meanImagePath=meanImagePath)
#meanImage = np.load(meanImagePath)
#resultDict = {}
#i = 0
#for tl in testingList:
#    i += 1
#    X_test = dr_keras.readTestingFragment(testDatapath, fragList=tl, imgSize=(1,224,224), meanImage=meanImage)
#
#    print('predicting fragment %d'%(i))
#    prediction = model.predict(X_test, batch_size=64)
#    resultDict = statPrediction(resultDict, fragList=tl, fragmentPrediction=prediction)
#
#print('fuse crop result')
#resultDict = fuseCropResult(resultDict)

prefix = '/home/liuzheng/competition/kaggle/distractedDrivers/gray_manualCrop/'
savePrediction_CSV(resultDict, prefix=prefix)


























