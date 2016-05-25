# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:50:27 2016

@author: liuzheng
"""

import DataAugmentation as da
#import skimage.io as skio
import os
isServer = False
if not isServer:
    pcProjectpath = '/home/liuzheng/competition/kaggle/distractedDrivers/'
    mxnetRoot = '/home/liuzheng/toolbox/mxnet/'

colorMode = 'rgb'
cropMode = 'entire'
reSize = (224,224)

trainListName = colorMode + '_' + cropMode + '_trainDatalist_distractedDrivers.lst'
testListName = colorMode + '_' + cropMode + '_testDatalist_distractedDrivers.lst'
trainRecName = colorMode + '_' + cropMode + '_trainRecord_distractedDrivers.rec'
testRecName = colorMode + '_' + cropMode + '_testRecord_distractedDrivers.rec'

scriptPath = os.getcwd() + '/'

trainDatapath = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/train/'
testDatapath = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/test/'
trainFolderList = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
classNum = len(trainFolderList)

trainSavepath = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/trainAugmentation_'+colorMode + '_' + cropMode+'_'+'%03d'%(reSize[0])+'/'
if not os.path.exists(trainSavepath):
    os.mkdir(trainSavepath)
testSavepath = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/testAugmentation_'+colorMode + '_' + cropMode+'_'+'%03d'%(reSize[0])+'/'
if not os.path.exists(testSavepath):
    os.mkdir(testSavepath)


# cropSize=[224,224]
'''============================= Processing training data ======================================='''
for c in range(classNum):
    classDatapath = trainDatapath+trainFolderList[c]+'/'
    da.augmentTrainingImages(datapath=classDatapath, classIdx=c, savepath=trainSavepath, reSize=reSize,\
                            needGray=False, needMirror=False, needRotate=False)

'''=============================================================================================='''

'''============================= Processing testing data ======================================='''
da.augmentTestingImages(datapath=testDatapath, savepath=testSavepath, reSize=reSize,\
                            needGray=False, needMirror=False, needRotate=False)
'''=============================================================================================='''

#'''============================= fragment data for insufficient memory ======================================='''
#da.fragmentTrainingData(trainSavepath, imgSize=cropSize, colorMode=colorMode, fragNum=10)
#da.fragmentTestingData(testSavepath, imgSize=cropSize, colorMode=colorMode, fragNum=10)












