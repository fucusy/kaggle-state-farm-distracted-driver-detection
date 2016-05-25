# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:21:51 2016

@author: liuzheng
"""

import DataReader_KERAS as dr_keras
import ModelInference as mi
import os
import pickle
import numpy as np
import skimage.io as skio

from keras.callbacks import ModelCheckpoint

def trainModel(datapath, imgSize=(1,224,224), nb_iter=5, batch_size=32, optimizer='sgd', savePrefix='', colorMode='gray',\
                saveEpoch=10, validSplit=0.3, isContinue=False, arcPath='', weightsPath=''):
    
    X_train, Y_train, X_valid, Y_valid = dr_keras.readTrainingImages(datapath, imgSize, meanNorm=True, stdNorm=True)
    input_shape = X_train[0, ...].shape
    classNum = Y_train.shape[1]
    
    if isContinue:
        model = mi.fromJson(arcPath, weightsPath, optimizer)
    else:
        model = mi.inference_xavier_prelu_sgd_224(input_shape, classNum)
#        caffe_root = '/home/liuzheng/toolbox/caffe-master/'
#        modelDefinitionPath = '/home/liuzheng/toolbox/caffe-master/models/VGG_ILSVRC_16_layers_deploy.prototxt'
#        modelWeightsPath = '/home/liuzheng/toolbox/caffe-master/models/VGG_ILSVRC_16_layers.caffemodel'
#        model = mi.caffeModel2keras(caffe_root, modelDefinitionPath, modelWeightsPath, saveConv=True,\
#                    saveFC=False, inputSize=input_shape, lrnSubstitution='dropout', classNum=classNum, optimizer='sgd')
    
    history = []
    print('start training')
    for it in range(1, nb_iter+1):
        print('iter %d'%(it))
        checkpoint = ModelCheckpoint(savePrefix + '_keras_checkpoint_weights_%03d.h5'%(it), monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        his = model.fit(X_train, Y_train, batch_size, nb_epoch=saveEpoch, callbacks=[checkpoint], validation_data=(X_valid,Y_valid))
        history.append(his)
        json_string = model.to_json()
        open(savePrefix + '_keras_model_architecture_%03d.json'%(it*saveEpoch), 'w').write(json_string)
        model.save_weights(savePrefix + '_keras_model_weights_%03d.h5'%(it*saveEpoch), overwrite=True)
    return history

def trainModel_fragment(datapath, expIter, imgSize=(1,224,224), classNum=10, nb_iter=100, modelName='vgg_keras', optimizer='sgd', savePrefix='',\
                validSplit=0.3, fragSize=5000, meanImagePath='', testpath='', isContinue=False, arcPath='', weightsPath=''):
    
    trainingList, validationList = dr_keras.prepareTrainingDataFragment(datapath=datapath, imgSize=imgSize,\
                                validSplit=validSplit, fragSize=fragSize, batchSize=batch_size, meanImagePath=meanImagePath, testingPath=testpath)
    meanImage = np.load(meanImagePath)
    
    if isContinue:
        model = mi.fromJson(arcPath, weightsPath, optimizer)
    else:
        if modelName == 'vgg_self':
            model = mi.inference_xavier_prelu_sgd_224(input_shape=imgSize, classNum=classNum)
        elif modelName == 'vgg_keras':
            model = mi.vgg_std16_model(img_rows=imgSize[1], img_cols=imgSize[2], color_type=3)
        elif modelName == 'vgg_caffe':
            caffe_root = '/home/liuzheng/toolbox/caffe-master/'
            modelDefinitionPath = '/home/liuzheng/toolbox/caffe-master/models/VGG_ILSVRC_16_layers_deploy.prototxt'
            modelWeightsPath = '/home/liuzheng/toolbox/caffe-master/models/VGG_ILSVRC_16_layers.caffemodel'
            model = mi.caffeModel2keras(caffe_root, modelDefinitionPath, modelWeightsPath, saveConv=True,\
                        saveFC=False, inputSize=imgSize, lrnSubstitution='dropout', classNum=classNum, optimizer='sgd')
    
    print('%s | start training'%(modelName))
    minValidLoss = np.inf
    maxValidAcc = 0.
    
    json_string = model.to_json()
    open(savePrefix + modelName + '_exp%d_'%(expIter) + 'keras_arch.json', 'w').write(json_string)
    
    for it in range(1, nb_iter+1):
        fragmentCount = 0
        totalFrag = len(trainingList)
        for tl in trainingList:
            fragmentCount += 1
            print('%s | iter %03d --> training fragment %d / %d'%(modelName, it, fragmentCount, totalFrag))
            X_train, Y_train = dr_keras.readTrainingFragment(datapath, fragList=tl, imgSize=imgSize, meanImage=meanImage, classNum=classNum)
            model.fit(X_train, Y_train, batch_size, nb_epoch=1)
        print('%s | iter %03d --> start validation'%(modelName, it))
        evaLoss = []
        evaAcc = []
        fragmentCount = 0
        totalFrag = len(validationList)
        for vl in validationList:
            fragmentCount += 1
            print('%s | iter %03d --> validation fragment %d / %d'%(modelName, it, fragmentCount, totalFrag))
            X_valid, Y_valid = dr_keras.readTrainingFragment(datapath, fragList=vl, imgSize=imgSize, meanImage=meanImage, classNum=classNum)
            loss, acc = model.evaluate(X_valid, Y_valid, batch_size=batch_size)
            evaLoss.append( loss )
            evaAcc.append( acc )
        evaLoss = np.mean(evaLoss)
        evaAcc = np.mean(evaAcc)
        print('%s | iter %03d --> validation loss: %f, acc: %f'%(modelName, it, evaLoss, evaAcc))
        if evaLoss < minValidLoss:
            if os.path.exists(savePrefix + modelName + '_exp%d_'%(expIter) + 'keras_weights_best_vLoss%.5f_vAcc%.3f.h5'%(minValidLoss,maxValidAcc)):
                os.remove(savePrefix + modelName + '_exp%d_'%(expIter) + 'keras_weights_best_vLoss%.5f_vAcc%.3f.h5'%(minValidLoss,maxValidAcc))
            minValidLoss = evaLoss
            maxValidAcc = evaAcc
            model.save_weights(savePrefix + modelName + '_exp%d_'%(expIter) + 'keras_weights_best_vLoss%.5f_vAcc%.3f.h5'%(minValidLoss,maxValidAcc), overwrite=True)
                 
    return minValidLoss, maxValidAcc
'''========================================================================================================'''

isServer = False
if not isServer:
    pcProjectpath = '/home/liuzheng/competition/kaggle/distractedDrivers/'
#    mxnetRoot = '/home/liuzheng/toolbox/mxnet/'

cropMode = 'entire'
colorMode = 'rgb'
reSize = [224, 224]

modelName = 'vgg_keras'

datapath = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/trainAugmentation_'+colorMode + '_' + cropMode+'_'+'%03d'%(reSize[0])+'/'
testpath = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/testAugmentation_'+colorMode + '_' + cropMode+'_'+'%03d'%(reSize[0])+'/'
savePrefix=pcProjectpath + colorMode + '_' + cropMode+'_'+'%03d'%(reSize[0])+'/'
if not os.path.exists(savePrefix):
    os.mkdir(savePrefix)
meanImagePath = savePrefix + 'meanImage.npy'

classNum = 10
## total epock = num_iter * saveEpock, every iter trains saveEpock epoches
# num_iter = 10
# saveEpoch = 10
batch_size = 32



#history = trainModel(datapath, imgSize=(1,224,224), nb_iter=num_iter, batch_size=batch_size,\
#                            savePrefix=savePrefix, colorMode=colorMode, validSplit=0.3, saveEpoch=saveEpoch)

for i in range(3):
    trainModel_fragment(datapath, expIter=i+1, imgSize=(3,224,224), classNum=10, nb_iter=50, modelName=modelName,\
                optimizer='sgd', savePrefix=savePrefix, validSplit=0.3, fragSize=64, meanImagePath=meanImagePath, testpath=testpath)

#with open('history.pickle', 'wb') as f:
#    pickle.dump(history, f)

























