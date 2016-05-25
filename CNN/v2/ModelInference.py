# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:16:45 2016

@author: liuzheng
"""

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta

from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization

import sys
import numpy as np

def fromJson(architectureFilepath, weightsFilepath, optimizer='sgd'):
    model = model_from_json(open(architectureFilepath).read())
    model.load_weights(weightsFilepath)
    
    if optimizer == 'sgd':
        opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def inference(input_shape, classNum):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(classNum))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    print('compiling model....')
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

def inference_xavier_prelu_sgd_64(input_shape, classNum):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape, init='glorot_normal'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(32, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='glorot_normal'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(64, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128, 3, 3, border_mode='valid', init='glorot_normal'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(128, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(classNum))
    model.add(Activation('softmax'))
    
#    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    print('compiling model....')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

def inference_xavier_prelu_sgd_224(input_shape, classNum):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape, init='glorot_normal'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(32, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='glorot_normal'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(64, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128, 3, 3, border_mode='valid', init='glorot_normal'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(128, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(256, 3, 3, border_mode='valid', init='glorot_normal'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(256, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(256, 3, 3, border_mode='valid', init='glorot_normal'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(256, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(classNum))
    model.add(Activation('softmax'))
    
#    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    print('compiling model....')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

def vgg_std16_model(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights('/home/liuzheng/competition/kaggle/distractedDrivers/gray_manualCrop/vgg16_weights.h5')

    # Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(10, activation='softmax'))
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model

def caffeModel2keras(caffe_root, modelDefinitionPath, modelWeightsPath, saveConv=True,\
                    saveFC=False, inputSize=(1, 224,224), lrnSubstitution='dropout', classNum=10, optimizer='sgd'):
    sys.path.insert(0, caffe_root + 'python')
    import caffe
    
    net = caffe.Net(modelDefinitionPath,      # defines the structure of the model
                modelWeightsPath,  # contains the trained weights
                caffe.TRAIN)     # use test mode (e.g., don't perform dropout)
                
    netLayers = np.array(net.layers)
    layerTypeList = []
    
    kerasModel = Sequential()
    fcStart = False
    start = True
    for i in range(len(netLayers)):
        layerTypeList.append(netLayers[i].type)
        
        if netLayers[i].type == 'Convolution':
            w = netLayers[i].blobs[0].data
            b = netLayers[i].blobs[1].data
            if start:
                if inputSize[0] == 1:
                    ww = np.zeros((w.shape[0], 1, w.shape[2], w.shape[3]))
                    ww[:, 0, ...] = w[:, 0, ...]
                    w = ww
                
                start = False
                if saveConv:
                    kerasModel.add(Convolution2D(w.shape[0], w.shape[2], w.shape[3],\
                                        weights=[w, b],\
                                        border_mode='valid', input_shape=inputSize) )
                else:
                    kerasModel.add(Convolution2D(w.shape[0], w.shape[2], w.shape[3], init='he_uniform',\
                                        border_mode='valid', input_shape=inputSize) )
            else:
                if saveConv:
                    kerasModel.add(Convolution2D(w.shape[0], w.shape[2], w.shape[3],\
                                        weights=[w, b],\
                                        border_mode='valid') )
                else:
                    kerasModel.add(Convolution2D(w.shape[0], w.shape[2], w.shape[3],\
                                        border_mode='valid', init='he_uniform') )
#            convWeightsList.append(w)
#            convBiasList.append(b)
        elif netLayers[i].type == 'ReLU':
            kerasModel.add(LeakyReLU(alpha=0.1))
        elif netLayers[i].type == 'LRN':
            if lrnSubstitution == 'dropout':
                kerasModel.add(Dropout(0.25))
            elif lrnSubstitution == 'batchNorm':
                kerasModel.add(BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None,\
                                                beta_init='zero', gamma_init='one'))
        elif netLayers[i].type == 'Pooling':
            kerasModel.add(MaxPooling2D(pool_size=(2, 2)))
        elif netLayers[i].type == 'InnerProduct':
            if not fcStart:
                fcStart = True
                kerasModel.add(Flatten())
            w = netLayers[i].blobs[0].data
            b = netLayers[i].blobs[1].data
            if not netLayers[i+1].type == 'Softmax':
                if not saveFC:
                    kerasModel.add(Dense(w.shape[0], init='he_uniform'))
                else:
                    w = w.swapaxes(0,1)
                    kerasModel.add(Dense(w.shape[0], weights=[w, b]))
            else:
                if not saveFC:
                    kerasModel.add(Dense(classNum, init='he_uniform'))
                else:
                    w = w.swapaxes(0,1)
                    kerasModel.add(Dense(classNum, weights=[w, b]))
        elif netLayers[i].type == 'Softmax':
            kerasModel.add(Activation('softmax'))
    
    del net
    del netLayers    
    
    if optimizer == 'sgd':
        opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    
    kerasModel.compile(loss='categorical_crossentropy', optimizer=opt)

    return kerasModel


























