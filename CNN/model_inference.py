# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 09:42:56 2016

@author: liuzheng

    File for contributers to define their DIY models.
    It is appreciated if contributers follow the same parameter list.
"""
import sys
sys.path.append('../')

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adam, Adagrad, RMSprop

from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization

import numpy as np

def load_conv_weights(model, filepath, need_layer_num):
    '''Load all layer weights from a HDF5 save file.
    '''
    import h5py
    f = h5py.File(filepath, mode='r')

    if not hasattr(model, 'flattened_layers'):
        print('Model Error!')
        exit()
    
        # support for legacy Sequential/Merge behavior
        # flattened_layers = model.flattened_layers
#    else:
#        flattened_layers = model.layers

    if 'nb_layers' in f.attrs:
        # legacy format
        nb_layers = f.attrs['nb_layers']
        if need_layer_num > nb_layers:
            print('Stop layer out of layer index!')
            exit()
        
#        if nb_layers != len(model.flattened_layers):
#            raise Exception('You are trying to load a weight file '
#                            'containing ' + str(nb_layers) +
#                            ' layers into a model with ' +
#                            str(len(model.flattened_layers)) + '.')

        for k in range(need_layer_num):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.flattened_layers[k].set_weights(weights)
    f.close()
    return model

def vgg_std16_extractor(input_shape, class_num, optimizer='sgd', weights_file=''):
    if not weights_file:
        print('Extractor need a weight file!')
        exit()
    model = Sequential()
    if input_shape[1] < 32:
        print('Too small input!')
        exit()
    if input_shape[1] >= 32:
        need_layer_num = 10
        # 5 layers
        model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        print model.output_shape
        # 5 layers
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        print model.output_shape
    if input_shape[1] >= 64:
        # 7 layers
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        print model.output_shape
    if input_shape[1] >= 112:
        # 7 layers
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        print model.output_shape
    if input_shape[1] >= 224:
        # 7 layers
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        print model.output_shape
    # 6layers
#    model.add(Flatten(name='flatten'))
#    model.add(Dense(4096, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(4096, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(1000, activation='softmax'))
    print('loading orininal vgg weights for extractor')
    model = load_conv_weights(model, weights_file, need_layer_num=need_layer_num)
    
    print('model output shape before flatten:')
    print model.output_shape
    model.add(Flatten(name='flatten'))
    print('model output shape after flatten:')
    print model.output_shape

    if optimizer == 'sgd':
        opt = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer == 'adagrad':
        opt = Adagrad(lr=0.01, epsilon=1e-08)
    elif optimizer == 'adadelta':
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    print('vgg model compiling')
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def vgg_std16_model(input_shape, class_num, optimizer='sgd', weights_file=''):
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
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

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    
    if weights_file:
        print('loading orininal vgg weights')
        model.load_weights(weights_file)
        #model = load_conv_weights(model, weights_file, stop_layer=10)

    # Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(class_num, activation='softmax'))
#    if continueFile:
#        print('load trained model to continue')
#        model.load_weights(continueFile)
    # Learning rate is changed to 0.001
    if optimizer == 'sgd':
        opt = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer == 'adagrad':
        opt = Adagrad(lr=0.01, epsilon=1e-08)
    elif optimizer == 'adadelta':
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    print('vgg model compiling')
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
def vgg_std16_model_modified_dense(input_shape, class_num, optimizer='sgd', weights_file=''):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(input_shape[0],
                                                 input_shape[1], input_shape[2])))
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

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    
    if weights_file:
        print('loading orininal vgg weights')
        model.load_weights(weights_file)

    # Code above loads pre-trained data and
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
#    if continueFile:
#        print('load trained model to continue')
#        model.load_weights(continueFile)
    # Learning rate is changed to 0.001
    if optimizer == 'sgd':
        opt = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer == 'adagrad':
        opt = Adagrad(lr=0.01, epsilon=1e-08)
    elif optimizer == 'adadelta':
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    print('vgg model compiling')
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model    

def vgg_std16_32(input_shape, class_num, optimizer='sgd', weights_file=''):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(input_shape[0],
                                                 input_shape[1], input_shape[2])))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    print(model.output_shape)

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    print(model.output_shape)

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    print(model.output_shape)

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
    
    if weights_file:
        print('loading orininal vgg weights')
        model.load_weights(weights_file)

#    model.add(Flatten(name='flatten'))
#    model.add(Dense(4096, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(4096, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(1000, activation='softmax'))
#    
#    if weights_file:
#        print('loading orininal vgg weights')
#        model.load_weights(weights_file)

    # Code above loads pre-trained data and
    for i in range(14):
        model.layers.pop()
    
    print(model.output_shape)
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
#    if continueFile:
#        print('load trained model to continue')
#        model.load_weights(continueFile)
    # Learning rate is changed to 0.001
    if optimizer == 'sgd':
        opt = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer == 'adagrad':
        opt = Adagrad(lr=0.01, epsilon=1e-08)
    elif optimizer == 'adadelta':
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    print('vgg model compiling')
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model   

def inference_64(input_shape, class_num, optimizer='sgd', weights_file=''):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(32, 3, 3, init='he_uniform', activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(32, 3, 3, init='he_uniform', activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, init='he_uniform', activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, init='he_uniform', activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
#    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
#    model.add(Convolution2D(128, 3, 3, init='he_uniform', activation='relu'))
#    model.add(ZeroPadding2D((1, 1)))
#    model.add(Convolution2D(128, 3, 3, init='he_uniform', activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
    
    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(class_num))
    model.add(Activation('softmax'))
    
    if weights_file:
        model.load_weights(weights_file)
    
    if optimizer == 'sgd':
        opt = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer == 'adagrad':
        opt = Adagrad(lr=0.01, epsilon=1e-08)
    elif optimizer == 'adadelta':
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    print('compiling model....')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

def inference_32(input_shape, class_num, optimizer='sgd', weights_file=''):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(16, 3, 3, init='he_uniform', activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3, init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    print(model.output_shape)
    
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(32, 3, 3, init='he_uniform', activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, init='he_uniform', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    print(model.output_shape)
    
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(32, 3, 3, init='he_uniform', activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, init='he_uniform', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    print(model.output_shape)
    
#    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
#    model.add(Convolution2D(128, 3, 3, init='he_uniform', activation='relu'))
#    model.add(ZeroPadding2D((1, 1)))
#    model.add(Convolution2D(128, 3, 3, init='he_uniform', activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
#    print(model.output_shape)
    
    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(class_num))
    model.add(Activation('softmax'))
    
    if weights_file:
        model.load_weights(weights_file)
    
    if optimizer == 'sgd':
        opt = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer == 'adagrad':
        opt = Adagrad(lr=0.01, epsilon=1e-08)
    elif optimizer == 'adadelta':
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    print('compiling model....')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

def inference_xavier_prelu_224(input_shape, class_num, optimizer='sgd', weights_file=''):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape, init='he_uniform'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(32, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='he_uniform'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(64, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128, 3, 3, border_mode='valid', init='he_uniform'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(128, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(256, 3, 3, border_mode='valid', init='he_uniform'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(256, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128, 3, 3, border_mode='valid', init='he_uniform'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(128, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(class_num))
    model.add(Activation('softmax'))
    
    if weights_file:
        model.load_weights(weights_file)
#    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    if optimizer == 'sgd':
        opt = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer == 'adagrad':
        opt = Adagrad(lr=0.01, epsilon=1e-08)
    elif optimizer == 'adadelta':
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    print('compiling model....')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

def inference_xavier_prelu_112(input_shape, class_num, optimizer='sgd', weights_file=''):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape, init='he_uniform'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(32, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(32, 3, 3, border_mode='valid', init='he_uniform'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(32, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='he_uniform'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(64, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128, 3, 3, border_mode='valid', init='he_uniform'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(128, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
#    model.add(Convolution2D(128, 3, 3, border_mode='valid', init='he_uniform'))
#    model.add(PReLU(init='zero', weights=None))
#    model.add(Convolution2D(128, 3, 3))
#    model.add(PReLU(init='zero', weights=None))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
    
    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
#    model.add(Dense(128))
#    model.add(Activation('relu'))
#    model.add(Dropout(0.25))
    
    model.add(Dense(class_num))
    model.add(Activation('softmax'))
    
    if weights_file:
        model.load_weights(weights_file)
#    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    if optimizer == 'sgd':
        opt = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer == 'adagrad':
        opt = Adagrad(lr=0.01, epsilon=1e-08)
    elif optimizer == 'adadelta':
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    print('compiling model....')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model

def inference_less_filter(input_shape, class_num, optimizer='sgd', weights_file=''):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape, init='he_uniform'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(32, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(32, 3, 3, border_mode='valid', init='he_uniform'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(32, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='he_uniform'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(64, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='he_uniform'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(64, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(128, 3, 3, border_mode='valid', init='he_uniform'))
    model.add(PReLU(init='zero', weights=None))
    model.add(Convolution2D(128, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(class_num))
    model.add(Activation('softmax'))
    
    if weights_file:
        model.load_weights(weights_file)
#    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    if optimizer == 'sgd':
        opt = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer == 'adagrad':
        opt = Adagrad(lr=0.01, epsilon=1e-08)
    elif optimizer == 'adadelta':
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    print('compiling model....')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def inference_dense(input_dim, class_num, optimizer='sgd', weights_file=''):
    model = Sequential()
    
    model.add(Dense(2048, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
#    model.add(Dense(256))
#    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
    
    model.add(Dense(class_num))
    model.add(Activation('softmax'))
    
    if weights_file:
        model.load_weights(weights_file)
#    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    if optimizer == 'sgd':
        opt = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer == 'adagrad':
        opt = Adagrad(lr=0.01, epsilon=1e-08)
    elif optimizer == 'adadelta':
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    print('compiling model....')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model




















