# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:32:52 2016

@author: liuzheng
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

import DataReader_KERAS as dr_keras

def inference():
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 64, 64)))
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
    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    print('compiling model....')
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model

def trainModel(datapath, isGenerator=True, nb_epoch=50, batch_size=32, savepath='', isContinue=False):
    if isContinue:
        model = model_from_json(open(savepath + 'keras_model_architecture.json').read())
        model.load_weights(savepath + 'keras_model_weights.h5')
        model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
	model = inference()
    
    X_train, Y_train = dr_keras.readTrainingImages_generator(datapath)
    
    datagen = ImageDataGenerator(featurewise_center=True,\
                                samplewise_center=False,\
                                featurewise_std_normalization=True,\
                                samplewise_std_normalization=False,\
                                zca_whitening=False,\
                                rotation_range=0.,\
                                width_shift_range=0.,\
                                height_shift_range=0.,\
                                shear_range=0.,\
                                #zoom_range=0.,\
                                #channel_shift_range=0.,\
                                #fill_mode='nearest',\
                                #cval=0.,\
                                #horizontal_flip=False,\
                                #vertical_flip=False,\
                                dim_ordering='th')
    print('generator is fitting data.....')
    datagen.fit(X_train)
    
    checkpoint = ModelCheckpoint(savepath+'keras_checkpoint.keras', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
    print('start training')
    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size),
                        samples_per_epoch=len(X_train), nb_epoch=nb_epoch, callbacks=[checkpoint])

    json_string = model.to_json()
    open(savepath + 'keras_model_architecture.json', 'w').write(json_string)
    model.save_weights(savepath + 'keras_model_weights.h5')
    return model, history




























