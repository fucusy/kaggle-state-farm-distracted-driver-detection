# -*- coding: utf-8 -*-
"""
Created on Sat May 14 09:58:44 2016

@author: WuPeng
"""
import sys
# add project root to python lib search path
sys.path.append("../../")

import config
import os
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import  Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D

from keras.optimizers import SGD
from keras.models import model_from_json
from CNN.keras_tool import *

img_rows, img_cols, color_type = 224, 224, 3

def save_model(model, index, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    model_filename = 'architecture_' + str(index) + cross + '.json'
    weights_filename = 'model_weights_' + str(index) + cross + '.h5'
    open(os.path.join('cache', model_filename), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weights_filename), overwrite=True)


def read_model(index, cross=''):
    model_filename = 'architecture_' + str(index) + cross + '.json'
    weights_filename = 'model_weights_' + str(index) + cross + '.h5'
    model = model_from_json(open(os.path.join('cache', model_filename)).read())
    model.load_weights(os.path.join('cache', weights_filename))
    return model

def VGG_16(weights_path=None):
    # standard VGG16 network architecture
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

    # replace last fc layer
    model.layers.pop()
    model.add(Dense(10, activation='softmax'))
    # load model weights
    if weights_path:
        model.load_weights(weights_path)


    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model

def train(nb_epoch=10,  model_desc=''):
    # Now it loads color image
    # input image dimensions

    batch_size = 64
    random_state = 20

    model = VGG_16(config.CNN.keras_train_weight)
    data_set = load_train_data_set(config.Project.train_img_folder_path)

    count = 0
    for i in range(nb_epoch):
        data_set.reset_index()
        while data_set.have_next():
            img_list, img_label, _ = data_set.next_batch(batch_size, True)
            img_label_cate = to_categorical(img_label, 10)
            if count % 10 == 0:
                loss_and_metrics = model.evaluate(img_list, img_label_cate, batch_size=batch_size)
                print(loss_and_metrics)
            model.train_on_batch(img_list, img_label_cate)
            count += 1
            
        print('end saving model............')
        save_model(model, model_desc)

if __name__ == '__main__':
    train(nb_epoch=3, model_desc='_vgg_16_2x20')
