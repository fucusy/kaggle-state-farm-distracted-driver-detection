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
from tool.keras_tool import *
from tool.file import generate_result_file
import logging

img_rows, img_cols, color_type = 224, 224, 3

def VGG_16(lr=1e-3, weights_path=None):
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

    weights_path_exist = False
    if weights_path is not None and os.path.exists(weights_path):
        weights_path_exist = True

    if weights_path is None or not weights_path_exist:
        logging.debug("load weigth from vgg weight")
        model.load_weights(config.CNN.vgg_weight_file_path)

    # replace last fc layer
    model.layers.pop()
    model.add(Dense(10, activation='softmax'))

    # load model weights
    if weights_path is not None and weights_path_exist:
        logging.debug("load weigth from fine-tuning weight %s" % weights_path)
        model.load_weights(weights_path)

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=['accuracy'])

    return model

def train_predict(nb_epoch=10, weights_path=None):
    # Now it loads color image
    # input image dimensions
    batch_size = 64
    model = VGG_16(lr=1e-3, weights_path=weights_path)
    data_set = load_train_data_set(config.Project.train_img_folder_path)

    count = 0
    total_count = data_set.num_examples * nb_epoch
    img_list, img_label, _ = data_set.load_all_image(need_label=True)
    img_label_cate = to_categorical(img_label, 10)
    model.fit(img_list, img_label_cate, batch_size=batch_size,
                    nb_epoch=nb_epoch, verbose=1, shuffle=False
                    ,validation_split=0.15)

    print('end saving model............')
    model.save_weights(weights_path, overwrite=True)

    test_data_set = load_test_data_set(config.Project.test_img_folder_path)
    predict = []
    
    while test_data_set.have_next():
        img_list, _ = test_data_set.next_batch(128)
        result = model.predict(img_list)
        predict += result
    predict = np.array(predict)
    generate_result_file(test_data_set.image_path_list[:len(predict)], predict)

if __name__ == '__main__':
    level = logging.DEBUG
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    train_predict(nb_epoch=5, weights_path=config.CNN.keras_train_weight)
