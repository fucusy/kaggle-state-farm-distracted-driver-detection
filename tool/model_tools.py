# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:45:34 2016

@author: liuzheng
"""
import sys
sys.path.append('../')

import config

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta
from keras.callbacks import ModelCheckpoint

from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
import logging
import numpy as np
import csv
import skimage.io as skio
import scipy.io as sio
import os

from keras import backend as K

def vgg_std16_model(img_rows, img_cols, color_type=1, model_weights_file='', continueFile='', optimizer='sgd'):
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

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    
    if not continueFile:
        print('loading orininal vgg weights')
        model.load_weights(model_weights_file)

    # Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(10, activation='softmax'))
    if continueFile:
        print('load trained model to continue')
        model.load_weights(continueFile)
    # Learning rate is changed to 0.001
    if optimizer == 'sgd':
        opt = SGD(lr=1.5e-3, decay=1e-6, momentum=0.9, nesterov=True)
    print('vgg model compiling')
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class KerasFeatureExtractor(object):
    '''
        Using existing model to extract CNN features.
        This class is mainly for the Sequential Keras Model, instead of the Keras Function API Model,
        which means the feature layer index must be given.
        The differences between them can be found in keras.io.
    '''
    def __init__(self,
                 model_name, data_set, model_arch_file='', model_weights_file='', feature_layer_index=-1, feature_save_path=''):
        self._model_name = model_name
        self._data_set = data_set
        self._model_arch_file = model_arch_file
        self._model_weights_file = model_weights_file
        self._feature_layer_index = feature_layer_index
        self._feature_save_path = feature_save_path
        if model_name == 'vgg_keras':
            img_size = data_set.get_img_size()
            self._model = vgg_std16_model(img_rows=img_size[1], img_cols=img_size[2], color_type=3,
                                          model_weights_file=model_weights_file, continueFile='')
        elif model_arch_file:
            self._model = model_from_json(open(self._model_arch_file).read())
            self._model.load_weights(self._model_weights_file)
            print('compiling model %s.....'%(self._model_name))
            self._model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        self._extractor = K.function([self._model.layers[0].input, K.learning_phase()],
                                  [self._model.layers[feature_layer_index].output])

    def extract_feature_images(self, images, save_file=''):
        features = self._extractor([images, 0])[0]
        if save_file:
            with open(self._feature_save_path + save_file, 'wb') as f:
                np.save(f, features)
        return features

    def extract_training_features(self):
        folder = os.listdir(self._data_set._training_data_path)
        total = len(folder)
        ch = self._data_set._img_size[0]
        trainingLabel = np.zeros((total), dtype=int)
        i = 0
        data = np.zeros((1, self._data_set._img_size[0], self._data_set._img_size[1], self._data_set._img_size[2]))
        for f in folder:
            i += 1
            print('get training feature %d / %d'%(i, total))
            trainingLabel[i-1] = np.int(f[0])
            img = skio.imread(self._data_set._training_data_path+f)
            if ch == 3:
                img = img.swapaxes(1, 2)
                img = img.swapaxes(0, 1)
            data[0, ...] = img - self._data_set._mean_image
            # output in test mode = 0
            feat = self._extractor([data, 0])[0]
            if i == 1:
                trainingData = np.zeros((total, feat.shape[1]))
                trainingData[i-1, :] = feat
            else:
                trainingData[i-1, :] = feat
        with open(self._feature_save_path + 'training_features.npy', 'wb') as f:
            np.save(f, trainingData)
        with open(self._feature_save_path + 'training_labels.npy', 'wb') as f:
            np.save(f, trainingLabel)
#            sio.savemat(f, {'trainigData':trainingData, 'trainingLabel':trainingLabel})

    def extract_testing_features(self):
        folder = os.listdir(self._data_set._testing_data_path)
        total = len(folder)
        i = 0
        imgNames = []
        ch = self._data_set._img_size[0]
        data = np.zeros((1, self._data_set._img_size[0], self._data_set._img_size[1], self._data_set._img_size[2]))
        for f in folder:
            i += 1
            print('get testing feature %d / %d'%(i, total))
            img = skio.imread(self._data_set._testing_data_path+f)
            if ch == 3:
                img = img.swapaxes(1, 2)
                img = img.swapaxes(0, 1)
            data[0, ...] = img - self._data_set._mean_image
            # output in test mode = 0
            feat = self._extractor([data, 0])[0]
            imgName = f[:f.rfind('_')] + '.jpg'
            imgNames.append(imgName)
            if i == 1:
                features = np.zeros((total, feat.shape[1]))
                features[i-1, :] = feat
            else:
                features[i-1, :] = feat
        
        with open(self._feature_save_path + 'testing_features.npy', 'wb') as f:
            np.save(f, features)
        with open(self._feature_save_path + 'testing_names.npy', 'wb') as f:
            np.save(f, np.array(imgNames))
#        with open(self._feature_save_path + 'testing_features.mat', 'wb') as f:
#            sio.savemat(f, {'testingData':features, 'imageName':imgNames})



class KerasModel(object):
    def __init__(self, cnn_model):

        model_name = config.CNN.model_name
        test_batch_size = config.CNN.test_batch_size
        n_iter = config.CNN.train_iter
        model_arch_file = config.CNN.model_arch_file_name
        model_weights_file = config.CNN.model_weights_file_name
        model_save_path = config.CNN.model_save_path
        prediction_save_file = config.CNN.prediction_save_file


        self._model_name = model_name
        self._n_iter = n_iter
        self._model_arch_file = model_arch_file
        self._model_weights_file = model_weights_file
        self._model_save_path = model_save_path
        self._batch_size = config.CNN.batch_size
        self._test_batch_size = test_batch_size

        self._prediction_save_file = prediction_save_file
        self._prediction = {}
        self._model = cnn_model
    
    def set_model_arch(self, model_arch):
        self._model = model_arch
    
    def set_model_weights(self, model_weights_file=''):
        self._model.load_weights(model_weights_file)
    
    def save_model_arch(self, arch_path_file):
        json_string = self._model.to_json()
        open(arch_path_file, 'w').write(json_string)
    
    def save_model_weights(self, weights_path_file, overwrite=True):
        self._model.save_weights(weights_path_file, overwrite=overwrite)
    
    def train_model(self, train_data, validation_data, save_best=True):
        json_string = self._model.to_json()
        json_path = os.path.join(self._model_save_path, self._model_name + '.json')
        open(json_path, 'w').write(json_string)

        fragment_size = config.CNN.load_image_to_memory_every_time
        if fragment_size > 0:
            '''If data fragment is needed.'''
            min_loss = np.inf
            max_acc = 0.
            
            for it in range(self._n_iter):
                '''Every iter, we read fragments one by one and train model with them setting nb_epoch=1.'''
                image_count = 0

                # set index to zero, prepare for have_next function
                train_data.reset_index()

                while train_data.have_next():
                    x_train, y_train, _ = train_data.next_fragment(fragment_size,need_label=True)
                    image_count += fragment_size
                    logging.info('%s | iter %03d --> validation fragment %d / %d'
                                   % (self._model_name, it, image_count, train_data.count()))

                    self._model.fit(x_train, y_train, batch_size=self._batch_size, nb_epoch=1)
                ''' After all training fragments are trained, we read validation fragments and evaluate the model on them. '''

                eva_loss = []
                eva_acc = []
                image_count = 0

                # set index to zero, prepare for have_next function
                validation_data.reset_index()

                while validation_data.have_next():
                    image_count += fragment_size
                    print('%s | iter %03d --> validation fragment %d / %d'
                          % (self._model_name, it, image_count, validation_data.count()))

                    x_valid, y_valid, _ = validation_data.next_fragment(fragment_size,need_label=True)

                    loss, acc = self._model.evaluate(x_valid, y_valid, batch_size=self._batch_size)
                    eva_loss.append(loss)
                    eva_acc.append(acc)
                ''' Compute mean evaluation loss and accuracy and output them.'''
                eva_loss = np.mean(eva_loss)
                eva_acc = np.mean(eva_acc)
                print('%s | iter %03d --> validation loss: %f, acc: %f'%(self._model_name, it, eva_loss, eva_acc))
                ''' Save the best model. '''
                if eva_loss < min_loss and save_best:
                    old_weight_path = os.path.join(self._model_save_path, self._model_name + 'keras_weights_best_vLoss%.5f_vAcc%.3f.h5'%(min_loss,max_acc))
                    if os.path.exists(old_weight_path):
                        os.remove(old_weight_path)
                    min_loss = eva_loss
                    max_acc = eva_acc
                    new_weight_path = os.path.join(self._model_save_path, self._model_name + 'keras_weights_best_vLoss%.5f_vAcc%.3f.h5'%(min_loss,max_acc))
                    self._model.save_weights(new_weight_path, overwrite=True)
            final_weight_path = os.path.join(self._model_save_path, self._model_name + 'keras_weights_final.h5')
            self._model.save_weights(final_weight_path, overwrite=True)
            
        else:
            ''' If no need to fragment, load all training images and train the model directly.'''
            logging.info('start training')
            CallBacks = []
            if save_best:
                weight_path = os.path.join(self._model_save_path, self._model_name + 'keras_checkpoint_weights_best.h5')
                checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
                CallBacks.append(checkpoint)

            x_train, y_train, _ = train_data.load_all_image(need_label=True)
            x_vali, y_vali, _ = validation_data.load_all_image(need_label=True)

            self._history = self._model.fit(x_train, y_train, batch_size=self._batch_size, nb_epoch=self._n_iter,
                                             validation_data=(x_vali, y_vali), callbacks=CallBacks)

    def predict_model(self, test_data):
        image_count = 0
        fragment_size = config.CNN.load_image_to_memory_every_time

        while test_data.have_next():
            image_count += fragment_size
            print('%s | --> testing fragment %d / %d'%(self._model_name,
                                   image_count, test_data.count()))


            x_test, name_list = test_data.next_fragment(fragment_size, need_label=False)
            frag_prediction = self._model.predict_proba(x_test, batch_size=self._test_batch_size)
            self.stat_prediction(frag_prediction, name_list)
        ''' We still call fuse function if only one prediction per image for universality.
            The self._prediction is a dict that every key (testing image names) has a list of prediction.
            No matter how many elements the list has, the final prediction is a numpy array for each image after fusing.
        '''
        self.fuse_prediction()
        self.save_prediction()

    ''' Update prediction dict. '''
    def stat_prediction(self, frag_prediction, frag_list):
        for i in range(frag_prediction.shape[0]):
            img_name = frag_list[i]
            if img_name in self._prediction.keys():
                self._prediction[img_name].append(frag_prediction[i, :])
            else:
                self._prediction[img_name] = [frag_prediction[i, :]]

    ''' In case some methods give multiple predictions to a testing image, this function fuse all predictions into one. '''
    def fuse_prediction(self):
        finalDict = {}
        for k in self._prediction.keys():
            self._prediction[k] = np.mean(np.array(self._prediction[k]), axis=0)


    ''' Save prediction dict '''
    def save_prediction(self):
        file_obj = open(self._prediction_save_file, 'wb')
        writer = csv.writer(file_obj)
        writer.writerow(['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
        keys = self._prediction.keys()
        keys.sort()
        for k in keys:
            line = [k]
            for i in range(10):
                line.append('%.3f' % (self._prediction[k][i]))
            writer.writerow(line)
        file_obj.close()