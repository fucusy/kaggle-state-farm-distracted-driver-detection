# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:45:34 2016

@author: liuzheng
"""
import sys
sys.path.append('../')

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta
from keras.callbacks import ModelCheckpoint

from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization

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
    def __init__(self,
                 model_name, data_set, test_batch_size=64, n_iter=50, model_arch_file='',
                 model_weights_file='', model_save_path='', prediction_save_file=''):
        self._model_name = model_name
        self._data_set = data_set
        # self._data_set._batch_size = batch_size
        self._n_iter = n_iter
        self._model_arch_file = model_arch_file
        self._model_weights_file = model_weights_file
        self._model_save_path = model_save_path
        if model_name == 'vgg_keras':
            img_size = data_set.get_img_size()
            self._model = vgg_std16_model(img_rows=img_size[1], img_cols=img_size[2], color_type=3,
                                          model_weights_file=model_weights_file, continueFile='')
        elif model_arch_file:
            self._model = model_from_json(open(self._model_arch_file).read())
            self._model.load_weights(self._model_weights_file)
            print('compiling model %s.....'%(self._model_name))
            self._model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            print('Warning: Model Architecture is not defined!')
        
        self._test_batch_size = test_batch_size
        self._prediction_save_file = prediction_save_file
        self._prediction = {}
    
    def set_model_arch(self, model_arch):
        self._model = model_arch
    
    def set_model_weights(self, model_weights_file=''):
        self._model.load_weights(model_weights_file)
    
    def save_model_arch(self, arch_path_file):
        json_string = self._model.to_json()
        open(arch_path_file, 'w').write(json_string)
    
    def save_model_weights(self, weights_path_file, overwrite=True):
        self._model.save_weights(weights_path_file, overwrite=overwrite)
    
    def train_model(self, save_best=True):
        json_string = self._model.to_json()
        open(self._model_save_path + self._model_name + 'keras_arch.json', 'w').write(json_string)
        if self._data_set.need_training_fragment:
            '''If data fragment is needed.'''
            minValidLoss = np.inf
            maxValidAcc = 0.
            
            for it in range(self._n_iter):
                '''Every iter, we read fragments one by one and train model with them setting nb_epoch=1.'''
                fragment_count = 0
                while self._data_set.have_next_training_fragment_reset:
                    fragment_count += 1
                    print('%s | iter %03d --> validation fragment %d / %d'%(self._model_name,
                                           it, fragment_count, self._data_set.get_training_fragment_num()))
                    x_train, y_train = self._data_set.next_training_fragment(validation_flag=False)
                    self._model.fit(x_train, y_train, batch_size=self._data_set._batch_size, nb_epoch=1)
                ''' After all training fragments are trained, we read validation fragments and evaluate the model on them. '''
                evaLoss = []
                evaAcc = []
                fragment_count = 0
                while self._data_set.have_next_validation_fragment_reset:
                    fragment_count += 1
                    print('%s | iter %03d --> validation fragment %d / %d'%(self._model_name,
                                           it, fragment_count, self._data_set.get_validation_fragment_num()))
                    x_valid, y_valid = self._data_set.next_training_fragment(validation_flag=True)
                    loss, acc = self._model.evaluate(x_valid, y_valid, batch_size=self._data_set._batch_size)
                    evaLoss.append( loss )
                    evaAcc.append( acc )
                ''' Compute mean evaluation loss and accuracy and output them.'''
                evaLoss = np.mean(evaLoss)
                evaAcc = np.mean(evaAcc)
                print('%s | iter %03d --> validation loss: %f, acc: %f'%(self._model_name, it+1, evaLoss, evaAcc))
                ''' Save the best model. '''
                if evaLoss < minValidLoss and save_best:
                    if os.path.exists(self._model_save_path + self._model_name + 'keras_weights_best_vLoss%.5f_vAcc%.3f.h5'%(minValidLoss,maxValidAcc)):
                        os.remove(self._model_save_path + self._model_name + 'keras_weights_best_vLoss%.5f_vAcc%.3f.h5'%(minValidLoss,maxValidAcc))
                    minValidLoss = evaLoss
                    maxValidAcc = evaAcc
                    self._model.save_weights(self._model_save_path + self._model_name + 'keras_weights_best_vLoss%.5f_vAcc%.3f.h5'%(minValidLoss,maxValidAcc), overwrite=True)
            
            self._model.save_weights(self._model_save_path + self._model_name + 'keras_weights_final.h5', overwrite=True)
            
        else:
            ''' If no need to fragment, load all training images and train the model directly.'''
            print('start training')
            CallBacks = []
            if save_best:
                checkpoint = ModelCheckpoint(self._model_save_path + self._model_name + 'keras_checkpoint_weights_best.h5',
                                            monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
                CallBacks.append(checkpoint)

            x_train, y_train = self._data_set.load_traininig_data()
            self._history = self._model.fit(x_train, y_train, batch_size=self._data_set._batch_size, nb_epoch=self._n_iter,
                                            validation_split=self._data_set._validation_split, callbacks=CallBacks)

    def predict_model(self):
        fragment_count = -1
        while self._data_set.have_next_testing_fragment_reset:
            fragment_count += 1
            print('%s | --> testing fragment %d / %d'%(self._model_name,
                                   fragment_count, self._data_set.get_testing_fragment_num()))
            x_test, name_list = self._data_set.next_testing_fragment()
            frag_prediction = self._model.predict_proba(x_test, batch_size=self._test_batch_size)
            self.stat_prediction(frag_prediction, name_list)
        ''' We still call fuse function if only one prediction per image for universality.
            The self._prediction is a dict that every key (testing image names) has a list of prediction.
            No matter how many elements the list has, the final prediction is a numpy array for each image after fusing.
        '''
        self.fuse_prediction()
        self.save_prediction()

    ''' Update prediction dict. '''
    def  stat_prediction(self, frag_prediction, frag_list):
        for i in range(frag_prediction.shape[0]):
            imgName = frag_list[i][:frag_list[i].rfind('_')]
            if imgName in self._prediction.keys():
                self._prediction[imgName].append(frag_prediction[i, :])
            else:
                self._prediction[imgName] = [frag_prediction[i, :]]

    ''' In case some methods give multiple predictions to a testing image, this function fuse all predictions into one. '''
    def fuse_prediction(self):
        finalDict = {}
        for k in self._prediction.keys():
            finalDict[k] = np.mean(np.array(self._prediction[k]), axis=0)
        self._prediction = finalDict

    ''' Save prediction dict '''
    def save_prediction(self):
        fileObj = open(self._prediction_save_file, 'wb')
        writer = csv.writer(fileObj)
        writer.writerow(['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
        keys = self._prediction.keys()
        keys.sort()
        for k in keys:
            line = [k+'.jpg']
            for i in range(10):
                line.append('%.3f'%(self._prediction[k][i]))
            writer.writerow(line)
        fileObj.close()












