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
import progressbar as pb

from keras import backend as K


class KerasFeatureExtractor(object):
    '''
        Using existing model to extract CNN features.
        This class is mainly for the Sequential Keras Model, instead of the Keras Function API Model,
        which means the feature layer index must be given.
        The differences between them can be found in keras.io.
    '''
    def __init__(self,
                 model_name, data_set,
                 model_inference=                       [],
                 model_arch_file=                       '',
                 model_weight_file=                    '',
                 feature_layer_index=                   -1,
                 feature_folder=                        ''
                 #training_feature_file_prefix=          '',
                 #training_label_file_prefix=            '',
                 #testing_feature_file_prefix=           '',
                 #testing_name_file_prefix=              '',
                 #validation_feature_file_prefix=        '',
                 #validation_label_file_prefix=          ''
                 ):
        self._model_name = model_name
        self._data_set = data_set
        self._model_arch_file = model_arch_file
        self._model_weight_file = model_weight_file
        self._feature_layer_index = feature_layer_index
        
        #self._training_feature_file_prefix = training_feature_file_prefix
        #self._testing_feature_file_prefix = testing_feature_file_prefix
        #self._validation_feature_file_prefix = validation_feature_file_prefix
        
        #self._training_label_file_prefix = training_label_file_prefix
        #self._testing_name_file_prefix = testing_name_file_prefix
        #self._validation_label_file_prefix = validation_label_file_prefix
        
        self._feature_folder = feature_folder#training_feature_file_prefix[:training_feature_file_prefix.rfind('/')] + '/'
        if not os.path.exists(self._feature_folder):
            os.mkdir(self._feature_folder)
        
        
#        if model_name == 'vgg_keras':
#            img_size = data_set.get_img_size()
#            self._model = vgg_std16_model(img_rows=img_size[1], img_cols=img_size[2], color_type=3,
#                                          model_weights_file=model_weights_file, continueFile='')
        if model_inference:
            self._model = model_inference
        elif model_arch_file:
            self._model = model_from_json(open(self._model_arch_file).read())
            self._model.load_weights(self._model_weight_file)
            print('compiling model %s.....'%(self._model_name))
            self._model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            print('Warning: Model Architecture is not defined!')

        self._extractor = K.function([self._model.layers[0].input, K.learning_phase()],
                                  [self._model.layers[feature_layer_index].output])

    def extract_feature_images(self, images, save_file=''):
        features = self._extractor([images, 0])[0]
        if save_file:
            with open(save_file, 'wb') as f:
                np.save(f, features)
        return features

    def extract_training_features(self):
        fragment_count = -1
        while self._data_set.have_next_training_fragment_reset():
            fragment_count += 1
            print('%s | --> extracting training driver %d / %d'%(self._model_name,
                                   fragment_count+1, self._data_set.get_training_fragment_num()))
            x_train, y_train = self._data_set.next_training_fragment(validation_flag=False, for_extractor=False)
            fragment_size = x_train.shape[0]
            batch_size = self._data_set._batch_size
            batch_num = np.int( np.ceil( np.float(fragment_size) / batch_size ) )
            progress_bar = pb.ProgressBar(batch_num)
            progress_bar.start()
            for batch_i in range(batch_num):
                if batch_i == batch_num - 1:
                    batch = x_train[batch_i*batch_size:, ...]
                else:
                    batch = x_train[batch_i*batch_size:(batch_i+1)*batch_size, ...]
                batch_feat = self._extractor([batch, 0])[0]
#                if self._model_name == 'vgg_std16_model' and self._data_set._img_size[1] == 64 and self._data_set._padded_img_size:
#                    batch_feat = batch_feat[..., 10:18, 10:18]
#                    feat_len = batch_feat.shape[1]*batch_feat.shape[2]*batch_feat.shape[3]
#                    tmp = np.zeros((batch_feat.shape[0], feat_len))
#                    for i in range(tmp.shape[0]):
#                        tmp[i, :] = batch_feat[i, ...].reshape([feat_len])
#                    batch_feat = tmp
                if batch_i == 0:
                    feat = batch_feat
                else:
                    feat = np.concatenate((feat, batch_feat), axis=0)
                progress_bar.update(batch_i)
            progress_bar.finish()
            
            print('%s | --> saving feature training driver %d / %d'%(self._model_name,
                                  fragment_count+1, self._data_set.get_training_fragment_num()))
            feat = np.concatenate((feat, y_train), axis=1)
            with open(self._feature_folder + 'train_driver_data_label_%02d.npy'%(fragment_count), 'wb') as f:
                np.save(f, feat)


    def extract_testing_features(self):
        fragment_count = -1
        pre_useful_col = []
        while self._data_set.have_next_testing_fragment_reset():
            fragment_count += 1
            print('%s | --> extracting feature testing fragment %d / %d'%(self._model_name,
                                   fragment_count+1, self._data_set.get_testing_fragment_num()))
            x_test, name_list = self._data_set.next_testing_fragment()
            fragment_size = x_test.shape[0]
            batch_size = self._data_set._batch_size
            batch_num = np.int( np.ceil( np.float(fragment_size) / batch_size ) )
            progress_bar = pb.ProgressBar(batch_num)
            progress_bar.start()
            for batch_i in range(batch_num):
                if batch_i == batch_num - 1:
                    batch = x_test[batch_i*batch_size:, ...]
                else:
                    batch = x_test[batch_i*batch_size:(batch_i+1)*batch_size, ...]
                batch_feat = self._extractor([batch, 0])[0]
#                if self._model_name == 'vgg_std16_model' and self._data_set._img_size[1] == 64 and self._data_set._padded_img_size:
#                    batch_feat = batch_feat[..., 10:18, 10:18]
#                    feat_len = batch_feat.shape[1]*batch_feat.shape[2]*batch_feat.shape[3]
#                    tmp = np.zeros((batch_feat.shape[0], feat_len))
#                    for i in range(tmp.shape[0]):
#                        tmp[i, :] = batch_feat[i, ...].reshape([feat_len])
#                    batch_feat = tmp
                if batch_i == 0:
                    feat = batch_feat
                else:
                    feat = np.concatenate((feat, batch_feat), axis=0)
                progress_bar.update(batch_i)
            progress_bar.finish()
            useful_col = ( np.mean(np.abs(feat), axis=0) > 0 )
            if fragment_count > 0:
                assert useful_col.all() == pre_useful_col.all()
            pre_useful_col = useful_col
            if len(useful_col) < feat.shape[1]:
                feat = feat[:, useful_col]
            print('%s | --> saving feature testing fragment %d / %d'%(self._model_name,
                                   fragment_count+1, self._data_set.get_testing_fragment_num()))
            with open(self._feature_folder + 'test_fragment_%02d.npy'%(fragment_count), 'wb') as f:
                np.save(f, feat)
            with open(self._feature_folder + 'test_names_%02d.npy'%(fragment_count), 'wb') as f:
                np.save(f, name_list)



class KerasModel(object):
    def __init__(self,
                 model_name, data_set, model_inference=[], test_batch_size=64, n_iter=50, model_arch_file='',
                 model_weight_file='', model_save_path='', prediction_save_file=''):
        self._model_name = model_name
        self._data_set = data_set
        # self._data_set._batch_size = batch_size
        self._n_iter = n_iter
        self._model_arch_file = model_arch_file
        self._model_weight_file = model_weight_file
        self._model_save_path = model_save_path
#        if model_name == 'vgg_keras':
#            img_size = data_set.get_img_size()
#            self._model = vgg_std16_model(img_rows=img_size[1], img_cols=img_size[2], color_type=3,
#                                          model_weights_file=model_weights_file, continueFile='')
        if model_inference:
            self._model = model_inference
        elif model_arch_file:
            self._model = model_from_json(open(self._model_arch_file).read())
            self._model.load_weights(self._model_weight_file)
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
        open(self._model_save_path + self._model_name + '_arch.json', 'w').write(json_string)
        log_file = open(self._model_save_path + self._model_name + '_log.csv', 'w')
        log_writer = csv.writer(log_file)
        log_writer.writerow(['iter', 'train_loss', 'train_acc', 'evaluation loss', 'evaluation accuracy'])
        if self._data_set._type == 'image':#True:#self._data_set._need_training_fragment or self._data_set._set_type == 'feature':
            '''If data fragment is needed.'''
            minValidLoss = np.inf
            maxValidAcc = 0.
            
            for it in range(self._n_iter):
                '''Every iter, we read fragments one by one and train model with them setting nb_epoch=1.'''
                fragment_count = 0
                train_loss = []
                train_acc = []
                while self._data_set.have_next_training_fragment_reset():
                    fragment_count += 1
                    print('%s | iter %03d / %03d --> training fragment %d / %d'%(self._model_name,
                                           it+1, self._n_iter, fragment_count, self._data_set.get_training_fragment_num()))
                    x_train, y_train = self._data_set.next_training_fragment(validation_flag=False)
                    history = self._model.fit(x_train, y_train, batch_size=self._data_set._batch_size, nb_epoch=1)
                    train_loss.append( history.history['loss'][0] )
                    train_acc.append( history.history['acc'][0] )
                train_loss = np.mean(train_loss)
                train_acc = np.mean(train_acc)
                ''' After all training fragments are trained, we read validation fragments and evaluate the model on them. '''
                evaLoss = []
                evaAcc = []
                fragment_count = 0
                while self._data_set.have_next_validation_fragment_reset():
                    fragment_count += 1
                    print('%s | iter %03d / %03d --> validation fragment %d / %d'%(self._model_name,
                                           it+1, self._n_iter, fragment_count, self._data_set.get_validation_fragment_num()))
                    x_valid, y_valid = self._data_set.next_training_fragment(validation_flag=True)
                    loss, acc = self._model.evaluate(x_valid, y_valid, batch_size=self._data_set._batch_size)
                    evaLoss.append( loss )
                    evaAcc.append( acc )
                ''' Compute mean evaluation loss and accuracy and output them.'''
                evaLoss = np.mean(evaLoss)
                evaAcc = np.mean(evaAcc)
                print('%s | iter %03d / %03d --> validation loss: %f, acc: %f'%(self._model_name, it+1, self._n_iter, evaLoss, evaAcc))
                ''' Save the best model. '''
                if evaLoss < minValidLoss and save_best:
                    if os.path.exists(self._model_save_path + self._model_name + '_weights_best_vLoss%.5f_vAcc%.3f.h5'%(minValidLoss,maxValidAcc)):
                        os.remove(self._model_save_path + self._model_name + '_weights_best_vLoss%.5f_vAcc%.3f.h5'%(minValidLoss,maxValidAcc))
                    minValidLoss = evaLoss
                    maxValidAcc = evaAcc
                    self._best_weight_file = self._model_save_path + self._model_name + '_weights_best_vLoss%.5f_vAcc%.3f.h5'%(minValidLoss,maxValidAcc)
                    self._model.save_weights(self._best_weight_file, overwrite=True)

                log_writer.writerow([it, train_loss, train_acc, evaLoss, evaAcc])
            
            self._model.save_weights(self._model_save_path + self._model_name + '_weights_final.h5', overwrite=True)
            
        elif self._data_set._type == 'feature':
            if self._data_set._need_train_feature_fragment:
                '''If data fragment is needed.'''
                minValidLoss = np.inf
                maxValidAcc = 0.
                for it in range(self._n_iter):
                    '''Every iter, we read fragments one by one and train model with them setting nb_epoch=1.'''
                    fragment_count = 0
                    train_loss = []
                    train_acc = []
                    while self._data_set.have_next_training_fragment_reset():
                        fragment_count += 1
                        print('%s | iter %03d / %03d --> training fragment %d / %d'%(self._model_name,
                                               it+1, self._n_iter, fragment_count, self._data_set._train_fragment_num))
                        x_train, y_train = self._data_set.next_training_fragment()
                        history = self._model.fit(x_train, y_train, batch_size=self._data_set._batch_size, nb_epoch=1)
                        train_loss.append( history.history['loss'][0] )
                        train_acc.append( history.history['acc'][0] )
                    train_loss = np.mean(train_loss)
                    train_acc = np.mean(train_acc)
                    ''' After all training fragments are trained, we read validation fragments and evaluate the model on them. '''
                    if self._data_set._validation_split > 0.0:
                        print('%s | iter %03d / %03d --> validating'%(self._model_name, it+1, self._n_iter))
                        x_valid, y_valid = self._data_set.get_train_data(validation_flag=True)
                        evaLoss, evaAcc = self._model.evaluate(x_valid, y_valid, batch_size=self._data_set._batch_size)
                        print('%s | iter %03d / %03d --> training loss: %f, acc: %f'%(self._model_name, it+1, self._n_iter, train_loss, train_acc))
                        print('%s | iter %03d / %03d --> validation loss: %f, acc: %f'%(self._model_name, it+1, self._n_iter, evaLoss, evaAcc))
                        ''' Save the best model. '''
                        if evaLoss < minValidLoss and save_best:
                            if os.path.exists(self._model_save_path + self._model_name + '_weights_best_vLoss%.5f_vAcc%.3f.h5'%(minValidLoss,maxValidAcc)):
                                os.remove(self._model_save_path + self._model_name + '_weights_best_vLoss%.5f_vAcc%.3f.h5'%(minValidLoss,maxValidAcc))
                            minValidLoss = evaLoss
                            maxValidAcc = evaAcc
                            self._best_weight_file = self._model_save_path + self._model_name + '_weights_best_vLoss%.5f_vAcc%.3f.h5'%(minValidLoss,maxValidAcc)
                            self._model.save_weights(self._best_weight_file, overwrite=True)
        
                        log_writer.writerow([it, train_loss, train_acc, evaLoss, evaAcc])
                    else:
                        print('%s | iter %03d / %03d --> training loss: %f, acc: %f'%(self._model_name, it+1, self._n_iter, train_loss, train_acc))
                        print('No Validation')
                        self._best_weight_file = self._model_save_path + self._model_name + '_weights_final.h5'
                
                self._model.save_weights(self._model_save_path + self._model_name + '_weights_final.h5', overwrite=True)
            else:
                ''' If no need to fragment, load all training images and train the model directly.'''
                print('start training')
                CallBacks = []
                if save_best:
                    checkpoint = ModelCheckpoint(self._model_save_path + self._model_name + '_keras_checkpoint_weights_best.h5',
                                                monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
                    CallBacks.append(checkpoint)
    
                x_train, y_train = self._data_set.get_train_data()
                x_valid, y_valid = self._data_set.get_train_data(validation_flag=True)
                self._history = self._model.fit(x_train, y_train, batch_size=self._data_set._batch_size, nb_epoch=self._n_iter,
                                                validation_data=(x_valid, y_valid), callbacks=CallBacks)
                self._best_weight_file = self._model_save_path + self._model_name + '_keras_checkpoint_weights_best.h5'
        log_file.close()
        
    def predict(self):
        fragment_count = 0
        while self._data_set.have_next_testing_fragment_reset():
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





if __name__ == '__main__':
    caffe_root = '/home/liuzheng/toolbox/caffe-master/'
    converter = Caffe2KerasConverter(caffe_root =               caffe_root,
                                     caffe_prototxt_file =      caffe_root + 'models/VGG_ILSVRC_16_layers_deploy.prototxt',
                                     caffe_model_file =         caffe_root + 'models/VGG_ILSVRC_16_layers.caffemodel',
                                     keras_arch_save_file =     caffe_root + 'models/vgg_16_converted.json',
                                     keras_weight_save_file =   caffe_root + 'models/vgg_16_weight_converted.h5')
    converter.convert()

























