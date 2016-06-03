# -*- coding: utf-8 -*-
"""
Created on Tue May 31 20:43:33 2016

@author: ZhengLiu
"""

#import sys
#sys.path.append('../../')
import sys
sys.path.append('../')

import config
import numpy as np
import skimage
import skimage.io as skio
import skimage.transform as sktr
import os
import random
from keras.utils import np_utils

''' Make images, which maybe in shape of (n, h, w, ch) for multiple color images,
                                         (h, w, ch) for single color images,
                                         (n, h, w) for multiple gray images or
                                         (h, w) for single gray images,
    to data form that capible to keras models, which in the shape of (n, ch, h, w).
'''
def images_swap_axes(images, color_type=3):
    # images = [n, h, w, ch] or [h, w, ch] or [n, h, w] or [h, w]
    if color_type == 3:
        
        if len(images.shape) == 3:
            swaped_images = np.zeros((1, images.shape[2], images.shape[0], images.shape[1]))
            images = images.swapaxes(-2, -1)
            images = images.swapaxes(-3, -2)
            swaped_images[0, ...] = images
        else:
            swaped_images = np.zeros((images.shape[0], images.shape[3], images.shape[1], images.shape[2]))
            images = images.swapaxes(-2, -1)
            images = images.swapaxes(-3, -2)
            swaped_images = images

    elif color_type == 1:
        if len(images.shape) == 3:
            swaped_images = np.zeros((images.shape[0], 1, images.shape[1], images.shape[2]))
            for i in range(images.shape[0]):
                swaped_images[i, 0, ...] = images[i, ...]
        else:
            swaped_images = np.zeros((1, 1, images.shape[1], images.shape[2]))
            swaped_images[0, 0, ...] = images
    return swaped_images

'''
    Computing mean images. Using all training and testing images.
'''
def compute_mean_image(training_data_path, testing_data_path, save_flag=True, save_file=''):
    print('computing mean images')
    folder = os.listdir(training_data_path)
    trainNum = len(folder)
    init_flag = 1
    for f in folder:
        img = skimage.img_as_float( skio.imread(training_data_path+f) )
        if init_flag:
            mean_image = img
        else:
            mean_image += img
    
    folder = os.listdir(testing_data_path)
    testNum = len(folder)
    for f in folder:
        img = skimage.img_as_float( skio.imread(training_data_path+f) )
        mean_image += img
    
    mean_image /= (trainNum + testNum)
    
    
    if len(mean_image.shape) == 2:
        '''if gray, (h, w) to (1, h, w)'''
        tmp = np.zeros((1, mean_image.shape[0], mean_image.shape[1]))
        tmp[0, ...] = mean_image
        mean_image = tmp
    else:
        '''if color, swap (h, w, ch) to (ch, h, w)'''
        mean_image = mean_image.swapaxes(1,2)
        mean_image = mean_image.swapaxes(0,1)
    if save_flag:
        with open(save_file, 'wb') as f:
            np.save(f, mean_image)
    return mean_image

'''
    Save all resized images to training and testing folders.
    training images with all labels are saved into one folder, the label is the first character of the file names.
'''



def resize_image(original_training_data_path=config.Project.original_training_folder
                 , original_testing_data_path=config.Project.original_testing_folder
                 , training_save_path=config.Project.train_img_folder_path
                 , testing_save_path=config.Project.test_img_folder_path
                 , img_size=config.Data.img_size):
    if not os.path.exists(training_save_path):
        os.makedirs(training_save_path)
    if not os.path.exists(testing_save_path):
        os.makedirs(testing_save_path)

    if img_size[0] == 1:
        as_grey = True
    else:
        as_grey = False

    class_list = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    for c in class_list:
        folder = os.listdir(original_training_data_path + c)
        for f in folder:
            img = skio.imread( original_training_data_path + c + f, as_grey=as_grey )
            img = sktr.resize(img, output_shape=[img_size[1], img_size[2]])
            skio.imsave(training_save_path + c[1] + '_' + f)

    folder = os.listdir(original_testing_data_path)
    for f in folder:
        img = skio.imread( original_testing_data_path + f, as_grey=as_grey )
        img = sktr.resize(img, output_shape=[img_size[1], img_size[2]])
        skio.imsave(training_save_path + f)



class DataSet(object):
    
    def __init__(self,
               training_data_path, testing_data_path, mean_image_file_name, fragment_size=2048, img_size=[3, 224, 224],
               validation_split=0.0, batch_size=32, class_num=10):
        """
        
        """
        self._training_data_path = training_data_path
        self._testing_data_path = testing_data_path
        self._fragment_size = fragment_size
        if fragment_size < len(os.listdir(training_data_path)):
            self._need_training_fragment = True
        self._mean_image = np.load(mean_image_file_name)
        self._img_size = img_size
        self._validation_split = validation_split
        self._batch_size = batch_size
        print('DataSet |-->  start preparing data fragments')
        self._training_fragments_list, self._validation_fragments_list, self._testing_fragments_list = self.prepare_fragments()
        self._training_fragment_index = -1
        self._validation_fragment_index = -1
        self._testing_fragment_index = -1
        self._class_num = class_num
        self._need_training_fragment = False
        
        
    def prepare_fragments(self):
        training_data_path = self._training_data_path
        testing_data_path = self._testing_data_path
        imgSize = self._img_size
        validSplit = self._validation_split
        fragSize = self._fragment_size
        batchSize = self._batch_size
        
        ch, ih, iw = imgSize
        
        folder = os.listdir(training_data_path)
        np.random.shuffle(folder)
        trainNum = np.int( len(folder) * (1-validSplit) )
        
        trainingFile = folder[:trainNum]
        validationFile = folder[trainNum:]
        
        print('start preparing fragment')
        trainingList = []
        fragList = []
        fragCount = 0
        i = 0
        for tf in trainingFile:
            if not tf[-4:] == '.jpg':
                continue
            
            fragList.append(tf)
            fragCount += 1
            if fragCount == fragSize:
                i += 1
                print('training fragment %d finished'%(i))
                trainingList.append(fragList)
                fragList = []
                fragCount = 0
    #            break
        if len(fragList) < batchSize:
            trainingList[-1] += fragList
        else:
            trainingList.append(fragList)
        print('training fragment %d finished'%(i+1))
        
        validationList = []
        if len(validationFile) > 0:
            fragList = []
            fragCount = 0
            i = 0
            for vf in validationFile:
                if not tf[-4:] == '.jpg':
                    continue
                fragList.append(vf)
                fragCount += 1
                if fragCount == fragSize:
                    i += 1
                    print('validation fragment %d finished'%(i))
                    validationList.append(fragList)
                    fragList = []
                    fragCount = 0
        #            break
            if len(fragList) < batchSize:
                validationList[-1] += fragList
            else:
                validationList.append(fragList)
            print('validation fragment %d finished'%(i+1))
        
        '''testing fragments'''
        folder = os.listdir(testing_data_path)
        folder.sort()
        
        print('start preparing testing fragment')
        testingList = []
        fragList = []
        fragCount = 0
        i = 0
        for f in folder:
            if not f[-4:] == '.jpg':
                continue
            
            fragList.append(f)
            fragCount += 1
            if fragCount == fragSize:
                i += 1
                print('testing fragment %d finished'%(i))
                testingList.append(fragList)
                fragList = []
                fragCount = 0
    #            break
        if len(fragList) < batchSize:
            testingList[-1] += fragList
        else:
            testingList.append(fragList)
        return trainingList, validationList, testingList
    
    ''' judge if there is next fragment, if not, reset the fragment index. '''
    def have_next_training_fragment_reset(self):
        flag = True
        if self._training_fragment_index == len(self._training_fragments_list) - 1:
            flag = False
            self._training_fragment_index = -1
        return flag
    
    def have_next_testing_fragment_reset(self):
        flag = True
        if self._testing_fragment_index == len(self._testing_fragments_list) - 1:
            flag = False
            self._testing_fragment_index = -1
        return flag
    
    def have_next_validation_fragment_reset(self):
        flag = True
        if self._validation_fragment_index == len(self._validation_fragments_list) - 1:
            flag = False
            self._validation_fragment_index = -1
        return flag
    
    ''' read next fragments '''
    def next_training_fragment(self, validation_flag=False):
        if validation_flag:
            self._validation_fragment_index += 1
            frag_list = self._validation_fragments_list[self._validation_fragment_index]
        else:
            self._training_fragment_index += 1
            frag_list = self._training_fragments_list[self._training_fragment_index]
        data_path = self._training_data_path
        img_size = self._img_size
        mean_image = self._mean_image
        class_num = self._class_num
        
        
        fragment_data = []
        fragment_label = []
        
        ch, ih, iw = img_size
        fragLen = len(frag_list)
        if ch == 1:
            fragment_data = np.zeros((fragLen, 1, ih, iw))
            fragment_label = np.zeros((fragLen), dtype=int)
            idx = -1
            if validation_flag:
                print('reading validation data fragment %d'%(self._validation_fragment_index))
            else:
                print('reading training data fragment %d'%(self._training_fragment_index))
            for f in frag_list:
                idx += 1
                # print(f)
                label = np.int(f[0])
                img = skimage.img_as_float(skio.imread(data_path+f) )
    #            img -= meanImage
                fragment_data[idx, 0, ...] = img
                fragment_label[idx] = label
        elif ch == 3:
            fragment_data = np.zeros((fragLen, 3, ih, iw))
            fragment_label = np.zeros((fragLen), dtype=int)
            idx = -1
            if validation_flag:
                print('reading validation data fragment %d'%(self._validation_fragment_index))
            else:
                print('reading training data fragment %d'%(self._training_fragment_index))
            for f in frag_list:
                idx += 1
                label = np.int(f[0])
                img = skimage.img_as_float(skio.imread(data_path+f) )
                img = img.swapaxes(1, 2)
                img = img.swapaxes(0, 1)
    #            img -= meanImage
                fragment_data[idx, ...] = img
                fragment_label[idx] = label
        fragment_data -= np.tile(mean_image, [fragLen, 1, 1, 1])
        fragment_label = np_utils.to_categorical(fragment_label, class_num)
        return fragment_data, fragment_label
    
    def next_testing_fragment(self):
        self._testing_fragment_index += 1
        fragList = self._testing_fragments_list[self._testing_fragment_index]
        datapath = self._testing_data_path
        imgSize = self._img_size
        meanImage = self._mean_image

        ch, ih, iw = imgSize
        fragLen = len(fragList)
        if ch == 1:
            X = np.zeros((fragLen, 1, ih, iw))
            idx = -1
            print('reading data')
            for f in fragList:
                idx += 1
                # print(f)
                img = skimage.img_as_float(skio.imread(datapath+f) )
    #            img -= meanImage
                X[idx, 0, ...] = img
        elif ch == 3:
            X = np.zeros((fragLen, 3, ih, iw))
            idx = -1
            print('reading data')
            for f in fragList:
                idx += 1
                img = skimage.img_as_float(skio.imread(datapath+f) )
                img = img.swapaxes(1, 2)
                img = img.swapaxes(0, 1)
    #            img -= meanImage
                X[idx, ...] = img
        X -= np.tile(meanImage, [fragLen, 1, 1, 1])
        return X, fragList
    
    def load_training_data(self):
        train_data_path = self._training_data_path
        img_size = self._img_size
        mean_image = self._mean_image
        
        ch, h, w = img_size
        folder = os.listdir(train_data_path)
        random.shuffle(folder)
        
        x_train = np.zeros((len(folder), ch, h, w))
        y_train = np.zeros((len(folder)), dtype=int)
        i = -1
        for f in folder:
            img = skimage.img_as_float( skio.imread(train_data_path + f) )
            i += 1
            if ch == 1:
                img -= mean_image
                x_train[i, 0, ...] = img
            else:
                img = mean_image.swapaxes(1,2)
                img = mean_image.swapaxes(0,1)
                img -= mean_image
                x_train[i, ...] = img
            y_train[i] = np.int(f[0])
        classNum = np.max(y_train) + 1
        y_train = np_utils.to_categorical(y_train, classNum)
                
        return x_train, y_train
    
    def get_img_size(self):
        return self._img_size
    
    def need_training_fragment(self):
        return self._need_training_fragment
    
    def get_training_fragment_num(self):
        return len(self._training_fragments_list)
    
    def get_testing_fragment_num(self):
        return len(self._testing_fragments_list)
    
    def get_validation_fragment_num(self):
        return len(self._validation_fragments_list)

if __name__ == '__main__':
    isServer = False
    if not isServer:
        pcProjectpath = '/home/liuzheng/competition/kaggle/distractedDrivers/'
    #    mxnetRoot = '/home/liuzheng/toolbox/mxnet/'
    else:
        pcProjectpath = '/home/zhengliu/kaggle_drivers/'
    cropMode = 'entire'
    colorMode = 'rgb'
    reSize = [224, 224]
    if colorMode == 'rgb':
        imgSize = [3, reSize[0], reSize[1]]
    elif colorMode == 'gray':
        imgSize = [1, reSize[0], reSize[1]]
    
    modelName = 'vgg_self'
    continueFile = ''
    
    datapath = pcProjectpath + 'imgs/trainAugmentation_'+colorMode + '_' + cropMode+'_'+'%03d'%(reSize[0])+'/'
    testpath = pcProjectpath + 'imgs/testAugmentation_'+colorMode + '_' + cropMode+'_'+'%03d'%(reSize[0])+'/'
    savePrefix=pcProjectpath# + colorMode + '_' + cropMode+'_'+'%03d'%(reSize[0])+'/'
    if not os.path.exists(savePrefix):
        os.mkdir(savePrefix)
    meanImagePath = savePrefix + 'imgs/meanImage.npy'
    
    data_set = DataSet(training_data_path=datapath, testing_data_path=testpath, mean_image_file_name=meanImagePath,
                       fragment_size=2048, img_size=[3, 224, 224],
                       validation_split=0.2, batch_size=32, class_num=10)






























