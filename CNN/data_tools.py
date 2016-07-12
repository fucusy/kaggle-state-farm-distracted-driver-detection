# -*- coding: utf-8 -*-
"""
Created on Tue May 31 20:43:33 2016

@author: ZhengLiu
"""

#import sys
#sys.path.append('../../')
import sys
sys.path.append('../')

import numpy as np
import skimage
import skimage.io as skio
import skimage.transform as sktr
import os
import random
import csv
from keras.utils import np_utils
import glob

#import config

''' Make images, which maybe in shape of (n, h, w, ch) for multiple color images,
                                         (h, w, ch) for single color images,
                                         (n, h, w) for multiple gray images or
                                         (h, w) for single gray images,
    to data form that capible to keras models, which in the shape of (n, ch, h, w).
'''
def replace_zeros_with_point_ones(file_name, save_file):
    file_obj = file(file_name, 'rb')
    reader = csv.reader(file_obj)
    
    save_obj = file(save_file, 'wb')
    writer = csv.writer(save_obj)
    
    i = -1
    for line in reader:
        i += 1
        if i > 0:
            print('line %d'%(i))
            re = np.array(line[1:], dtype=float)
            re[re < 0.01] = 0.05
            save_line = [line[0]] + list(re)
            writer.writerow(save_line)
        else:
            writer.writerow(line)
    file_obj.close()
    save_obj.close()
    
    
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
    init_flag = True
    for f in folder:
        img = skimage.img_as_float( skio.imread(training_data_path+f) )
        if init_flag:
            mean_image = img
            init_flag = False
        else:
            mean_image += img
    
    folder = os.listdir(testing_data_path)
    testNum = len(folder)
    for f in folder:
        img = skimage.img_as_float( skio.imread(testing_data_path+f) )
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
def resize_image(original_training_data_path, original_testing_data_path, training_save_path, testing_save_path,
                 img_size=(3, 224, 224), postfix=[], mean_save_flag=True, mean_save_file=''):
    for i in range(len(training_save_path)):
        if training_save_path[i] != testing_save_path[i]:
            break
    sub_folder = training_save_path[:i-1]
    if not os.path.exists(sub_folder):
        os.mkdir(sub_folder)
    if not os.path.exists(training_save_path):
        os.mkdir(training_save_path)
    if not os.path.exists(testing_save_path):
        os.mkdir(testing_save_path)
    
    if img_size[0] == 1:
        as_grey = True
    else:
        as_grey = False
    
    class_list = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
#    train_num = 0
    for c in class_list:
        folder = os.listdir(original_training_data_path + c)
        for f in folder:
            if f[-4:] != '.jpg' or f[:2] == '._':
                continue
            print('processing train %s %s'%(c, f))
            img = skio.imread( original_training_data_path + c + '/' + f, as_grey=as_grey )
            img = sktr.resize(img, output_shape=[img_size[1], img_size[2]])
            skio.imsave(training_save_path + c[1] + '_' + f[:-4] + postfix[0] + '.jpg', img)
    
    folder = os.listdir(original_testing_data_path)
    for f in folder:
        if f[-4:] != '.jpg' or f[:2] == '._':
            continue
        print('processting test %s'%(f))
        img = skio.imread( original_testing_data_path + f, as_grey=as_grey )
        img = sktr.resize(img, output_shape=[img_size[1], img_size[2]])
        skio.imsave(testing_save_path + f[:-4] + postfix[0] + '.jpg', img)
    
            
def crop_resize_image(original_training_data_path, original_testing_data_path, training_save_path, testing_save_path,
                      img_size=(3, 224, 224), crop_num=5, postfix='', mean_save_flag=True, mean_save_file=''):
    for i in range(len(training_save_path)):
        if training_save_path[i] != testing_save_path[i]:
            break
    sub_folder = training_save_path[:i-1]
    if not os.path.exists(sub_folder):
        os.mkdir(sub_folder)
    if not os.path.exists(training_save_path):
        os.mkdir(training_save_path)
    if not os.path.exists(testing_save_path):
        os.mkdir(testing_save_path)
    
    if img_size[0] == 1:
        as_grey = True
    else:
        as_grey = False
    
    if crop_num > 1:
        crop_size = [np.int(img_size[1]*1.3), np.int(img_size[2]*1.3)]
        y_x = [[0, 0], [0, crop_size[1]-img_size[2]], [(crop_size[0]-img_size[1])/2, (crop_size[1]-img_size[2])/2],
               [crop_size[0]-img_size[1], 0],
               [crop_size[0]-img_size[1], crop_size[1]-img_size[2]]]
    else:
        crop_size = [img_size[1], img_size[2]]
    
    class_list = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    init_flag = True
    train_num = 0
    for c in class_list:
        folder = os.listdir(original_training_data_path + c)
        for f in folder:
            if f[-4:] != '.jpg' or f[:2] == '._':
                continue
            print('processing train %s %s'%(c, f))
            img = skio.imread( original_training_data_path + c + '/' + f, as_grey=as_grey )
            img = sktr.resize(img, output_shape=crop_size)
            if crop_num > 1:
                for i in range(len(y_x)):
                    y,x = y_x[i]
                    crop_img = img[y:y+img_size[1], x:x+img_size[2], ...]
                    train_num += 1
                    if init_flag:
                        mean_image = crop_img
                        init_flag = False
                    else:
                        mean_image += crop_img
                    skio.imsave(training_save_path + c[1] + '_' + f[:-4] + postfix[i] + '.jpg', crop_img)
            else:
                train_num += 1
                if init_flag:
                    mean_image = img
                    init_flag = False
                else:
                    mean_image += img
                skio.imsave(training_save_path + c[1] + '_' + f[:-4] + postfix[0] + '.jpg', img)
    
    test_num = 0
    folder = os.listdir(original_testing_data_path)
    for f in folder:
        if f[-4:] != '.jpg' or f[:2] == '._':
            continue
        print('processting test %s'%(f))
        img = skio.imread( original_testing_data_path + f, as_grey=as_grey )
        img = sktr.resize(img, output_shape=[img_size[1], img_size[2]])
        test_num += 1
        if init_flag:
            mean_image = img
            init_flag = False
        else:
            mean_image += img
        skio.imsave(testing_save_path + f[:-4] + postfix[0] + '.jpg', img)
    
    '''process mean image'''
    mean_image /= (train_num + test_num)
    
    if len(mean_image.shape) == 2:
        '''if gray, (h, w) to (1, h, w)'''
        tmp = np.zeros((1, mean_image.shape[0], mean_image.shape[1]))
        tmp[0, ...] = mean_image
        mean_image = tmp
    else:
        '''if color, swap (h, w, ch) to (ch, h, w)'''
        mean_image = mean_image.swapaxes(1,2)
        mean_image = mean_image.swapaxes(0,1)
    if mean_save_flag:
        with open(mean_save_file, 'wb') as f:
            np.save(f, mean_image)
    return mean_image


class ImageSet(object):
    
    def __init__(self,
               training_data_path, testing_data_path, mean_image_file_name, driver_list_file,
               augmentation_postfix=    '',
               fragment_size=           2048,
               img_size=                [3, 224, 224],
               padded_img_size=         [224, 224],
               validation_split=        0.0,
               batch_size=              32,
               class_num=               10,
               for_extractor=           False,
               train_test_folder=       ''):
        """
        
        """
        self._train_test_folder = train_test_folder
        if not os.path.exists(train_test_folder):
            os.mkdir(train_test_folder)
        self._training_data_path = training_data_path
        self._testing_data_path = testing_data_path
        self._fragment_size = fragment_size
        self._need_training_fragment = False
        if fragment_size < len(os.listdir(training_data_path)):
            self._need_training_fragment = True
        self._mean_image = np.load(mean_image_file_name)
        self._img_size = img_size
        if img_size[1] == padded_img_size[0] and img_size[2] == padded_img_size[1]:
            padded_img_size = []
        self._padded_img_size = padded_img_size
        if padded_img_size:
            self._pad_y = (padded_img_size[0] - img_size[1]) / 2
            self._pad_x = (padded_img_size[1] - img_size[2]) / 2
        self._validation_split = validation_split
        self._batch_size = batch_size
        print('DataSet |-->  start preparing data fragments')
        
        self._training_fragment_index = -1
        self._validation_fragment_index = -1
        self._testing_fragment_index = -1
        self._class_num = class_num
        
        self._for_extractor = for_extractor
        
        if for_extractor:
            self._driver_dict = self.get_driver_dict(driver_list_file, augmentation_postfix,
                                                    discard_samples=False, sort_or_rand='rand')
#            self._driver_num = len(self._driver_dict)
#            self._driver_index = -1
            self._training_fragments_list, self._validation_fragments_list = self.prepare_train_fragments(self._driver_dict,
                                                                                                          for_extractor)
        else:
            self._driver_dict = self.get_driver_dict(driver_list_file, augmentation_postfix,
                                                    discard_samples=True, sort_or_rand='rand')
            self._training_fragments_list, self._validation_fragments_list = self.prepare_train_fragments(self._driver_dict,
                                                                                                          for_extractor)
        self._testing_fragments_list = self.prepare_test_fragments()
        
        self._type = 'image'
    
    def analyze_driver_file(self, driver_list_file):
        driver_dict = {}
        driver_file = file(driver_list_file, 'rb')
        reader = csv.reader(driver_file)
        i = -1
        for line in reader:
            i += 1
            if i > 0:
                person = line[0]
                if not driver_dict.has_key(person):
                    driver_dict[person] = {}
                    for i in range(self._class_num):
                        driver_dict[person][i] = []
                label = np.int( line[1][1] )
                driver_dict[person][label].append(line[2])
        driver_file.close()
        
        driver_std = {}
        for driver in driver_dict.keys():
            sample_num = np.zeros((self._class_num))
            for i in range(self._class_num):
                sample_num[i] = len(driver_dict[driver][i])
                driver_std[driver] = np.std(sample_num)
        return driver_dict, driver_std
    
    def get_driver_dict(self, driver_list_file, augmentation_postfix, discard_samples=False, sort_or_rand='rand'):
        driver_dict = {}
        driver_file = file(driver_list_file, 'rb')
        reader = csv.reader(driver_file)
        i = -1
        for line in reader:
            i += 1
            if i > 0:
                person = line[0]
                if not driver_dict.has_key(person):
                    driver_dict[person] = {}
                    for i in range(self._class_num):
                        driver_dict[person][i] = []
                label = np.int( line[1][1] )
                for postfix in augmentation_postfix:
                    driver_dict[person][label].append('%01d_'%(label)+line[2][:-4]+postfix+'.jpg')
        driver_file.close()
        
        if discard_samples:
            for driver in driver_dict.keys():
                sample_num = np.zeros((self._class_num))
                for i in range(self._class_num):
                    sample_num[i] = len(driver_dict[driver][i])
                reserve_num = np.int( np.min(sample_num) )
                for i in range(self._class_num):
                    if sort_or_rand == 'rand':
                        random.shuffle( driver_dict[driver][i] )
                    elif sort_or_rand == 'sort':
                        driver_dict[driver][i].sort()
                    driver_dict[driver][i] = driver_dict[driver][i][:reserve_num]
        return driver_dict
    
    def prepare_train_fragments(self, driver_dict, for_extractor=False):
        fragSize = self._fragment_size
        batchSize = self._batch_size
        
        if for_extractor:
            trainingList = []
            validationList = []
            drivers = driver_dict.keys()
            for d in drivers:
                print('preparing training driver %s'%(d))
                frag_list = []
                for l in range(self._class_num):
                    frag_list += driver_dict[d][l]
                trainingList.append(frag_list)
        
#        if os.path.exists(self._train_test_folder+'train_fragment_list.npy'):
#            with open(self._train_test_folder+'train_fragment_list.npy', 'rb') as f:
#                trainingList = np.load(f)
#            trainingList = list(trainingList)
#            with open(self._train_test_folder+'valid_fragment_list.npy', 'rb') as f:
#                validationList = np.load(f)
#            validationList = list(validationList)
        else:
            drivers = driver_dict.keys()
            random.shuffle(drivers)
            train_num = np.int( len(drivers) * (1-self._validation_split) )
            training_drivers = drivers[:train_num]
            validation_drivers = drivers[train_num:]
            
            print('start preparing training fragments')
            training_list = []
            for d in training_drivers:
                for l in range(self._class_num):
                    training_list += driver_dict[d][l]
            random.shuffle(training_list)
            trainingList = []
            fragList = []
            fragCount = 0
            i = 0
            for tf in training_list:
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
            
            validation_list = []
            for d in validation_drivers:
                for l in range(self._class_num):
                    validation_list += driver_dict[d][l]
            validationList = []
            if len(validation_list) > 0:
                fragList = []
                fragCount = 0
                i = 0
                for vf in validation_list:
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
        with open(self._train_test_folder+'train_fragment_list.npy', 'wb') as f:
            np.save(f, np.array(trainingList))
        with open(self._train_test_folder+'valid_fragment_list.npy', 'wb') as f:
            np.save(f, np.array(validationList))
        return trainingList, validationList
    
    def prepare_test_fragments(self):
        fragSize = self._fragment_size
        batchSize = self._batch_size
        testing_data_path = self._testing_data_path
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
        return testingList
    
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
    def next_training_fragment(self, validation_flag=False, for_extractor=False):
        if for_extractor:
            self._training_fragment_index += 1
            frag_list = self._training_fragments_list[self._training_fragment_index]
        else:
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
            if self._padded_img_size:
                fragment_data = np.zeros((fragLen, 1, self._padded_img_size[0], self._padded_img_size[1]))
            else:
                fragment_data = np.zeros((fragLen, 1, ih, iw))
            fragment_label = np.zeros((fragLen), dtype=int)
            idx = -1
            if validation_flag:
                print('reading validation data fragment')
            else:
                print('reading training data fragment')
            for f in frag_list:
                idx += 1
                # print(f)
                label = np.int(f[0])
                img = skimage.img_as_float(skio.imread(data_path+f) )
                img -= mean_image
                if self._padded_img_size:
                    fragment_data[idx, 0, self._pad_y:self._pad_y+ih, self._pad_x:self._pad_x+iw] = img
                else:
                    fragment_data[idx, 0, ...] = img
                fragment_label[idx] = label
        elif ch == 3:
            if self._padded_img_size:
                fragment_data = np.zeros((fragLen, 3, self._padded_img_size[0], self._padded_img_size[1]))
            else:
                fragment_data = np.zeros((fragLen, 3, ih, iw))
            fragment_label = np.zeros((fragLen), dtype=int)
            idx = -1
            if validation_flag:
                print('reading validation data fragment')
            else:
                print('reading training data fragment')
            for f in frag_list:
                idx += 1
                label = np.int(f[0])
                img = skimage.img_as_float(skio.imread(data_path+f) )
                img = img.swapaxes(1, 2)
                img = img.swapaxes(0, 1)
                img -= mean_image
                if self._padded_img_size:
                    fragment_data[idx, :, self._pad_y:self._pad_y+ih, self._pad_x:self._pad_x+iw] = img
                else:
                    fragment_data[idx, ...] = img
                fragment_label[idx] = label
        # fragment_data -= np.tile(mean_image, [fragLen, 1, 1, 1])
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
            if self._padded_img_size:
                X = np.zeros((fragLen, 1, self._padded_img_size[0], self._padded_img_size[1]))
            else:
                X = np.zeros((fragLen, 1, ih, iw))
            idx = -1
            print('reading data')
            for f in fragList:
                idx += 1
                # print(f)
                img = skimage.img_as_float(skio.imread(datapath+f) )
                img -= meanImage
                if self._padded_img_size:
                    X[idx, 0, self._pad_y:self._pad_y+ih, self._pad_x:self._pad_x+iw] = img
                else:
                    X[idx, 0, ...] = img
        elif ch == 3:
            if self._padded_img_size:
                X = np.zeros((fragLen, 3, self._padded_img_size[0], self._padded_img_size[1]))
            else:
                X = np.zeros((fragLen, 3, ih, iw))
            idx = -1
            print('reading data')
            for f in fragList:
                idx += 1
                img = skimage.img_as_float(skio.imread(datapath+f) )
                img = img.swapaxes(1, 2)
                img = img.swapaxes(0, 1)
                img -= meanImage
                if self._padded_img_size:
                    X[idx, :, self._pad_y:self._pad_y+ih, self._pad_x:self._pad_x+iw] = img
                else:
                    X[idx, ...] = img
#        X -= np.tile(meanImage, [fragLen, 1, 1, 1])
        return X, fragList
    
#    def have_next_driver(self):
#        if self._driver_index == self._driver_num - 1:
#            self._driver_index = -1
#            return False
#        else:
#            return True
#    
#    def next_driver_data(self):
#        self._driver_index += 1
#        driver_list, label_list = self._training_fragments_list[self._driver_index]
        
    
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


class FeatureSet(object):
    def __init__(self,
                 feature_folders=               [''],
                 feature_train_test_folder=     '',
                 feature_fragment_size=         4096,
                 need_train_feature_fragment=   False,
                 batch_size=                    32,
                 class_num=                     10,
                 validation_split=              0.2):
        
        self._feature_folders = feature_folders
        self._feature_fragment_size = feature_fragment_size
        
        self._train_feature_folder = feature_train_test_folder + 'train/'
        self._test_feature_folder = feature_train_test_folder + 'test/'
        
        self._feature_fragment_size = feature_fragment_size
        self._need_train_feature_fragment = need_train_feature_fragment
        self._batch_size = batch_size
        self._class_num = class_num
        self._validation_split = validation_split
        
        self._training_fragment_index = -1
        self._testing_fragment_index = -1
        self._validation_fragment_index = -1
        
        print('feature set initializing')        
        
        self.get_feature_dim()
        
        #self._feature_size = feature_size
        self._type = 'feature'
        
        if not os.path.exists(feature_train_test_folder):
            os.mkdir(feature_train_test_folder)
        self.generate_train_valid_driver_list()
        self._test_feature_fragment_num = len(glob.glob(feature_folders[0]+'test_fragment_*'))
        if need_train_feature_fragment:
            self._fragment_driver_list = {}
            self._driver_frag_size = {}
            self._total_train_sample_num = 0
            sample_folder = feature_folders[0]
            for driver in self._train_drivers:
                with open(sample_folder+'train_driver_data_label_%02d.npy'%(driver), 'rb') as f:
                    data = np.load(f)
                sample_num = data.shape[0]
                self._total_train_sample_num += sample_num
                self._fragment_driver_list[driver] = np.arange(sample_num)
                np.random.shuffle(self._fragment_driver_list[driver])
            self._train_fragment_num = np.int(np.ceil(np.float(self._total_train_sample_num) / feature_fragment_size))
            for driver in self._train_drivers:
                self._driver_frag_size[driver] = np.int(len(self._fragment_driver_list[driver]) / self._train_fragment_num)
#            os.mkdir(self._train_feature_folder)
#            os.mkdir(self._test_feature_folder)
#            self.concatenate_shuffle_train_features()
#            self.concatenate_shuffle_test_features()
#        else:
#            self._train_feature_fragment_num = len(glob.glob(self._train_feature_folder+'train_features_*.npy'))
#            self._test_feature_fragment_num = len(glob.glob(self._test_feature_folder+'test_fragment_*.npy'))
#            self._valid_feature_fragment_num = len(glob.glob(self._train_feature_folder+'valid_features.npy'))
    
    def generate_train_valid_driver_list(self):
        sample_folder = self._feature_folders[0]
        driver_num = len(glob.glob(sample_folder+'train_driver_*'))
        driver_index = range(driver_num)
        random.shuffle(driver_index)
        train_num = np.int( driver_num * (1-self._validation_split) )
        self._train_drivers = driver_index[:train_num]
        self._valid_drivers = driver_index[train_num:]
        
    
    def get_train_data(self, validation_flag=False):
        if validation_flag:
            driver_list = self._valid_drivers
            print('loading validation data')
        else:
            driver_list = self._train_drivers
            print('loading training data')
        
        init_flag = True
        y_data = np.zeros((0, self._class_num))
        for driver in driver_list:
            driver_init = True
            label_flag = True
            print('loading feature of driver %d'%(driver))
            for folder in self._feature_folders:
                with open(folder+'train_driver_data_label_%02d.npy'%(driver), 'rb') as f:
                    data = np.load(f)
                if label_flag:
                    y_data = np.concatenate((y_data, data[:, -self._class_num:]), axis=0)
                    label_flag = False
                data = data[:, :-self._class_num]
                if driver_init: 
                    driver_data = data
                    driver_init = False
                else:
                    driver_data = np.concatenate((driver_data, data), axis=1)
            if init_flag:
                x_data = driver_data
                init_flag = False
            else:
                x_data = np.concatenate((x_data, driver_data), axis=0)
        if not validation_flag:
            print('shuffling data')
            indeces = range(x_data.shape[0])
            np.random.shuffle(indeces)
            x_data = x_data[indeces, :]
            y_data = y_data[indeces, :]
        return x_data, y_data
    
    ''' judge if there is next fragment, if not, reset the fragment index. '''
    def have_next_training_fragment_reset(self):
        flag = True
        if self._training_fragment_index == self._train_fragment_num - 1:
            flag = False
            self._training_fragment_index = -1
        return flag
    
    def have_next_testing_fragment_reset(self):
        flag = True
        if self._testing_fragment_index == self._test_feature_fragment_num - 1:
            flag = False
            self._testing_fragment_index = -1
        return flag
    
    def have_next_validation_fragment_reset(self):
        flag = True
        if self._validation_fragment_index == self._valid_feature_fragment_num - 1:
            flag = False
            self._validation_fragment_index = -1
        return flag

    def next_training_fragment(self):
        self._training_fragment_index += 1
        
        print('loading data')
        labels = np.zeros((0, self._class_num))
        frag_data = np.zeros((0, self._feature_dim))
        for driver in self._train_drivers:
            driver_frag_size = self._driver_frag_size[driver]
            if self._training_fragment_index == self._train_fragment_num - 1:
                driver_frag_index = self._fragment_driver_list[driver][self._training_fragment_index*driver_frag_size:]
            else:
                driver_frag_index = self._fragment_driver_list[driver][self._training_fragment_index*driver_frag_size:(self._training_fragment_index+1)*driver_frag_size]
            label_flag = True
            driver_init = True
            for folder in self._feature_folders:
                with open(folder+'train_driver_data_label_%02d.npy'%(driver), 'rb') as f:
                    data = np.load(f)
                data = data[driver_frag_index, :]
                if label_flag:
                    labels = np.concatenate((labels, data[:, -self._class_num:]), axis=0)
                    label_flag = False
                data = data[:, :-self._class_num]
                if driver_init: 
                    driver_data = data
                    driver_init = False
                else:
                    driver_data = np.concatenate((driver_data, data), axis=1)
            frag_data = np.concatenate((frag_data, driver_data), axis=0)
        return frag_data, labels
    
    def next_testing_fragment(self):
        self._testing_fragment_index += 1
        name_flag = True
        frag_init = True
        for folder in self._feature_folders:
            with open(folder+'test_fragment_%02d.npy'%(self._testing_fragment_index), 'rb') as f:
                data = np.load(f)
            if name_flag:
                with open(folder+'test_names_%02d.npy'%(self._testing_fragment_index), 'rb') as f:
                    names = np.load(f)
                name_flag = False
            if frag_init: 
                frag_data = data
                frag_init = False
            else:
                frag_data = np.concatenate((frag_data, data), axis=1)
        return frag_data, names
    
    def get_training_fragment_num(self):
        return self._train_feature_fragment_num
    
    def get_testing_fragment_num(self):
        return self._test_feature_fragment_num
    
    def get_validation_fragment_num(self):
        return self._valid_feature_fragment_num
    
    def get_feature_dim(self):
        self._feature_dim = 0
        for folder in self._feature_folders:
            with open(folder+'train_driver_data_label_00.npy', 'rb') as f:
                driver_data = np.load(f)
            self._feature_dim += driver_data.shape[1] - self._class_num
        return self._feature_dim


if __name__ == '__main__':
    
    replace_zeros_with_point_ones(file_name='/home/liuzheng/competition/kaggle/distractedDrivers/feature_train_test_224_112/inference_dense_prediction.csv',
                                  save_file='/home/liuzheng/competition/kaggle/distractedDrivers/feature_train_test_224_112/inference_dense_prediction01.csv')
                                  
#    isServer = False
#    if not isServer:
#        pcProjectpath = '/home/liuzheng/competition/kaggle/distractedDrivers/'
#    #    mxnetRoot = '/home/liuzheng/toolbox/mxnet/'
#    else:
#        pcProjectpath = '/home/zhengliu/kaggle_drivers/'
#    cropMode = 'entire'
#    colorMode = 'rgb'
#    reSize = [224, 224]
#    if colorMode == 'rgb':
#        imgSize = [3, reSize[0], reSize[1]]
#    elif colorMode == 'gray':
#        imgSize = [1, reSize[0], reSize[1]]
#    
#    modelName = 'vgg_self'
#    continueFile = ''
#    
#    datapath = pcProjectpath + 'imgs/trainAugmentation_'+colorMode + '_' + cropMode+'_'+'%03d'%(reSize[0])+'/'
#    testpath = pcProjectpath + 'imgs/testAugmentation_'+colorMode + '_' + cropMode+'_'+'%03d'%(reSize[0])+'/'
#    savePrefix=pcProjectpath# + colorMode + '_' + cropMode+'_'+'%03d'%(reSize[0])+'/'
#    if not os.path.exists(savePrefix):
#        os.mkdir(savePrefix)
#    meanImagePath = savePrefix + 'imgs/meanImage.npy'
#    
#    data_set = DataSet(training_data_path=datapath, testing_data_path=testpath, mean_image_file_name=meanImagePath,
#                       fragment_size=2048, img_size=[3, 224, 224],
#                       validation_split=0.2, batch_size=32, class_num=10)






























