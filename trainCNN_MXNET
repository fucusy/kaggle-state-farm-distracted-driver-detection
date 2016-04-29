# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:31:39 2016

@author: liuzheng
"""

import mxnet as mx
import logging
import DataReader as dr

import skimage
import skimage.io as skio
import numpy as np
import os
import scipy.io as sio

    
def inference(inputName, classNum):
    '''---------------cnn model------------------------------'''
    input_data_small = mx.symbol.Variable(name=inputName)
    '''---------------convolution 1------------------------------'''
    conv1_small = mx.symbol.Convolution(
        data=input_data_small, kernel=(3, 3), stride=(2, 2), num_filter=32, name="conv1")
    relu1_small = mx.symbol.Activation(data=conv1_small, act_type="relu", name="relu1")
    pool1_small = mx.symbol.Pooling(
        data=relu1_small, pool_type="max", kernel=(3, 3), stride=(2,2), name="pool1")
    lrn1_small = mx.symbol.LRN(data=pool1_small, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    '''---------------convolution 2------------------------------'''
    conv2_small = mx.symbol.Convolution(
        data=lrn1_small, kernel=(3, 3), pad=(2, 2), num_filter=64, name="conv2")
    relu2_small = mx.symbol.Activation(data=conv2_small, act_type="relu", name="relu2")
    pool2_small = mx.symbol.Pooling(data=relu2_small, kernel=(3, 3), stride=(1, 1), pool_type="max", name="pool2")
    lrn2_small = mx.symbol.LRN(data=pool2_small, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    '''---------------convolution 3------------------------------'''
    conv3_small = mx.symbol.Convolution(
        data=lrn2_small, kernel=(3, 3), pad=(2, 2), num_filter=256, name="conv3")
    relu3_small = mx.symbol.Activation(data=conv3_small, act_type="relu", name="relu3")
    '''---------------convolution 4------------------------------'''
    conv4_small = mx.symbol.Convolution(
        data=relu3_small, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv4")
    relu4_small = mx.symbol.Activation(data=conv4_small, act_type="relu", name="relu4")
    '''---------------convolution 5------------------------------'''
    conv5_small = mx.symbol.Convolution(
        data=relu4_small, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv5")
    relu5_small = mx.symbol.Activation(data=conv5_small, act_type="relu", name="relu5")
    pool3_small = mx.symbol.Pooling(data=relu5_small, kernel=(2, 2), stride=(1, 1), pool_type="max", name="pool3")
    lrn3_small = mx.symbol.LRN(data=pool3_small, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    '''---------------convolution 6------------------------------'''
    conv6_small = mx.symbol.Convolution(
        data=lrn3_small, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv6")
    relu6_small = mx.symbol.Activation(data=conv6_small, act_type="relu", name="relu6")
    #pool4_small = mx.symbol.Pooling(data=relu6_small, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool4")
    #lrn4_small = mx.symbol.LRN(data=pool4_small, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    #'''---------------convolution 7------------------------------'''
    #conv7_small = mx.symbol.Convolution(
    #    data=pool4_small, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv7")
    #relu7_small = mx.symbol.Activation(data=conv7_small, act_type="relu", name="relu7")
    #pool5_small = mx.symbol.Pooling(data=relu7_small, kernel=(2, 2), stride=(2, 2), pool_type="max", name="pool5")
    ##lrn5_small = mx.symbol.LRN(data=pool5_small, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    #'''---------------convolution 8------------------------------'''
    #conv8_small = mx.symbol.Convolution(
    #    data=pool5_small, kernel=(3, 3), pad=(1, 1), num_filter=32, name="conv8")
    #relu8_small = mx.symbol.Activation(data=conv8_small, act_type="relu", name="relu8")
    '''---------------global pooling------------------------------'''
    global_pool = mx.symbol.Pooling(data=relu6_small, kernel=(2, 2), stride=(2, 2), pool_type="max", name="global_pool")
    '''---------------flatten------------------------------'''
    flatten_small = mx.symbol.Flatten(data=global_pool, name="flatten")
    '''---------------fully connection 1------------------------------'''
    fc1_small = mx.symbol.FullyConnected(data=flatten_small, num_hidden=1024, name="fc1")
    relu6 = mx.symbol.Activation(data=fc1_small, act_type="relu", name="relu6")
    dropout1 = mx.symbol.Dropout(data=relu6, p=0.5)
    '''---------------fully connection 2------------------------------'''
    fc2 = mx.symbol.FullyConnected(data=dropout1, num_hidden=1024, name="fc2")
    relu7 = mx.symbol.Activation(data=fc2, act_type="relu", name="relu7")
    dropout2 = mx.symbol.Dropout(data=relu7, p=0.5)
    '''---------------fully connection 3 and softmax------------------------------'''
    fc3 = mx.symbol.FullyConnected(data=dropout2, num_hidden=classNum, name="fc3")
    softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
    
    return softmax

'''
layerOutputSahpe = layer.infer_shape(data_small=(batchNum, channel, H, W))
'''

'''---------------read training data------------------------------'''
#train = dr.reidDataIterator(data_name='data_small', path_imgrec=datapath+'trainrec.bin',\
#                                data_shape=(3,64,64), batch_size=50, rand_crop=False, rand_mirror=True,\
#                                shuffle=False,  preprocess_threads=4, prefetch_buffer=1)
#, mean_img=datapath+'meanrec.bin'
def trainModel(data, symbol, num_epoch, batch_size, learning_rate=0.001, momentum=0.9,\
                    wd=0.00001, recordSavepath=''):
    '''---------------train model------------------------------'''
    num_gpus = 1
    gpus = [mx.gpu(i) for i in range(num_gpus)]
    
    model = mx.model.FeedForward(
        ctx           = gpus,
        symbol        = symbol,
        num_epoch     = num_epoch,
        learning_rate = learning_rate,
        momentum      = momentum,
        wd            = wd)
    logging.basicConfig(level = logging.DEBUG)
    model.fit(X = data, batch_end_callback = mx.callback.Speedometer(batch_size=batch_size))
    
    '''---------------save model------------------------------'''
    model.save(recordSavepath+'', num_epoch)
