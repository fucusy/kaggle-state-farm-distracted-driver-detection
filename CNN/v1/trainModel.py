import DataReader as dr

import trainCNN_MXNET as cnn_mxnet
import trainCNN_KERAS as cnn_keras

isServer = False
if not isServer:
    pcProjectpath = '/home/liuzheng/competition/kaggle/distractedDrivers/'
    mxnetRoot = '/home/liuzheng/toolbox/mxnet/'

trainListName = 'trainDatalist_distractedDrivers.lst'
#testListName = 'testDatalist_distractedDrivers.lst'
trainRecName = 'trainRecord_distractedDrivers.rec'
#testRecName = 'testRecord_distractedDrivers.rec'

mxnetRoot = 'path/to/mxnet/'
lstPath_name = pcProjectpath+trainListName
datapath = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/trainAugmentation/'
#recPath = 'path/to/database/rec/file/'
#recName = 'binName.rec'

data_name = 'distractedDrivers_64'

data_shape = (3,64,64)
batch_size = 50
rand_crop = False
rand_mirror = False
shuffle = False
preprocess_threads = 4
prefetch_buffer = 1

classNum = 10
num_epoch = 100
batch_size = 32


model, history = cnn_keras.trainModel(datapath, isGenerator=True, nb_epoch=num_epoch, batch_size=batch_size, savepath=pcProjectpath)
'''=============================================================================================================================='''
#traindata = dr.reidDataIterator(data_name, data_shape, batch_size, rand_crop, rand_mirror,\
#                        shuffle, preprocess_threads, prefetch_buffer, mxnetRoot, lstPath_name,\
#                        datapath, recPath=pcProjectpath, recName=trainRecName)
#
#symbol = cnn_mxnet.inference(inputName=data_name, classNum=classNum)
#
#model = cnn_mxnet.trainModel(data=traindata, symbol=symbol, num_epoch=num_epoch, batch_size=batch_size, learning_rate=0.0001, momentum=0.9,\
#                    wd=0.00001, recordSavepath='')






















