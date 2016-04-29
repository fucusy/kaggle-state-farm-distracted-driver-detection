import DataReader as dr

import trainCNN_MXNET as cnn_mxnet

mxnetRoot = 'path/to/mxnet/'
lstPath_name = 'path/to/data/list/'
datapath = 'path/to/image/data/'
recPath = 'path/to/database/rec/file/'
recName = 'binName.rec'

data_name = 'distractedDrivers_64'

data_shape = (3,64,64)
batch_size = 50
rand_crop = False
rand_mirror = True
shuffle = False
preprocess_threads = 4
prefetch_buffer = 1

classNum = 10
num_epoch = 30
batch_size = 32

'''=============================================================================================================================='''
traindata = dr.reidDataIterator(data_name, data_shape, batch_size, rand_crop, rand_mirror,\
                        shuffle, preprocess_threads, prefetch_buffer, mxnetRoot, lstPath_name, datapath, recPath, recName)

symbol = cnn_mxnet.inference(inputName=data_name, classNum)

model = trainModel(data=traindata, symbol=symbol, num_epoch=num_epoch, batch_size=batch_size, learning_rate=0.001, momentum=0.9,\
                    wd=0.00001, recordSavepath='')
