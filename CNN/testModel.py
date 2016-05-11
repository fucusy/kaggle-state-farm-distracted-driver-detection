import testCNN_KERAS as cnn_keras

isServer = False
if not isServer:
    pcProjectpath = '/home/liuzheng/competition/kaggle/distractedDrivers/'
    mxnetRoot = '/home/liuzheng/toolbox/mxnet/'

datapath = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/testAugmentation/'


prediction = cnn_keras.testModel(path=pcProjectpath, datapath=datapath, classNum=10)
