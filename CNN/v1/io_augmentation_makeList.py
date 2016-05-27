import DataAugmentation as da
import MakeDataList as mdl
#import skimage.io as skio
import os
isServer = False
if not isServer:
    pcProjectpath = '/home/liuzheng/competition/kaggle/distractedDrivers/'
    mxnetRoot = '/home/liuzheng/toolbox/mxnet/'

trainListName = 'trainDatalist_distractedDrivers.lst'
testListName = 'testDatalist_distractedDrivers.lst'
trainRecName = 'trainRecord_distractedDrivers.rec'
testRecName = 'testRecord_distractedDrivers.rec'

scriptPath = os.getcwd() + '/'

trainDatapath = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/train/'
testDatapath = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/test/'
trainFolderList = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
classNum = len(trainFolderList)

trainSavepath = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/trainAugmentation/'
testSavepath = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/testAugmentation/'

reSize = [64,64]

'''============================= Processing training data ======================================='''
for c in range(classNum):
    classDatapath = trainDatapath+trainFolderList[c]+'/'
    da.augmentTrainingImages_distractedDrivers(datapath=classDatapath, classIdx=c,\
								savepath=trainSavepath, reSize=reSize)

print('making training data list...')
mdl.makeTrainList_distractedDrivers(datapath=trainSavepath, prefix=scriptPath, listName=trainListName)

os.system(mxnetRoot+'bin/im2rec '+scriptPath+trainListName+' '+trainSavepath+' '+pcProjectpath+trainRecName)
'''=============================================================================================='''

'''============================= Processing testing data ======================================='''
da.augmentTestingImages_distractedDrivers(datapath=testDatapath, savepath=testSavepath, reSize=reSize)

print('making testing data list...')
mdl.makeTestList_distractedDrivers(datapath=testSavepath, prefix=scriptPath, listName=testListName)

os.system(mxnetRoot+'bin/im2rec '+scriptPath+testListName+' '+testSavepath+' '+pcProjectpath+testRecName)
'''=============================================================================================='''















