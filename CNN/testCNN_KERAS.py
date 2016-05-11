from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import pickle
import csv
import DataReader_KERAS as dr_keras

def loadModel(path, loss='categorical_crossentropy', optimizer='adagrad'):
    model = model_from_json(open(path + 'keras_model_architecture.json').read())
    model.load_weights(path + 'keras_model_weights.h5')
    print('compiling model.....')
    model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def savePrediction_pickle(nameList, prediction, path):
    resultDict = {}
    i = -1
    for f in nameList:
        i += 1
        resultDict[f] = prediction[i, :]
    print('saving pickle result...')
    with open(path+'prediction.pickle', 'wb') as f:
        pickle.dump(f, resultDict)

def savePrediction_CSV(nameList, prediction, path):
    fileObj = open(path+'prediction.csv', 'wb')
    writer = csv.writer(fileObj)
    writer.writerow(['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
    for idx in range(len(nameList)):
        line = [nameList[idx]] + list(prediction[idx, :])
        writer.writerow(line)
    fileObj.close()
        

def testModel(path, datapath, classNum=10):
    model = loadModel(path)
    X_test, Y_test, nameList = dr_keras.readTestingImages_generator_noLabel_nameList(datapath, classNum)
    datagen = ImageDataGenerator(featurewise_center=True,\
                                samplewise_center=False,\
                                featurewise_std_normalization=True,\
                                samplewise_std_normalization=False,\
                                zca_whitening=False,\
                                rotation_range=0.,\
                                width_shift_range=0.,\
                                height_shift_range=0.,\
                                shear_range=0.,\
                                #zoom_range=0.,\
                                #channel_shift_range=0.,\
                                #fill_mode='nearest',\
                                #cval=0.,\
                                #horizontal_flip=False,\
                                #vertical_flip=False,\
                                dim_ordering='th')
    print('generator is fitting data.....')
    datagen.fit(X_test)
    print('predicting...')
    
    testIter = 0
    for X_batch, Y_batch in datagen.flow(X_test, Y_test, batch_size=Y_test.shape[0]):
        testIter += 1
        print('testIter %d'%(testIter))
        prediction = model.predict(X_batch, batch_size=64)
        if testIter > 0:
            break
    savePrediction_CSV(nameList, prediction, path='')
    resultDict = {}
    i = -1
    for f in nameList:
        i += 1
        resultDict[f] = prediction[i, :]
    return resultDict
    

