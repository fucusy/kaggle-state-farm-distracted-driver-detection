# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:36:01 2016

@author: liuzheng
"""

import skimage.io as skio
import skimage.transform as sktr
import skimage.color as skcolor
import numpy as np
import os
from keras.utils import np_utils
import scipy.io as sio

def mirrorImage(img):
    if len(img.shape) == 3:
        h,w,ch = img.shape
        mirrorImg = np.zeros_like(img)
        for x in range(w/2):
            mirrorImg[:,x,:] = img[:,w-1-x,:]
            mirrorImg[:,w-1-x,:] = img[:,x,:]
    else:
        h,w = img.shape
        mirrorImg = np.zeros_like(img)
        for x in range(w/2):
            mirrorImg[:,x] = img[:,w-1-x]
            mirrorImg[:,w-1-x] = img[:,x]
    return mirrorImg

def augmentImage(img, reSize, prefix):
    resizeImg = sktr.resize(img, reSize)
    skio.imsave(prefix+'resize.jpg', resizeImg)
    rotateImg = sktr.rotate(resizeImg, 180)
    skio.imsave(prefix+'rotate.jpg', rotateImg)
    mirrorImg = mirrorImage(resizeImg)
    skio.imsave(prefix+'mirror.jpg', mirrorImg)

def computeMeanImage(dataset, datapath, savepath):
    peopleList = os.listdir(datapath)
    init = False
    count = 0.
    for person in peopleList:
        personImg = os.listdir(datapath+person)
        for imgname in personImg:
            count += 1
            img = skio.imread(datapath+person+'/'+imgname)
            img = img / 255.
            if not init:
                meanimg = np.zeros_like(img)
                init = True
            meanimg += img
    meanimg /= count
    meanimg2 = meanimg
    meanimg = meanimg*255
    meanimg = np.array( meanimg*255, dtype=np.uint8)
    skio.imsave(savepath+'%s_meanimg.jpg'%(dataset), meanimg)
    return meanimg, meanimg2

def augmentTrainingImages(datapath, classIdx, savepath, reSize=[224,224], needGray=False, needMirror=False, needRotate=False):
    folder = os.listdir(datapath)
    totalImg = len(folder)
    count = 0
    for imgName in folder:
        if not imgName[-3:] == 'jpg':
            continue
        if imgName[0] == '.':
            imgName = imgName[2:]
        count += 1
        print('augmenting training class %d, %d / %d'%(classIdx, count, totalImg))
        prefix = savepath + '%01d_%s_'%(classIdx, imgName[:-4])
        if needGray:
            img = skio.imread(datapath + imgName, as_grey=True)
        else:
            img = skio.imread(datapath + imgName, as_grey=False)
        if reSize:
            resizeImg = sktr.resize(img, reSize)
            skio.imsave(prefix+'resize.jpg', resizeImg)
        if needMirror:
            mirrorImg = mirrorImage(resizeImg)
            skio.imsave(prefix+'mirror.jpg', mirrorImg)
        if needRotate:
            rotateImg = sktr.rotate(resizeImg, 180)
            skio.imsave(prefix+'rotate.jpg', rotateImg)
        # augmentImage(img, reSize, prefix)

def augmentTestingImages(datapath, savepath, reSize=[224,224], needGray=False, needMirror=False, needRotate=False):
    folder = os.listdir(datapath)
    totalImg = len(folder)
    count = 0
    for imgName in folder:
        if not imgName[-3:] == 'jpg':
            continue
        if imgName[0] == '.':
            imgName = imgName[2:]
        count += 1 
        print('augmenting testing data, %d / %d'%(count, totalImg))
        prefix = savepath + '%s_'%(imgName[:-4])
        if needGray:
            img = skio.imread(datapath + imgName, as_grey=True)
        else:
            img = skio.imread(datapath + imgName, as_grey=False)
        if reSize:
            resizeImg = sktr.resize(img, reSize)
            skio.imsave(prefix+'resize.jpg', resizeImg)
        if needMirror:
            mirrorImg = mirrorImage(resizeImg)
            skio.imsave(prefix+'mirror.jpg', mirrorImg)
        if needRotate:
            rotateImg = sktr.rotate(resizeImg, 180)
            skio.imsave(prefix+'rotate.jpg', rotateImg)
        # augmentImage(img, reSize, prefix)

def cropImg_augmentation(img, cropBlock, reSize=[64,64], rotateAngle=[], isMirror=False, prefix=''):
    y, x, h, w = cropBlock
    if len(img.shape) == 2:
        img = img[y:y+h, x:x+w]
    else:
        img = img[y:y+h, x:x+w]
    
    if not reSize == []:
        img = sktr.resize(img, reSize)
        skio.imsave(prefix+'resize.jpg', img)
    else:
        skio.imsave(prefix+'origin.jpg', img)
    if isMirror:
        mirrorImg = mirrorImage(img)
        skio.imsave(prefix+'mirror.jpg', mirrorImg)
    if not rotateAngle == []:
        rotateImg = sktr.rotate(img, rotateAngle)
        skio.imsave(prefix+'rotate.jpg', rotateImg)
    
    return img

def randCropImg_augmentation(img, reSize=[300, 300], cropSize=[224,224], cropNum=10, rotateAngle=[], isMirror=False, prefix=''):
    h, w = cropSize
    if len(img.shape) == 2:
        colorMode = 'gray'
        img = img[:, 80:560]
    else:
        colorMode = 'rgb'
        img = img[:, 80:560, :]
    
    if not reSize == []:
        img = sktr.resize(img, reSize)
    
    for i in range(cropNum):
        y = np.random.randint(0, reSize[0]-h)
        x = np.random.randint(0, reSize[1]-w)
        if colorMode == 'gray':
            iimg = img[y:y+h, x:x+w]
        elif colorMode == 'rgb':
            iimg = img[y:y+h, x:x+w, :]
        skio.imsave(prefix+'crop%02d.jpg'%(i+1), iimg)
    
        if isMirror:
            mirrorImg = mirrorImage(iimg)
            skio.imsave(prefix+'mirror.jpg', mirrorImg)
        if not rotateAngle == []:
            rotateImg = sktr.rotate(iimg, rotateAngle)
            skio.imsave(prefix+'rotate.jpg', rotateImg)
    
    return img

def manualCropImg_augmentation(img, reSize=[300, 300], cropSize=[224,224], cropY=[0,38,76], cropX=[0,38,76], rotateAngle=[], isMirror=False, prefix=''):
    h, w = cropSize
    if len(img.shape) == 2:
        colorMode = 'gray'
        img = img[:, 80:560]
    else:
        colorMode = 'rgb'
        img = img[:, 80:560, :]
    
    if not reSize == []:
        img = sktr.resize(img, reSize)
    
    count = 0
    for y in cropY:
        for x in cropX:
            count += 1
            if colorMode == 'gray':
                iimg = img[y:y+h, x:x+w]
            elif colorMode == 'rgb':
                iimg = img[y:y+h, x:x+w, :]
            skio.imsave(prefix+'crop%02d.jpg'%(count), iimg)
    
            if isMirror:
                mirrorImg = mirrorImage(iimg)
                skio.imsave(prefix+'mirror.jpg', mirrorImg)
            if not rotateAngle == []:
                rotateImg = sktr.rotate(iimg, rotateAngle)
                skio.imsave(prefix+'rotate.jpg', rotateImg)
    
    return img

def augmentTrainingImages_crop_gray(datapath, classIdx, savepath, reSize=[64,64], cropBlock=[0,128,300,300],\
                                    rotateAngle=180, isMirror=True):
    
    folder = os.listdir(datapath)
    totalImg = len(folder)
    count = 0
    for imgName in folder:
        if not imgName[-3:] == 'jpg':
            continue
        if imgName[0] == '.':
            imgName = imgName[2:]
        count += 1
        print('augmenting training class %d, %d / %d'%(classIdx, count, totalImg))
        prefix = savepath + '%01d_%s_'%(classIdx, imgName[:-4])
        img = skio.imread(datapath + imgName)
        img = skcolor.rgb2gray(img)
        cropImg_augmentation(img, cropBlock=cropBlock, reSize=reSize, rotateAngle=rotateAngle, isMirror=isMirror, prefix=prefix)

def augmentTestingImages_crop_gray(datapath, savepath, reSize=[64,64], cropBlock=[0,128,300,300],\
                                    rotateAngle=[], isMirror=False):
    folder = os.listdir(datapath)
    totalImg = len(folder)
    count = 0
    for imgName in folder:
        if not imgName[-3:] == 'jpg':
            continue
        if imgName[0] == '.':
            imgName = imgName[2:]
        count += 1 
        print('augmenting testing data, %d / %d'%(count, totalImg))
        prefix = savepath + '%s_'%(imgName[:-4])
        img = skio.imread(datapath + imgName)
        img = skcolor.rgb2gray(img)
        cropImg_augmentation(img, cropBlock=cropBlock, reSize=reSize, rotateAngle=rotateAngle, isMirror=isMirror, prefix=prefix)

def augmentTrainingImages_manualCrop_gray(datapath, classIdx, savepath, reSize=[300,300], cropSize=[224,224], cropY=[0,38,76], cropX=[0,38,76],\
                                    rotateAngle=[], isMirror=False):
    
    folder = os.listdir(datapath)
    totalImg = len(folder)
    count = 0
    for imgName in folder:
        if not imgName[-3:] == 'jpg':
            continue
        if imgName[0] == '.':
            imgName = imgName[2:]
        count += 1
        print('augmenting training class %d, %d / %d'%(classIdx, count, totalImg))
        prefix = savepath + '%01d_%s_'%(classIdx, imgName[:-4])
        img = skio.imread(datapath + imgName)
        img = skcolor.rgb2gray(img)
        manualCropImg_augmentation(img, cropSize=cropSize, cropY=cropY, cropX=cropX, reSize=reSize, rotateAngle=rotateAngle, isMirror=isMirror, prefix=prefix)

def augmentTestingImages_manualCrop_gray(datapath, savepath, reSize=[300,300], cropSize=[224,224], cropY=[0,38,76], cropX=[0,38,76],\
                                    rotateAngle=[], isMirror=False):
    folder = os.listdir(datapath)
    totalImg = len(folder)
    count = 0
    for imgName in folder:
        if not imgName[-3:] == 'jpg':
            continue
        if imgName[0] == '.':
            imgName = imgName[2:]
        count += 1 
        print('augmenting testing data, %d / %d'%(count, totalImg))
        prefix = savepath + '%s_'%(imgName[:-4])
        img = skio.imread(datapath + imgName)
        img = skcolor.rgb2gray(img)
        manualCropImg_augmentation(img, cropSize=cropSize, cropY=cropY, cropX=cropX, reSize=reSize, rotateAngle=rotateAngle, isMirror=isMirror, prefix=prefix)

def fragmentTrainingData(datapath, imgSize=(224,224), colorMode='gray', fragNum=10):
    ih, iw = imgSize
    folder = os.listdir(datapath)
    fragLen = np.int( np.ceil( len(folder) / fragNum ) )
    lastFragLen = len(folder) - fragLen*( fragNum-1 )
    fragCount = 0
    if colorMode == 'gray':
        X_train = np.zeros((fragLen, 1, ih, iw))
        Y_train = np.zeros((fragLen), dtype=int)

        print('data fragment %d / %d'%( fragCount+1, fragNum))
        idx = -1
        for f in folder:
#            print('data %s, fragment %d / %d'%( f, fragCount+1, fragNum))
            idx += 1
            label = np.int(f[0])
            img = skio.imread(datapath+f)
            X_train[idx, 0, ...] = img
            Y_train[idx] = label
            if idx + 1 == fragLen:
                fragCount += 1
                classNum = np.max(Y_train) + 1
                Y_train = np_utils.to_categorical(Y_train, classNum)
                sio.savemat(datapath+'fragment_%02d.mat'%(fragCount), {'X_train':X_train, 'Y_train':Y_train})
                if fragCount == fragNum:
                    X_train = np.zeros((lastFragLen, 1, ih, iw))
                    Y_train = np.zeros((lastFragLen), dtype=int)
                else:
                    X_train = np.zeros((fragLen, 1, ih, iw))
                    Y_train = np.zeros((fragLen), dtype=int)
                idx = -1
                print('reading data fragment %d / %d'%( fragCount+1, fragNum))
                
    elif colorMode == 'rgb':
        X_train = np.zeros((fragLen, 3, ih, iw))
        Y_train = np.zeros((fragLen), dtype=int)

        print('data fragment %d / %d'%( fragCount+1, fragNum))
        idx = -1
        for f in folder:
#            print('data %s, fragment %d / %d'%( f, fragCount+1, fragNum))
            idx += 1
            label = np.int(f[0])
            img = skio.imread(datapath+f)
            img = img.swapaxes(1, 2)
            img = img.swapaxes(0, 1)
            X_train[idx, 0, ...] = img
            Y_train[idx] = label
            if idx + 1 == fragLen:
                fragCount += 1
                classNum = np.max(Y_train) + 1
                Y_train = np_utils.to_categorical(Y_train, classNum)
                sio.savemat(datapath+'fragment_%02d.mat'%(fragCount), {'X_train':X_train, 'Y_train':Y_train})
                if fragCount == fragNum:
                    X_train = np.zeros((lastFragLen, 3, ih, iw))
                    Y_train = np.zeros((lastFragLen), dtype=int)
                else:
                    X_train = np.zeros((fragLen, 3, ih, iw))
                    Y_train = np.zeros((fragLen), dtype=int)
                idx = -1
                print('data fragment %d / %d'%( fragCount+1, fragNum))
                

def fragmentTestingData(datapath, imgSize=(224,224), colorMode='gray', fragNum=10):
    ih, iw = imgSize
    folder = os.listdir(datapath)
    fragLen = np.int( np.ceil( len(folder) / fragNum ) )
    lastFragLen = len(folder) - fragLen*( fragNum-1 )
    fragCount = 0
    if colorMode == 'gray':
        X_test = np.zeros((fragLen, 1, ih, iw))
        nameList = []

        print('data fragment %d / %d'%( fragCount+1, fragNum))
        idx = -1
        for f in folder:
#            print('data %s, fragment %d / %d'%( f, fragCount+1, fragNum))
            idx += 1
            img = skio.imread(datapath+f)
            X_test[idx, 0, ...] = img
            nameList.append(f)
            if idx + 1 == fragLen:
                fragCount += 1
                sio.savemat(datapath+'fragment_%02d.mat'%(fragCount), {'X_test':X_test, 'nameList':nameList})
                if fragCount == fragNum:
                    X_test = np.zeros((lastFragLen, 1, ih, iw))
                    nameList = []
                else:
                    X_test = np.zeros((fragLen, 1, ih, iw))
                    nameList = []
                idx = -1
                print('data fragment %d / %d'%( fragCount+1, fragNum))
                
    elif colorMode == 'rgb':
        X_test = np.zeros((fragLen, 3, ih, iw))
        nameList = []

        print('data fragment %d / %d'%( fragCount+1, fragNum))
        idx = -1
        for f in folder:
#            print('data %s, fragment %d / %d'%( f, fragCount+1, fragNum))
            idx += 1
            img = skio.imread(datapath+f)
            img = img.swapaxes(1, 2)
            img = img.swapaxes(0, 1)
            X_test[idx, 0, ...] = img
            nameList.append(f)
            if idx + 1 == fragLen:
                fragCount += 1
                sio.savemat(datapath+'fragment_%02d.mat'%(fragCount), {'X_test':X_test, 'nameList':nameList})
                if fragCount == fragNum:
                    X_test = np.zeros((lastFragLen, 3, ih, iw))
                    nameList = []
                else:
                    X_test = np.zeros((fragLen, 3, ih, iw))
                    nameList = []
                idx = -1
                print('data fragment %d / %d'%( fragCount+1, fragNum))  
    










