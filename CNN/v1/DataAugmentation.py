# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:36:01 2016

@author: liuzheng
"""

import skimage.io as skio
import skimage.transform as sktr
import numpy as np
import os
import random

def mirrorImage(img):
    h,w,ch = img.shape
    mirrorImg = np.zeros_like(img)
    for x in range(w/2):
        mirrorImg[:,x,:] = img[:,w-1-x,:]
        mirrorImg[:,w-1-x,:] = img[:,x,:]
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

def augmentImages_distractedDrivers(datapath, imgNameList, personIndexList, classList, savepath, reSize=[64,64]):
    n = len(imgNameList)
    for i in range(n):
        imgName = imgNameList[i]
        personIdx = personIndexList[i]
        c = classList[i]
        prefix = savepath + '%03d_%01d_%s_'%(personIdx, c, imgName)
        img = skio.imread(datapath + imgName)
        augmentImage(img, reSize, prefix)

















