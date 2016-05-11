# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:55:52 2016

@author: liuzheng
"""

import os
#import random
import numpy as np
import glob

def makeTrainList_distractedDrivers(datapath, prefix):

    countList = []
    count = 0
    labelList = []
    label = -1
    nameList = []
    imageList = os.listdir(datapath)
    # imageList.sort()
    for imgName in imageList:
        print('reading data %s'%(imgName))
        label = np.int(imgName[4])
        count += 1
        nameList.append(imgName)
        labelList.append(label)
        countList.append(count)
        
    
    print('randperm')
    index = range(len(countList))
    np.random.shuffle(index)
    countList = np.array(countList)
    countList = countList[index]
    labelList = np.array(labelList)
    labelList = labelList[index]
    nameList = np.array(nameList)
    nameList = nameList[index]
    
    print('writing')
    f = open(prefix+'trainDatalist_distractedDrivers.lst', 'w')
    for i in range(len(index)):
        f.writelines("%d \t %d \t %s\n"%(countList[i], labelList[i], nameList[i]))
    
    f.close()
