# -*- coding: utf-8 -*-
"""
Created on Tue May 31 20:43:33 2016

@author: ZhengLiu
"""


import sys
sys.path.append('../')

import config
import numpy as np
import skimage
import skimage.io as skio
import os
import logging

from tool.keras_tool import load_image_path_list

''' Make images, which maybe in shape of (n, h, w, ch) for multiple color images,
                                         (h, w, ch) for single color images,
                                         (n, h, w) for multiple gray images or
                                         (h, w) for single gray images,
    to data form that capible to keras models, which in the shape of (n, ch, h, w).
'''
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

def compute_mean_image(training_data_path=config.Project.train_img_folder_path
                       , testing_data_path=config.Project.test_img_folder_path
                       , save_file=config.Data.mean_image_file_name):

    if os.path.exists(save_file):
        logging.info("mean file already exists, return it directly")
        return np.fromfile(save_file)
    logging.info('computing mean images')
    folder = ["c%d" % x for x in range(10)]
    total_num = 0
    mean_image = None
    for f in folder:
        folder_path = os.path.join(training_data_path, f)
        for img_path in load_image_path_list(folder_path):
            total_num += 1
            img = skimage.img_as_float(skio.imread(img_path))
            if mean_image is None:
                mean_image = img
            else:
                mean_image += img

    for file_path in load_image_path_list(testing_data_path):
        total_num += 1
        img = skimage.img_as_float( skio.imread(file_path))
        mean_image += img

    mean_image /= total_num
    
    if len(mean_image.shape) == 2:
        '''if gray, (h, w) to (1, h, w)'''
        tmp = np.zeros((1, mean_image.shape[0], mean_image.shape[1]))
        tmp[0, ...] = mean_image
        mean_image = tmp
    else:
        '''if color, swap (h, w, ch) to (ch, h, w)'''
        mean_image = mean_image.swapaxes(1,2)
        mean_image = mean_image.swapaxes(0,1)
    if save_file != "":
        base_path = os.path.dirname(save_file)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        with open(save_file, 'wb') as f:
            np.save(f, mean_image)
            logging.debug("saving mean file to %s" % save_file)

    return mean_image

if __name__ == '__main__':

    LEVELS = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}
    level = logging.INFO
    if len(sys.argv) >= 2:
        level_name = sys.argv[1]
        level = LEVELS.get(level_name, logging.INFO)
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    '''=====================================Data resize=================================================='''


    compute_mean_image()

