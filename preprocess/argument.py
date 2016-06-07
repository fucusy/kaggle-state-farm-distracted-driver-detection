__author__ = 'fucus'


import sys
sys.path.append("../")

import numpy as np
from skimage.io import imsave
from skimage.io import imread
import config
import os
import shutil
import skimage.io as skio
import logging
from tool.keras_tool import load_image_path_list

def shift_left(img, left=10.0):
    """

    :param numpy.array img: represented by numpy.array
    :param float left: how many pixels to shift to left, this value can be negative that means shift to
                    right {-left} pixels
    :return: numpy.array
    """
    if 0 < abs(left) < 1:
        left = int(left * img.shape[1])
    else:
        left = int(left)

    img_shift_left = np.zeros(img.shape)
    if left > 0:
        img_shift_left = img[:, left:, :]
    else:
        img_shift_left = img[:, :left, :]

    return img_shift_left


def shift_right(img, right=10.0):
    return shift_left(img, -right)


def shift_up(img, up=10.0):
    """
    :param numpy.array img: represented by numpy.array
    :param float up: how many pixels to shift to up, this value can be negative that means shift to
                    down {-up} pixels
    :return: numpy.array
    """


    if 0 < abs(up) < 1:
        up = int(up * img.shape[0])
    else:
        up = int(up)

    img_shift_up = np.zeros(img.shape)
    if up > 0:
        img_shift_up = img[up:, :, :]
    else:
        img_shift_up = img[:up, :, :]

    return img_shift_up

def shift_down(img, down=10.0):
    return shift_up(img, -down)


def argument_image(from_path, to_path, method, args, force=False):
    if force:
        if os.path.exists(to_path):
            shutil.rmtree(to_path)
    if os.path.exists(to_path):
        logging.info("resize path exists, no need to resize again, skip this")
        return

    class_list = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    logging.info("doing argument now")

    if not os.path.exists(to_path):
        for c in class_list:
            class_path = os.path.join(to_path, c)
            os.makedirs(class_path)

    for c in class_list:
        class_path = os.path.join(from_path, c)
        file_paths = load_image_path_list(class_path)
        for file_path in file_paths:
            f = os.path.basename(file_path)
            img = skio.imread(file_path)
            args["img"] = img
            img = method(**args)
            save_path = os.path.join(to_path, c, f)
            skio.imsave(save_path, img)
    logging.info("argument done")


def argument_main():
    from_path = config.Project.original_training_folder
    to_path = "%s_shift_left_0.2" % from_path
    argument_image(from_path, to_path, shift_left, {"left": 0.2})


if __name__ == '__main__':
    driver = imread(config.Project.test_img_example_path)    
    
    driver_shift_left = shift_left(driver, 0.2)
    driver_shift_right = shift_right(driver, 0.2)

    driver_shift_up = shift_up(driver, 0.2)
    driver_shift_down = shift_down(driver, 0.2)

    imsave("driver.jpg", driver)
    imsave("driver_shift_left.jpg", driver_shift_left)
    imsave("driver_shift_right.jpg", driver_shift_right)
    imsave("driver_shift_up.jpg", driver_shift_up)
    imsave("driver_shift_down.jpg", driver_shift_down)




