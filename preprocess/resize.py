__author__ = 'fucus'

import sys
sys.path.append("../")

import config
import os
import logging
import skimage.io as skio
import skimage.transform as sktr
import shutil
from tool.keras_tool import load_image_path_list
from preprocess.argument import loop_process_train_image
from preprocess.argument import loop_process_test_image
import numpy as np

'''
    Save all resized images to training and testing folders.
    training images with all labels are saved into one folder, the label is the first character of the file names.
'''

def resize_image(img, img_size):
    return sktr.resize(img, output_shape=img_size)


def add_padding(img, img_size):
    result = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    origin_size = img.shape
    offset_height = (img_size[0] - origin_size[0]) / 2
    offset_width = (img_size[1] - origin_size[1]) / 2

    for w in range(origin_size[0]):
        for h in range(origin_size[1]):
            result[w + offset_height, h + offset_width, :] = img[w, h, :]

    return result

def resize_image_main(from_path, img_size, force=False):
    for path in from_path:
        img_size_str_list = [str(x) for x in img_size]
        to_path = "%s_%s" % (path, "_".join(img_size_str_list))
        args = {"img_size": img_size}
        logging.info("resize image in %s to %s" % (path, to_path))
        if 'c0' in os.listdir(path):
            loop_process_train_image(path, to_path, resize_image, args, force)
        else:
            loop_process_test_image(path, to_path, resize_image, args, force)

def crop_center(img, img_size):

    height, width = img.shape[0], img.shape[1]

    corp_img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)

    height_offset = (height - img_size[0]) / 2
    width_offset = (width - img_size[1]) / 2

    for h in range(img_size[0]):
        for w in range(img_size[1]):
            corp_img[h, w, :] = img[h + height_offset, w + width_offset, :]

    return corp_img

if __name__ == "__main__":
    level = logging.DEBUG
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    driver = skio.imread(config.Project.project_path + "/test_image/img_94.jpg")
    to_size = (224, 224)
    crop = crop_center(driver, to_size)
    skio.imsave("crop_center_94.jpg", crop)
