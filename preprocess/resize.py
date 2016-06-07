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

'''
    Save all resized images to training and testing folders.
    training images with all labels are saved into one folder, the label is the first character of the file names.
'''

def resize_image(img, img_size):
    return sktr.resize(img, output_shape=img_size)


def resize_train_image(from_path, img_size, force=False):
    to_path = "%s_%s_" % (from_path, "_".join(img_size))
    args = {"img_size": img_size}
    loop_process_train_image(from_path, to_path, resize_image, args, force)


def resize_image_main():
    img_size = (244, 244)
    force = False

    train_base_path = config.Project.original_training_folder
    train_img_path_list = ["%s_shift_down_0.2" % train_base_path,
                           "%s_shift_up_0.2" % train_base_path,
                           "%s_shift_left_0.2" % train_base_path,
                           "%s_shift_right_0.2" % train_base_path, ]
    for path in train_img_path_list:
        resize_train_image(path, img_size, force)

if __name__ == "__main__":
    level = logging.INFO
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    resize_image_main()
