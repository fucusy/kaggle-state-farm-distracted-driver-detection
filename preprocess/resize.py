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

'''
    Save all resized images to training and testing folders.
    training images with all labels are saved into one folder, the label is the first character of the file names.
'''

def resize_image(img, img_size):
    return sktr.resize(img, output_shape=img_size)


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



if __name__ == "__main__":
    level = logging.DEBUG
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    resize_image_main()
