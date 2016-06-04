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

'''
    Save all resized images to training and testing folders.
    training images with all labels are saved into one folder, the label is the first character of the file names.
'''



def resize_image(original_training_data_path=config.Project.original_training_folder
                 , original_testing_data_path=config.Project.original_testing_folder
                 , training_save_path=config.Project.train_img_folder_path
                 , testing_save_path=config.Project.test_img_folder_path
                 , img_size=config.Data.img_size
                 , force=False):
    if force:
        if os.path.exists(training_save_path):
            shutil.rmtree(training_save_path)
        if os.path.exists(testing_save_path):
            shutil.rmtree(testing_save_path)
    if os.path.exists(training_save_path):
        logging.info("resize path exists, no need to resize again, skip this")
        return

    class_list = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    logging.info("doing resize now")

    if not os.path.exists(training_save_path):
        for c in class_list:
            class_path = os.path.join(training_save_path, c)
            os.makedirs(class_path)
    if not os.path.exists(testing_save_path):
        os.makedirs(testing_save_path)

    as_grey = (img_size[0] == 1)

    logging.info("resize train image now")
    for c in class_list:
        class_path = os.path.join(original_training_data_path, c)
        file_paths = load_image_path_list(class_path)
        for file_path in file_paths:
            f = os.path.basename(file_path)
            img = skio.imread(file_path , as_grey=as_grey )
            img = sktr.resize(img, output_shape=[img_size[1], img_size[2]])
            save_path = os.path.join(training_save_path, c, f)
            skio.imsave(save_path, img)

    logging.info("resize test image now")
    file_paths = load_image_path_list(original_testing_data_path)
    for file_path in file_paths:
        f = os.path.basename(file_path)
        img = skio.imread(file_path , as_grey=as_grey )
        img = sktr.resize(img, output_shape=[img_size[1], img_size[2]])
        save_path = os.path.join(testing_save_path, f)
        skio.imsave(save_path, img)

    logging.info("resize done")

if __name__ == "__main__":
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

    resize_image(force=True)
