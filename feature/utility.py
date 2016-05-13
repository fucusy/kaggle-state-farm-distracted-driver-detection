__author__ = 'fucus'

import os
import skimage.io
import logging

def load_train_data(img_data_path):
    driver_type_num_check = {"c0": 2489, "c1": 2267, "c2": 2317, "c3": 2346, "c4": 2326
                            , "c5": 2312, "c6": 2325, "c7": 2002, "c8": 1911,"c9": 2129}

    x_data = {}
    x_img = []
    x_name = []
    y = []
    driver_type_list = sorted(driver_type_num_check.keys())
    for driver_type in driver_type_list:
        images = sorted([x for x in os.listdir("%s/%s" % (img_data_path, driver_type)) if x.endswith(".jpg")])
        if len(images) != driver_type_num_check[driver_type]:
            logging.warning("the type of %s train images number:%d is not equal to %d, it's incorrect"
                            % (driver_type, len(images), driver_type_num_check[driver_type]))
        else:
            logging.info("the type of %s train images number:%d is equal to %d, it's correct"
                            % (driver_type, len(images), driver_type_num_check[driver_type]))
        for img in images:
            img_path = "%s/%s/%s" % (img_data_path, driver_type, img)
            x_name.append(img)
            x_img.append(skimage.io.imread(img_path))
            y.append(driver_type)
    x_data["img"] = x_img
    x_data["name"] = x_name
    return x_data, y


def load_test_data(img_data_path):
    x_data = {}
    x_img = []
    x_name = []
    test_img_num = 79726
    images = sorted([x for x in os.listdir("%s" % img_data_path) if x.endswith(".jpg")])
    if len(images) != test_img_num:
        logging.warning("the test images number:%d is not equal to %d, it's incorrect"
                        % (len(images), test_img_num))
    else:
        logging.info("the test images number:%d is equal to %d, it's correct"
                        % (len(images), test_img_num))
    for img in images:
        img_path = "%s/%s" % (img_data_path, img)
        x_name.append(img)
        x_img.append(skimage.io.imread(img_path))
    x_data["img"] = x_img
    x_data["name"] = x_name
    return x_data


def extract_feature(x_img):
    return x_img