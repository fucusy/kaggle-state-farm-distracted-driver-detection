__author__ = 'fucus'
import os
import skimage.io
import logging
from feature.hog import get_hog
from config import Project
import pickle


cache_path = "%s/cache" % Project.project_path
if not os.path.exists(cache_path):
    os.makedirs(cache_path)
hog_feature_cache_file_path = "%s/%s" % (cache_path, "hog_feature_cache.pickle")


def load_cache():
    # load cache
    hog_feature_cache = {}
    if os.path.exists(hog_feature_cache_file_path):
        hog_feature_file = open(hog_feature_cache_file_path, "rb")
        hog_feature_cache = pickle.load(hog_feature_file)
        hog_feature_file.close()

    return hog_feature_cache

def save_cache(hog_feature_cache):
    hog_feature_file = open(hog_feature_cache_file_path, "wb")
    pickle.dump(hog_feature_cache, hog_feature_file)
    hog_feature_file.close()


def load_train_feature(img_data_path, hog_feature_cache, limit=-1):

    driver_type_num_check = {"c0": 2489, "c1": 2267, "c2": 2317, "c3": 2346, "c4": 2326
                            , "c5": 2312, "c6": 2325, "c7": 2002, "c8": 1911,"c9": 2129}
    x_feature = []
    y = []
    relevant_image_path_list = []

    logging.info("start to check the train image")
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
            img_path = "%s/%s" % (driver_type, img)
            relevant_image_path_list.append(img_path)
            y.append(driver_type)

    logging.info("check the train image end")

    logging.info("start to load feature from train image")
    count = 0
    for relevant_image_path in relevant_image_path_list:
        if count >= limit > 0:
            break
        if count % 1000 == 0:
            logging.info("extract %s th image feature now" % count)
        count += 1
        full_path = "%s/%s" % (img_data_path, relevant_image_path)
        x_feature.append(extract_feature(full_path, hog_feature_cache))
    logging.info("load feature from train image end")

    return relevant_image_path_list[:count], x_feature[:count], y[:count]


def load_test_feature(img_data_path, hog_feature_cache, limit=-1):
    test_img_num = 79726
    x_feature = []
    relevant_image_path_list = sorted([x for x in os.listdir("%s" % img_data_path) if x.endswith(".jpg")])
    if len(relevant_image_path_list) != test_img_num:
        logging.warning("the test images number:%d is not equal to %d, it's incorrect"
                        % (len(relevant_image_path_list), test_img_num))
    else:
        logging.info("the test images number:%d is equal to %d, it's correct"
                        % (len(relevant_image_path_list), test_img_num))
    logging.info("start to load feature from test image")
    count = 0
    for img in relevant_image_path_list:
        if count >= limit > 0:
            break
        if count % 1000 == 0:
            logging.info("extract %s th image feature now" % count)
        count += 1
        img_path = "%s/%s" % (img_data_path, img)
        x_feature.append(extract_feature(img_path, hog_feature_cache))

    logging.info("load feature from test image end")
    return relevant_image_path_list[:count], x_feature[:count]

def extract_feature(img_path, hog_feature_cache):
    img_name = img_path.split("/")[-1]
    img = skimage.io.imread(img_path)
    feature = []
    if img_name in hog_feature_cache:
        hog_feature = hog_feature_cache[img_name]
    else:
        hog_feature = get_hog(img)
        hog_feature_cache[img_name] = hog_feature
    feature += hog_feature
    return feature