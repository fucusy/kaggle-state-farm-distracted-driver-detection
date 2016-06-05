__author__ = 'fucus'
import os
import skimage.io
import logging
from feature.hog import get_hog
from feature.lbp import get_lbp_his
from config import Project
import config
import pickle
from tool.keras_tool import load_train_validation_data_set, load_test_data_set


cache_path = "%s/cache" % Project.project_path
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

hog_feature_cache_file_path = "%s/%s" % (cache_path, "hog_feature_cache.pickle")
lbp_feature_cache_file_path = "%s/%s" % (cache_path, "lbp_feature_cache.pickle")

def load_cache():
    # load cache
    hog_feature_cache = {}
    if os.path.exists(hog_feature_cache_file_path):
        hog_feature_file = open(hog_feature_cache_file_path, "rb")
        hog_feature_cache = pickle.load(hog_feature_file)
        hog_feature_file.close()


    lbp_feature_cache = {}
    if os.path.exists(lbp_feature_cache_file_path):
        lbp_feature_file = open(lbp_feature_cache_file_path, "rb")
        lbp_feature_cache = pickle.load(lbp_feature_file)
        lbp_feature_file.close()


    return hog_feature_cache, lbp_feature_cache

def save_cache(hog_feature_cache):
    hog_feature_file = open(hog_feature_cache_file_path, "wb")
    pickle.dump(hog_feature_cache, hog_feature_file)
    hog_feature_file.close()


def load_train_feature(img_data_path, hog_feature_cache, lbp_feature_cache, limit=-1):

    x_feature = []
    y = []
    relevant_image_path_list = []
    train_data, _ = load_train_validation_data_set(config.Project.train_img_folder_path)


    image_path_list = train_data.image_path_list
    image_label_list = train_data.image_label_list

    logging.info("start to load feature from train image")
    count = 0
    for i in range(len(image_path_list)):
        relevant_image_path = image_path_list[i].split("/")[-2:]
        if count >= limit > 0:
            break
        if count % 1000 == 0:
            logging.info("extract %s th image feature now" % count)
        count += 1
        full_path = "%s/%s" % (img_data_path, relevant_image_path)
        x_feature.append(extract_feature(full_path, hog_feature_cache, lbp_feature_cache))
    logging.info("load feature from train image end")

    return image_path_list[:count], x_feature[:count], image_label_list[:count]


def load_test_feature(img_data_path, hog_feature_cache, lbp_feature_cache, limit=-1):



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


    test_data = load_test_data_set(config.Project.test_img_folder_path)
    relevant_image_path_list = [os.path.basename(x) for x in test_data.image_path_list]

    count = 0
    for img in relevant_image_path_list:
        if count >= limit > 0:
            break
        if count % 1000 == 0:
            logging.info("extract %s th image feature now" % count)
        count += 1
        img_path = "%s/%s" % (img_data_path, img)
        x_feature.append(extract_feature(img_path, hog_feature_cache, lbp_feature_cache))

    logging.info("load feature from test image end")
    return relevant_image_path_list[:count], x_feature[:count]

def extract_feature(img_path, hog_feature_cache, lbp_feature_cache):
    img_name = img_path.split("/")[-1]
    img = skimage.io.imread(img_path)
    feature = []
    if img_name in hog_feature_cache:
        hog_feature = hog_feature_cache[img_name]
    else:
        hog_feature = get_hog(img)
        hog_feature_cache[img_name] = hog_feature


    if img_name in lbp_feature_cache:
        lbp_feature = lbp_feature_cache[img_name]
    else:
        lbp_feature = get_lbp_his(img)
        lbp_feature_cache[img_name] = lbp_feature

    feature += list(hog_feature)
    feature += list(lbp_feature)

    return feature
