# -*- coding: utf-8 -*-
import sys
sys.path.append("../../")
import config
import os

import glob
import math


from scipy.misc import imread, imresize

def read_image(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = imread(path, True)
    elif color_type == 3:
        img = imread(path)
    # Reduce size
    img_resized = imresize(img, (img_cols, img_rows))
    return img_resized

def get_driver_imgs_list():
    imgs_list = dict()
    path = config.Project.driver_img_list_path
    print('read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        imgs_list[arr[2]] = arr[0]
    f.close()
    return imgs_list

def load_train(img_rows, img_cols, color_type=1):
    X_train = []
    y_train = []
    driver_id = []

    imgs_list = get_driver_imgs_list()

    print('read train images')
    for k in range(10):
        print('load folder c{}'.format(k))
        path = os.path.join(config.Project.train_img_folder_path, 'c' + str(k), '*.jpg')
        files = glob.glob(path)
        for f in files:
            fbase = os.path.basename(f)
            img = read_image(f, img_rows, img_cols, color_type)
            X_train.append(img)
            y_train.append(k)
            driver_id.append(imgs_list[fbase])

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers

def load_test(img_rows, img_cols, color_type=1):
    print('read test images')
    path = os.path.join(config.Project.test_img_folder_path, '*.jpg')
    files = glob.glob(path)
    X_test = []
    test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for f in files:
        fbase = os.path.basename(f)
        img = read_image(f, img_rows, img_cols, color_type)
        X_test.append(img)
        test_id.append(fbase)
        total += 1
        if total%thr == 0:
            print('read {} images from {}'.format(total, len(files)))

    return X_test, test_id

def load_data(img_rows, img_cols, color_type=1):
    X_train, y_train, driver_id, unique_drivers = load_train(img_rows, img_cols, color_type)
    X_test, test_id = load_test(img_rows, img_cols, color_type)
    return X_train, y_train, driver_id, unique_drivers, X_test, test_id

