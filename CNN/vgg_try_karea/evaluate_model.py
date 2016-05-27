__author__ = 'fucus'

import sys
# append the project root as python lib search
sys.path.append("../../")

from keras.utils.np_utils import to_categorical
from CNN.keras_tool import *
import config
from tool.file import generate_result_file
from CNN.vgg_try_karea.kaggle_driver_cnn import VGG_16
import numpy as np

batch_size = 128

try:
    model = load_model_from_file(config.CNN.keras_structure_files, config.CNN.keras_train_weight)
    print("load graph from json")
except:
    print("fail to load graph from json")
    model = VGG_16(config.CNN.keras_train_weight)
    print("loaded graph from code")


data_set = load_train_data_set(config.Project.train_img_folder_path)
predict = []

while data_set.have_next():
    img_list, img_label, _ = data_set.next_batch(1024, need_label=True)
    img_label_cate = to_categorical(img_label)
    loss_and_metrics = model.evaluate(img_list, img_label_cate, batch_size=1024)
    print(loss_and_metrics)
