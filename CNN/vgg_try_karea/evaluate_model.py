__author__ = 'fucus'

import sys
# append the project root as python lib search
sys.path.append("../../")


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


test_data_set = load_test_data_set(config.Project.test_img_folder_path)
predict = []

while test_data_set.have_next():
    img_list, _ = test_data_set.next_batch(128)
    result = model.predict(img_list)
    predict.append(result)
    break
predict = np.array(predict)
generate_result_file(test_data_set.image_path_list[:len(predict)], predict)
