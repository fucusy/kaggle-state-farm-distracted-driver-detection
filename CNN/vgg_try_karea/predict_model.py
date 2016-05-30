__author__ = 'fucus'

import sys
# append the project root as python lib search
sys.path.append("../../")

from tool.keras_tool import *
import config
from tool.file import generate_result_file
from model.cnn_model import VGG_16
import numpy as np

batch_size = 128

model = VGG_16(weights_path=config.CNN.keras_train_weight)

test_data_set = load_test_data_set(config.Project.test_img_folder_path)
predict = []

while test_data_set.have_next():
    img_list, _ = test_data_set.next_batch(128)
    result = model.predict(img_list)
    predict += list(result)
predict = np.array(predict)
generate_result_file(test_data_set.image_path_list[:len(predict)], predict)
