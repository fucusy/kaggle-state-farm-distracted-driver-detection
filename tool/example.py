import sys
sys.path.append('../')

import config
import tool.data_tools as dt
import tool.model_tools as mt
import tool.model_inference as mi

from tool.data_tools import DataSet
from tool.model_tools import KerasModel
from tool.model_tools import KerasFeatureExtractor
import os
import numpy as np

'''=====================================Data resize=================================================='''
if not os.path.exists(config.Project.train_img_folder_path):
    dt.resize_image()


if not os.path.exists(config.Data.mean_image_file_name):
    mean_image = dt.compute_mean_image()



'''====================================Train and test================================================'''

data_set = DataSet()

model = KerasModel(data_set=data_set)
model.train_model(save_best=True)
model.set_model_arch(model_arch=mi.inference(input_shape=   data_set.get_img_size,
                                                         classNum=      data_set._class_num,
                                                         weights_file=  '') )

''' Note that if you already have weights file, you can either load the weights via inference function,
    or load them by KerasModel API. '''

model.set_model_weights(config.CNN.model_weights_file_name)

model.predict_model()










