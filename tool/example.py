import sys
sys.path.append('../')

import config
import data_tools as dt
import model_tools as mt
import model_inference as mi

from data_tools import DataSet
from model_tools import KerasModel
from model_tools import KerasFeatureExtractor
import os
import numpy as np

'''=====================================Data resize=================================================='''
if not os.path.exists(config.Project.train_img_folder_path):
    os.mkdir(config.Project.train_img_folder_path)
    os.mkdir(config.Project.test_img_folder_path)
    dt.resize_image(config.Project.original_training_folder, config.Project.original_testing_folder,
	            training_save_path=config.Project.train_img_folder_path, testing_save_path=config.Project.test_img_folder_path,
                 img_size=config.Data.img_size)


if not os.path.exists(config.Data.mean_image_file_name):
    mean_image = dt.compute_mean_image(config.Project.train_img_folder_path,
								   config.Project.test_img_folder_path,
								   save_flag=True,
								   save_file=config.Data.mean_image_file_name)



'''====================================Train and test================================================'''

data_set = DataSet(training_data_path=      config.Project.train_img_folder_path,
                   testing_data_path=       config.Project.test_img_folder_path,
                   mean_image_file_name=    config.Data.mean_image_file_name,
                   fragment_size=	      config.Data.fragment_size,
                   img_size=		      config.Data.img_size,
                   validation_split=        config.Data.validation_split,
                   batch_size=	           config.Data.batch_size,
                   class_num=	           config.Data.class_num)


model = KerasModel(model_name=              config.CNN.model_name,
                   data_set=	           data_set,
                   test_batch_size=		config.CNN.test_batch_size,
                   n_iter=				config.CNN.train_iter,
                   model_arch_file=		config.CNN.model_arch_file_name,
                   model_weights_file=		config.CNN.model_weights_file_name,
                   model_save_path=		config.CNN.model_save_path,
                   prediction_save_file=	config.CNN.prediction_save_file)

model.train_model(save_best=True)
model.predict()


model.set_model_arch(model_arch=            mi.inference(input_shape=   data_set.get_img_size,
                                                         classNum=      data_set._class_num,
                                                         weights_file=  '') )

''' Note that if you already have weights file, you can either load the weights via inference fuction,
    or load them by KerasModel API. '''

model.set_model_weights(config.CNN.model_weights_file_name)


extractor = KerasFeatureExtractor(model_name=               'vgg_keras',
                                  data_set=                 data_set,
                                  model_arch_file=          config.CNN.model_arch_file_name,
                                  model_weights_file=       config.CNN.model_weights_file_name,
                                  feature_layer_index=      31,# 31 is for vgg_keras, which is the index of the 'Flatten' layer.
                                  feature_save_path=        config.CNN.feature_save_path)

images = np.random.rand(10, 3, 224, 224)
''' if save_file='', the feature is not saved. Note we save features in '.npy' file.'''
feature = extractor.extract_feature_images(images, save_file='')














