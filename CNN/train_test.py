import sys
sys.path.append('../')

from configuration import Config
import data_tools as dt
#import model_tools as mt
#import model_inference as mi

from data_tools import ImageSet
from model_tools import KerasModel

import os
#import numpy as np

config = Config( img_size =                                 [3, 64, 64],
                 crop_num =                                 0,
                 fragment_size =                            4096,
                 batch_size =                               32,
                 class_num =                                10,
                 validation_split =                         0.15,
                 train_sample_mode =                        'equal',
                 model_name =                               'vgg_std16_model',
                 model_arch_file_name =                     '',
                 model_weight_file_name =                   '/home/liuzheng/competition/kaggle/distractedDrivers/vgg16_weights.h5',
                 optimizer =                                'adam',
                 train_iter =                               30,
                 test_batch_size =                          64)

'''=====================================Data resize=================================================='''
if not os.path.exists(config.project.train_img_folder_path):
    dt.crop_resize_image(original_training_data_path=       config.project.original_training_folder,
                         original_testing_data_path=        config.project.original_testing_folder,
                         training_save_path=                config.project.train_img_folder_path,
                         testing_save_path=                 config.project.test_img_folder_path,
                         img_size=                          config.data.img_size,
                         crop_num=                          config.data.crop_num,
                         postfix=                           config.data.augmentation_postfix,
                         mean_save_flag=                    True,
                         mean_save_file=                    config.data.mean_image_file_name)


if not os.path.exists(config.data.mean_image_file_name):
    mean_image = dt.compute_mean_image(config.project.train_img_folder_path,
								   config.project.test_img_folder_path,
								   save_flag=True,
								   save_file=config.data.mean_image_file_name)



'''====================================Train and test================================================'''

data_set = ImageSet(training_data_path=     config.project.train_img_folder_path,
                   testing_data_path=       config.project.test_img_folder_path,
                   mean_image_file_name=    config.data.mean_image_file_name,
                   driver_list_file=        config.data.driver_list_file,
                   augmentation_postfix=    config.data.augmentation_postfix,
                   fragment_size=	      config.data.fragment_size,
                   img_size=		      config.data.img_size,
                   validation_split=        config.data.validation_split,
                   batch_size=	           config.data.batch_size,
                   class_num=	           config.data.class_num)

#model_arch=mi.inference_xavier_prelu_224(input_shape=    data_set.get_img_size(),
#                                         class_num=       data_set._class_num,
#                                         optimizer=      config.nn.optimizer,
#                                         weights_file=   '')

model = KerasModel(model_name=              config.nn.model_name,
                   data_set=	           data_set,
                   model_inference=         config.get_model(),
                   test_batch_size=		config.nn.test_batch_size,
                   n_iter=				config.nn.train_iter,
                   model_arch_file=		config.nn.model_arch_file_name,
                   model_weight_file=		config.nn.model_weight_file_name,
                   model_save_path=		config.nn.model_save_path,
                   prediction_save_file=	config.nn.prediction_save_file)

if config.print_info(do_not_ask=True):
    model.train_model(save_best=True)
    model.set_model_weights(model._best_weight_file)
    #model.set_model_weights('/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_112/inference_xavier_prelu_112_weights_best_vLoss1.17028_vAcc0.679.h5')
    model.predict()


#model.set_model_arch(model_arch=            mi.inference(input_shape=   data_set.get_img_size,
#                                                         classNum=      data_set._class_num,
#                                                         weights_file=  '') )
#
#''' Note that if you already have weights file, you can either load the weights via inference fuction,
#    or load them by KerasModel API. '''
#
#model.set_model_weights(config.nn.model_weights_file_name)
#
#
#extractor = KerasFeatureExtractor(model_name=               'vgg_keras',
#                                  data_set=                 data_set,
#                                  model_arch_file=          config.nn.model_arch_file_name,
#                                  model_weights_file=       config.nn.model_weights_file_name,
#                                  feature_layer_index=      31,# 31 is for vgg_keras, which is the index of the 'Flatten' layer.
#                                  feature_save_path=        config.nn.feature_save_path)
#
#images = np.random.rand(10, 3, 224, 224)
#''' if save_file='', the feature is not saved. Note we save features in '.npy' file.'''
#feature = extractor.extract_feature_images(images, save_file='')














