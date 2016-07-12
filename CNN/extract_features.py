# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 16:49:47 2016

@author: liuzheng
"""

from data_tools import ImageSet, FeatureSet
from model_tools import KerasFeatureExtractor, KerasModel
import data_tools as dt
import os
from configuration import Config
#import configuration
config = Config( img_size =                                 [3, 112, 112],
                 crop_num =                                 0,
                 fragment_size =                            2048,
                 batch_size =                               32,
                 class_num =                                10,
                 validation_split =                         0.2,
                 train_sample_mode =                        'all',
                 
                 feature_layer_index =                      -1,
                 feature_fragment_size =                    4096,
#                 training_feature_file_prefix =             'train_features_',#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggTrainingFeature'
#                 training_label_file_prefix =               'train_labels',#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggTrainingLabel'
#                 
#                 testing_feature_file_prefix =              'test_features_',#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggTestingImageFeature'
#                 testing_name_file_prefix =                 'test_names_',#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggTestingImageName'
#
#                 validation_feature_file_prefix =           'valid_features_',#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggValidationFeature'
#                 validation_label_file_prefix =             'valid_label_',#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggValidationLabel'
                 
                 model_name =                               'vgg_std16_extractor',
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
                                         
keras_extractor = KerasFeatureExtractor(model_name=                                    config.nn.model_name,
                                        data_set=                                      data_set,
                                        model_inference=                               config.get_model(),
                                        model_arch_file=                               config.nn.model_arch_file_name,
                                        model_weight_file=                            config.nn.model_weight_file_name,
                                        feature_layer_index=                           config.feature.feature_layer_index,
                                        feature_folder=                                 config.feature.feature_folder
                                        )
#                                        training_feature_file_prefix=                  config.feature.training_feature_file_prefix,
#                                        training_label_file_prefix=                    config.feature.training_label_file_prefix,
##                                        training_feature_fragment_num=                 config.feature.training_feature_fragment_num,
#                                        testing_feature_file_prefix=                   config.feature.testing_feature_file_prefix,
#                                        testing_name_file_prefix=                      config.feature.testing_name_file_prefix,
##                                        testing_feature_fragment_num=                  config.feature.testing_feature_fragment_num,
#                                        validation_feature_file_prefix=                config.feature.validation_feature_file_prefix,
#                                        validation_label_file_prefix=                  config.feature.validation_label_file_prefix)
#                                        validation_features_fragment_num=              config.feature.validation_feature_fragment_num,)
#images = np.random.rand(32, 3, 224, 224)
#
#feat = keras_extractor.extract_feature_images(images)

keras_extractor.extract_training_features()
keras_extractor.extract_testing_features()

#feature_set = FeatureSet(training_feature_file_prefix=                  config.feature.training_feature_file_prefix,
#                         training_label_file_prefix=                    config.feature.training_label_file_prefix,
#                         training_feature_fragment_num=                 keras_extractor._data_set.get_training_fragment_num(),
#                         testing_feature_file_prefix=                   config.feature.testing_feature_file_prefix,
#                         testing_name_file_prefix=                      config.feature.testing_name_file_prefix,
#                         testing_feature_fragment_num=                  keras_extractor._data_set.get_testing_fragment_num(),
#                         validation_feature_file_prefix=                config.feature.validation_feature_file_prefix,
#                         validation_label_file_prefix=                  config.feature.validation_label_file_prefix,
#                         validation_feature_fragment_num=               keras_extractor._data_set.get_validation_fragment_num(),
#                         batch_size=                                    config.data.batch_size,
#                         class_num=                                     config.data.class_num)
#
##model_arch = mi.inference_dense(25000, 10)
#
#model = KerasModel(model_name=              config.nn.model_name,
#                   data_set=	           feature_set,
#                   model_inference=         config.get_model(feature_dim=feature_set._feature_dim),
#                   test_batch_size=		config.nn.test_batch_size,
#                   n_iter=				config.nn.train_iter,
#                   model_arch_file=		config.nn.model_arch_file_name,
#                   model_weights_file=		config.nn.model_weights_file_name,
#                   model_save_path=		config.nn.model_save_path,
#                   prediction_save_file=	config.nn.prediction_save_file)
#
#model.train_model(save_best=True)
#model.set_model_weights(model._best_weight_file)
##model.set_model_weights('/home/liuzheng/competition/kaggle/distractedDrivers/inference_dense_weights_best_vLoss0.92203_vAcc0.743.h5')
#model.predict()
