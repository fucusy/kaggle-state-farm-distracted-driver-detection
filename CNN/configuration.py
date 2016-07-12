# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:28:57 2016

@author: liuzheng
"""
import model_inference as mi
#from model.models import RandomForestClassification 
class Config_Data(object):
    def __init__(self,
                 img_size =                 [3, 224, 224],
                 padded_img_size =          [224, 224],
                 crop_num =                 5,
                 fragment_size =            4096,
                 batch_size =               32,
                 class_num =                10,
                 validation_split =         0.2,
                 train_sample_mode =        'equal',
                 project_path =     '',#"/home/liuzheng/competition/kaggle/distractedDrivers/%s_%s_%d/mean_image.npy"%(color_mode, image_crop, img_size[1])
                 driver_list_file =         '/home/liuzheng/competition/kaggle/distractedDrivers/driver_imgs_list.csv'):
        
        
        self.crop_num = crop_num
        if crop_num > 1:
            self.augmentation_postfix = []
            for i in range(crop_num):
                self.augmentation_postfix.append('_crop%02d'%(i))
            self.image_crop = 'croped'
        else:
            self.augmentation_postfix = ['_resize']
            self.image_crop = 'entire'
        
        self.img_size = img_size
        if img_size[0] == 3:
            self.color_mode = 'rgb'
        elif img_size[0] == 1:
            self.color_mode = 'gray'
        
        self.padded_img_size = padded_img_size
        self.fragment_size = fragment_size
        self.batch_size = batch_size
        self.class_num = class_num
        self.validation_split = validation_split
        self.train_sample_mode = train_sample_mode
        
        self.mean_image_file_name = project_path + '%s_%s_%d/mean_image.npy'%(self.color_mode, self.image_crop, img_size[1])
        self.driver_list_file = driver_list_file
    
    

class Config_Feature(object):
    def __init__(self,
                 feature_layer_index =                  31,
                 feature_folder =                       '',
                 feature_fragment_size =                4096,
                 need_train_feature_fragment =          False,
                 train_feature_folder_path =            '',
                 test_feature_folder_path =             ''):
#                 training_feature_file_prefix =         '',#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggTrainingFeature'
#                 training_label_file_prefix =           '',
#                 testing_feature_file_prefix =          '',#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggTestingImageFeature'
#                 testing_name_file_prefix =             '',
#                 validation_feature_file_prefix =       '',#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggValidationFeature'
#                 validation_label_file_prefix =         ''):
#        self.feature_dim = feature_dim
        
        self.feature_layer_index = feature_layer_index
        self.feature_folder = feature_folder
        self.feature_fragment_size = feature_fragment_size
        self.need_train_feature_fragment = need_train_feature_fragment
        self.train_feature_folder_path = train_feature_folder_path
        self.test_feature_folder_path = test_feature_folder_path
#        self.training_feature_file_prefix = training_feature_file_prefix
#        self.training_label_file_prefix = training_label_file_prefix
##        self.training_feature_fragment_num = training_feature_fragment_num
#        
#        self.testing_feature_file_prefix = testing_feature_file_prefix
#        self.testing_name_file_prefix = testing_name_file_prefix
##        self.testing_feature_fragment_num = testing_feature_fragment_num
#        
#        self.validation_feature_file_prefix = validation_feature_file_prefix
#        self.validation_label_file_prefix = validation_label_file_prefix
#        self.validation_feature_fragment_num = validation_feature_fragment_num
    
class Config_Project(object):
    def __init__(self,
                 project_path =                 '',
                 original_training_folder =     '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/train/',
                 original_testing_folder =      '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/test/',
                 train_img_folder_path =        '',
                 test_img_folder_path =         ''):
        # required, your project's absolute path, in other way, it's the absolute path for this file
        self.project_path = project_path
    
        self.original_training_folder = original_training_folder
        
        self.original_testing_folder = original_testing_folder
    
        # required, this path contain the train image in sub-folder, the there are ten sub-folders, c0, c1, c2, c3 .... c9
        self.train_img_folder_path = train_img_folder_path
    
        # required, this path contain imgs to be tested
        self.test_img_folder_path = test_img_folder_path


class Config_NN(object):
    def __init__(self,
                 model_name =                               'vgg_std16_model',
                 model_arch_file_name =                     '',
                 model_weight_file_name =                   '',
                 optimizer =                                'adam',
                 train_iter =                               30,
                 test_batch_size =                          64,
                 model_save_path =                          '',
                 prediction_save_file =                     '',
                 img_size =                                 [3, 224, 224],
                 class_num =                                10):
        self.model_name = model_name#'vgg_std16_model'
        
        '''/path/to/json/vgg_self_exp1_keras_arch.json
        model_arch_file_name = '/home/liuzheng/competition/kaggle/distractedDrivers/vgg_self_exp1_keras_arch.json'
        '''
        self.model_arch_file_name = model_arch_file_name
        '''/path/to/h5/vgg16_weights.h5
        model_weights_file_name = '/home/liuzheng/competition/kaggle/distractedDrivers/vgg_self_exp1_keras_weights_best_vLoss0.11026_vAcc0.969.h5'
        '''
        self.model_weight_file_name = model_weight_file_name
        
        #vgg_std_weight_file = '/home/liuzheng/competition/kaggle/distractedDrivers/vgg16_weights.h5'
    
        self.optimizer = optimizer
        
        
                                            
        self.train_iter = train_iter
        
        self.test_batch_size = test_batch_size
    
        '''path/to/save/model/'''
        self.model_save_path = model_save_path
    
        '''path/to/prediction/and/file/name/'''
        self.prediction_save_file = prediction_save_file
        
#        self.feature_save_path = feature_save_path


class Config(object):
    def __init__(self,
                 project_path =                             '/home/liuzheng/competition/kaggle/distractedDrivers/',
                 original_training_folder =                 '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/train/',
                 original_testing_folder =                  '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/test/',
                 feature_train_test_folder_name =           '',
                 
                 img_size =                                 [3, 224, 224],
                 padded_img_size =                          [224, 224],
                 crop_num =                                 5,
                 fragment_size =                            4096,
                 batch_size =                               32,
                 class_num =                                10,
                 validation_split =                         0.2,
                 driver_list_file =                         '/home/liuzheng/competition/kaggle/distractedDrivers/driver_imgs_list.csv',
                 train_sample_mode=                         'equal',
                 #feature_dim =                              25088,
                 feature_layer_index =                      31,
                 feature_fragment_size =                    4096,
                 need_train_feature_fragment =              False,
#                 training_feature_file_prefix =             '',#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggTrainingFeature'
#                 training_label_file_prefix =               '',#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggTrainingLabel'
##                 training_feature_fragment_num =            9,
#                 testing_feature_file_prefix =              '',#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggTestingImageFeature'
#                 testing_name_file_prefix =                 '',#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggTestingImageName'
##                 testing_feature_fragment_num =             39,
#                 validation_feature_file_prefix =           '',#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggValidationFeature'
#                 validation_label_file_prefix =             '',#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggValidationLabel'
##                 validation_feature_fragment_num =          3,
                 
                 model_name =                               'vgg_std16_model',
                 model_arch_file_name =                     '',
                 model_weight_file_name =                   '',
                 optimizer =                                'adam',
                 train_iter =                               30,
                 test_batch_size =                          64):
        
        
        self.data = Config_Data( img_size =                 img_size,
                                 padded_img_size =          padded_img_size,
                                 crop_num =                 crop_num,
                                 fragment_size =            fragment_size,
                                 batch_size =               batch_size,
                                 class_num =                class_num,
                                 validation_split =         validation_split,
                                 train_sample_mode =        train_sample_mode,
                                 project_path =             project_path,#"/home/liuzheng/competition/kaggle/distractedDrivers/%s_%s_%d/mean_image.npy"%(color_mode, image_crop, img_size[1])
                                 driver_list_file =         driver_list_file)        
        
        self.project = Config_Project(project_path =                 project_path,
                                      original_training_folder =     original_training_folder,
                                      original_testing_folder =      original_testing_folder,
                                      train_img_folder_path =        project_path + '%s_%s_%d/train/'%(self.data.color_mode, self.data.image_crop, self.data.img_size[1]),
                                      test_img_folder_path =         project_path + '%s_%s_%d/test/'%(self.data.color_mode, self.data.image_crop, self.data.img_size[1]))
        
        feature_folder = project_path + '%s_%s_%d/%s_%d_features/'%(self.data.color_mode, self.data.image_crop, self.data.img_size[1], model_name, feature_layer_index)
        self.feature = Config_Feature(feature_layer_index =                  feature_layer_index,
                                      feature_folder =                       feature_folder,
                                      feature_fragment_size =                feature_fragment_size,
                                      need_train_feature_fragment =          need_train_feature_fragment,
                                      train_feature_folder_path =            project_path + feature_train_test_folder_name + '/train/',
                                      test_feature_folder_path =             project_path + feature_train_test_folder_name + '/test/')
#                                      training_feature_file_prefix =         feature_folder+training_feature_file_prefix,#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggTrainingFeature'
#                                      training_label_file_prefix =           feature_folder+training_label_file_prefix,#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggTrainingLabel'
##                                      training_feature_fragment_num =        training_feature_fragment_num,
#                                      testing_feature_file_prefix =          feature_folder+testing_feature_file_prefix,#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggTestingImageFeature'
#                                      testing_name_file_prefix =             feature_folder+testing_name_file_prefix,#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggTestingImageName'
##                                      testing_feature_fragment_num =         testing_feature_fragment_num,
#                                      validation_feature_file_prefix =       feature_folder+validation_feature_file_prefix,#'/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/vggValidationFeature'
#                                      validation_label_file_prefix =         feature_folder+validation_label_file_prefix)
        if feature_train_test_folder_name:
            model_save_path = project_path + feature_train_test_folder_name + '/'
        else:
            model_save_path = project_path + '%s_%s_%d/'%(self.data.color_mode, self.data.image_crop, self.data.img_size[1])
        prediction_save_file = model_save_path + '%s_prediction.csv'%(model_name)
        
        self.nn = Config_NN( model_name =                               model_name,
                             model_arch_file_name =                     model_arch_file_name,
                             model_weight_file_name =                   model_weight_file_name,
                             optimizer =                                optimizer,
                             train_iter =                               train_iter,
                             test_batch_size =                          test_batch_size,
                             model_save_path =                          model_save_path,
                             prediction_save_file =                     prediction_save_file,
                             img_size =                                 img_size,
                             class_num =                                class_num)

    def print_info(self, do_not_ask=False):
        confirm_flag = False
        print("model name: %s"%(self.nn.model_name))
        print("image size: %d %d %d"%(self.data.img_size[0], self.data.img_size[1], self.data.img_size[2]))
        print("image crop: %s"%(self.data.image_crop))
        print("fragment size: %d"%(self.data.fragment_size))
        print("batch size: %d"%(self.data.batch_size))
        print("training iter: %d"%(self.nn.train_iter))
        if not do_not_ask:
            got_input = raw_input("input y to continue, others to exit:")
            if got_input == 'y':
                confirm_flag = True
        else:
            confirm_flag = True
        return confirm_flag
    
    def get_model(self, feature_dim=0):
        if self.data.padded_img_size:
            input_shape = [self.data.img_size[0], self.data.padded_img_size[0], self.data.padded_img_size[1]]
        else:
            input_shape = self.data.img_size
        if self.nn.model_name == 'vgg_std16_model':
            model_arch = mi.vgg_std16_model(input_shape=   input_shape,
                                           class_num=      self.data.class_num,
                                           optimizer=      self.nn.optimizer,
                                           weights_file=   self.nn.model_weight_file_name)
        elif self.nn.model_name == 'inference_less_filter':
            model_arch = mi.inference_less_filter(input_shape=    input_shape,
                                                  class_num=      self.data.class_num,
                                                  optimizer=      self.nn.optimizer,
                                                  weights_file=   self.nn.model_weight_file_name)
        elif self.nn.model_name == 'inference_xavier_prelu_224':
            model_arch = mi.inference_xavier_prelu_224(input_shape=    input_shape,
                                                       class_num=      self.data.class_num,
                                                       optimizer=      self.nn.optimizer,
                                                       weights_file=   self.nn.model_weight_file_name)
        elif self.nn.model_name == 'inference_64':
            model_arch = mi.inference_64(input_shape=       input_shape,
                                            class_num=      self.data.class_num,
                                            optimizer=      self.nn.optimizer,
                                            weights_file=   self.nn.model_weight_file_name)
        elif self.nn.model_name == 'inference_dense' and feature_dim:
            model_arch = mi.inference_dense(input_dim=      feature_dim,
                                            class_num=      self.data.class_num,
                                            optimizer=      self.nn.optimizer,
                                            weights_file=   self.nn.model_weight_file_name)
        elif self.nn.model_name == 'inference_xavier_prelu_112':
            model_arch = mi.inference_xavier_prelu_112(input_shape=    input_shape,
                                                       class_num=      self.data.class_num,
                                                       optimizer=      self.nn.optimizer,
                                                       weights_file=   self.nn.model_weight_file_name)
        elif self.nn.model_name == 'inference_32':
            model_arch = mi.inference_32(input_shape=       input_shape,
                                            class_num=      self.data.class_num,
                                            optimizer=      self.nn.optimizer,
                                            weights_file=   self.nn.model_weight_file_name)
        elif self.nn.model_name == 'vgg_std16_32':
            model_arch = mi.vgg_std16_32(input_shape=       input_shape,
                                            class_num=      self.data.class_num,
                                            optimizer=      self.nn.optimizer,
                                            weights_file=   self.nn.model_weight_file_name)
        elif self.nn.model_name == 'vgg_std16_extractor':
            model_arch = mi.vgg_std16_extractor(input_shape=      input_shape,
                                            class_num=          self.data.class_num,
                                            optimizer=          self.nn.optimizer,
                                            weights_file=       self.nn.model_weight_file_name)
        return model_arch





















