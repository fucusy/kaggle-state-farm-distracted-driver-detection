# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:28:57 2016

@author: liuzheng
"""

#from model.models import RandomForestClassification 
class Project:

    # required, your project's absolute path, in other way, it's the absolute path for this file
    project_path = "/home/zhengliu/kaggle_drivers/"

    original_training_folder = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/train/'
    
    original_testing_folder = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/test/'

    # required, this path contain the train image in sub-folder, the there are ten sub-folders, c0, c1, c2, c3 .... c9
    train_img_folder_path =  project_path + 'imgs/trainAugmentation_'+colorMode + '_' + cropMode+'_'+'%03d'%(reSize[0])+'/'

    # required, this path contain imgs to be tested
    test_img_folder_path = project_path + 'imgs/testAugmentation_'+colorMode + '_' + cropMode+'_'+'%03d'%(reSize[0])+'/'


    # not required, a img path for you exercise program
    test_img_example_path = "/Users/fucus/Documents/buaa/projects/State_Farm_Distracted_Driver_Detection/data/imgs/train/c0/img_27327.jpg"

    # required, predict model
    # predict_model = model.RandomForestClassification()

    # required, result output path
#    result_output_path = "/home/liuzheng/competition/kaggle/distractedDrivers/rgb_entire_224/"

    # required, save cache or not
    save_cache = False

class Data:
    
    img_size = [3, 224, 224]
    
    augmentation_postfix = '_resize'
    
    mean_image_file_name = '/home/zhengliu/kaggle_drivers/imgs/meanImage.npy'
    
    driver_list_file = '/home/zhengliu/kaggle_drivers/driver_imgs_list.csv'
    
    image_crop = 'entire'# whether random crop images or not
    
    color_mode = 'rgb'# 'gray'

    fragment_size = 2048

    batch_size = 32
    
    class_num = 10
    
    validation_split = 0.2

class CNN:
    model_name = 'xavier_prelu_224'
    
    '''/path/to/json/vgg_self_exp1_keras_arch.json'''
    model_arch_file_name = '/home/zhengliu/kaggle_drivers/vgg_self_exp1_keras_arch.json'

    '''/path/to/h5/vgg16_weights.h5'''
    model_weights_file_name = '/home/zhengliu/kaggle_drivers/vgg_self_exp1_keras_weights_best_vLoss0.11026_vAcc0.969.h5'
    
    optimizer = 'adam'
    
    train_iter = 50
    
    test_batch_size = 64

    '''path/to/save/model/'''
    model_save_path = '/home/zhengliu/kaggle_drivers/'

    '''path/to/prediction/and/file/name/'''
    prediction_save_file = '/home/zhengliu/kaggle_drivers/prediction.csv'
    
    feature_save_path = '/home/zhengliu/kaggle_drivers/'
    
    #
#    keras_train_weight = "%s/CNN/vgg_try_karea/cache/model_weights_2_vgg_16_2x20.h5" % project_path
#
#
#    # keras structure files
#    keras_structure_files = ""
#    vgg_weight_file_path = "/home/chenqiang/kaggle_driver_data/vgg16_weights.h5"
#
#    fine_tuning_vgg_weight_file_path = ""