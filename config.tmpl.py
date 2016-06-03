__author__ = 'fucus'

from model.models import RandomForestClassification 
class Project:

    # required, your project's absolute path, in other way, it's the absolute path for this file
    project_path = "/Users/fucus/Documents/buaa/projects/State_Farm_Distracted_Driver_Detection/code/kaggle_distractedDrivers"

    driver_img_list_path = ""

    # required, this path contain the train image in sub-folder, the there are ten sub-folders, c0, c1, c2, c3 .... c9
    train_img_folder_path = "/Users/fucus/Documents/buaa/projects/State_Farm_Distracted_Driver_Detection/data/imgs/train"

    # required, this path contain imgs to be tested
    test_img_folder_path = "/Users/fucus/Documents/buaa/projects/State_Farm_Distracted_Driver_Detection/data/imgs/test"


    # not required, a img path for you exercise program
    test_img_example_path = "/Users/fucus/Documents/buaa/projects/State_Farm_Distracted_Driver_Detection/data/imgs/train/c0/img_27327.jpg"

    # required, predict model
    predict_model = RandomForestClassification()

    # required, result output path
    result_output_path = "./result/"

    # required, save cache or not
    save_cache = False


class CNN:
    #
    keras_train_weight = "%s/CNN/vgg_try_karea/cache/model_weights_2_vgg_16_2x20.h5" % Project.project_path


    # keras structure files
    keras_structure_files = ""
    vgg_weight_file_path = "/home/chenqiang/kaggle_driver_data/vgg16_weights.h5"

    fine_tuning_vgg_weight_file_path = ""

    model_name = 'vgg_self'

    '''/path/to/json/vgg_self_exp1_keras_arch.json'''
    model_arch_file_name = '/home/liuzheng/competition/kaggle/distractedDrivers/vgg_self_exp1_keras_arch.json'

    '''/path/to/h5/vgg16_weights.h5'''
    model_weights_file_name = '/home/liuzheng/competition/kaggle/distractedDrivers/vgg_self_exp1_keras_weights_best_vLoss0.11026_vAcc0.969.h5'

    optimizer = 'sgd'

    train_iter = 50

    test_batch_size = 64

    '''path/to/save/model/'''
    model_save_path = '/home/liuzheng/competition/kaggle/distractedDrivers/'

    '''path/to/prediction/and/file/name/'''
    prediction_save_file = '/home/liuzheng/competition/kaggle/distractedDrivers/prediction.csv'

    feature_save_path = '/home/liuzheng/competition/kaggle/distractedDrivers/'

class Data:

    img_size = [3, 224, 224]

    mean_image_file_name = '/home/liuzheng/competition/kaggle/distractedDrivers/imgs/meanImage.npy'

    image_crop = 'entire'# whether random crop images or not

    color_mode = 'rgb'# 'gray'

    fragment_size = 2048

    batch_size = 32

    class_num = 10

    validation_split = 0.2
