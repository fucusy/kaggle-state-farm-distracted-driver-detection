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
    predict_model = model.RandomForestClassification()

    # required, result output path
    result_output_path = "./result/"

    # required, save cache or not
    save_cache = False


class CNN:
    #
    keras_train_weight = "%s/CNN/vgg_try_karea/cache/model_weights_2_vgg_16_2x20.h5" % project_path


    # keras structure files
    keras_structure_files = ""
