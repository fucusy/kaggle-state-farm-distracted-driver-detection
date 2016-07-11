import sys
import config
import pickle
from preprocess.resize import resize_image_main
from preprocess.argument import argument_main
from tool.data_tools import compute_mean_image

from tool.model_tools import KerasModel
from tool.keras_tool import load_data
from model.cnn_model import VGG_16


import logging

if __name__ == '__main__':
    level = logging.DEBUG
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    train_data, validation_data, test_data = load_data(config.Project.train_img_folder_path, config.Project.test_img_folder_path)

    logging.info("train data image count %s" % train_data.count())
    logging.info("validation data image count %s" % validation_data.count())

    cnn_model = VGG_16()
    model = KerasModel(cnn_model=cnn_model)
    pickle_file = config.Project.project_path + "/cache/vgg_feature.pickle"
 

    model.predict_model(validation_data)
    prediction = model._prediction
    with open(pickle_file, 'wb') as f:
        pickle.dump(prediction, f)
    model.predict_model(train_data)
    model.predict_model(test_data)

    prediction = model._prediction
    with open(pickle_file, 'wb') as f:
        pickle.dump(prediction, f)
