import sys
import config


from preprocess.resize import resize_image_main
from preprocess.argument import argument_main
from tool.data_tools import compute_mean_image

from tool.model_tools import KerasModel
from tool.keras_tool import load_data
import model.cnn_model as model_factory


import logging

if __name__ == '__main__':
    level = logging.DEBUG
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    '''=====================================Data resize=================================================='''

    # argument_main()
    # resize_image_main()

    '''====================================Train and test================================================'''

    train_data, validation_data, test_data = load_data(config.Project.train_img_folder_path, config.Project.test_img_folder_path)

    logging.info("train data image count %s" % train_data.count())
    logging.info("validation data image count %s" % validation_data.count())

    cnn_model = getattr(model_factory, config.CNN.model_name)(weights_path=config.CNN.keras_train_weight, lr=config.CNN.lr)

    model = KerasModel(cnn_model=cnn_model)
    model.train_model(train_data, validation_data, save_best=True)
    model.predict_model(test_data)
