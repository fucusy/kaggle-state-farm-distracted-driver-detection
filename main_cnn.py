import sys
import config


from preprocess.resize import resize_image
from tool.data_tools import compute_mean_image

from tool.model_tools import KerasModel
from tool.keras_tool import load_data

import logging

if __name__ == '__main__':
    LEVELS = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}
    level = logging.INFO
    if len(sys.argv) >= 2:
        level_name = sys.argv[1]
        level = LEVELS.get(level_name, logging.INFO)
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    '''=====================================Data resize=================================================='''

    resize_image()
    compute_mean_image()

    '''====================================Train and test================================================'''

    train_data, validation_data, test_data = load_data(config.Project.train_img_folder_path, config.Project.test_img_folder_path)

    model = KerasModel(cnn_model=config.CNN.cnn_model)
    model.train_model(train_data, validation_data, save_best=True)
    model.predict_model(test_data)