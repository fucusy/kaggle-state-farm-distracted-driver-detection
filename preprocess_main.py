import sys
import config
from preprocess.resize import resize_image_main
from preprocess.argument import argument_main
from preprocess.resize import add_padding_main
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

    # resize begin
    path_list = ["/home/chenqiang/kaggle_driver_data/imgs/test_244_244", "/home/chenqiang/kaggle_driver_data/imgs/train_244_244"]

    img_size = (122, 122)
    resize_image_main(path_list, img_size)
    # resize end

    path_list = ["/home/chenqiang/kaggle_driver_data/imgs/test_244_244_122_122", "/home/chenqiang/kaggle_driver_data/imgs/train_244_244_122_122"]
    img_size = (244, 244)
    add_padding_main(path_list, img_size)
