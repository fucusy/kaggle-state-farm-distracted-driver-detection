__author__ = 'fucus'


import sys

# append project root to python lib search path, so that you can import config
sys.path.append("../../")
import config


from model.cnn_model import VGG_16
from tool.file import generate_result_file
from tool.keras_tool import load_train_data_set
from tool.keras_tool import to_categorical
from tool.keras_tool import load_test_data_set
import logging
import numpy as np


def train_predict(nb_epoch=10, weights_path=None):
    # Now it loads color image
    # input image dimensions
    batch_size = 64
    model = VGG_16(lr=1e-3, weights_path=weights_path)
    data_set = load_train_data_set(config.Project.train_img_folder_path)

    count = 0
    total_count = data_set.num_examples * nb_epoch
    img_list, img_label, _ = data_set.load_all_image(need_label=True)
    img_label_cate = to_categorical(img_label, 10)
    model.fit(img_list, img_label_cate, batch_size=batch_size,
                    nb_epoch=nb_epoch, verbose=1, shuffle=False
                    ,validation_split=0.15)

    print('end saving model............')
    model.save_weights(weights_path, overwrite=True)

    test_data_set = load_test_data_set(config.Project.test_img_folder_path)
    predict = []

    while test_data_set.have_next():
        img_list, _ = test_data_set.next_batch(128)
        result = model.predict(img_list)
        predict += list(result)
    predict = np.array(predict)
    generate_result_file(test_data_set.image_path_list[:len(predict)], predict)

if __name__ == '__main__':
    level = logging.DEBUG
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    train_predict(nb_epoch=5, weights_path=config.CNN.keras_train_weight)
