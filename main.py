__author__ = 'fucus'


import logging
import sys
import datetime
from config import Project
from feature.utility import load_train_data
from feature.utility import load_test_data
from feature.utility import extract_feature
from tool.file import generate_result_file


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

    start_time = datetime.datetime.now()
    logging.info('start program---------------------')

    logging.info("loading train data now")
    train_x_dic, train_y = load_train_data(Project.train_img_folder_path)
    logging.info("loading train data end")

    #
    # logging.info("loading test data now")
    # test_x_dic = load_test_data(Project.test_img_folder_path)
    # logging.info("loading test data end")

    logging.info("extract train data feature now")
    train_x_feature = extract_feature(train_x_dic['img'])
    logging.info("extract train data feature done")

    #
    # logging.info("extract train test feature now")
    # test_x_feature = extract_feature(test_x_dic['img'])
    # logging.info("extract train test feature done")


    logging.info("start to train the model")
    Project.predict_model.fit(x_train=train_x_feature, y_train=train_y)
    logging.info("train the model done")

    logging.info("start predict test data")
    predict_result = Project.predict_model.predict(test_x_feature)
    logging.info("predict test data done")

    logging.info("start to generate the final file used to submit")
    generate_result_file(test_x_dic['name'], predict_result)
    logging.info("generated the final file used to submit")

    end_time = datetime.datetime.now()
    logging.info('total running time: %.2f second' % (end_time - start_time).seconds)
    logging.info('end program---------------------')