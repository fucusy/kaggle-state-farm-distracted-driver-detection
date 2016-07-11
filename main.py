__author__ = 'fucus'


import logging
import sys
import datetime
from config import Project
from feature.utility import load_train_validation_feature
from feature.utility import load_test_feature
from tool.file import generate_result_file
from feature.utility import load_cache
from feature.utility import load_feature_from_pickle
from feature.utility import save_cache
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss


hog_feature_cache = {}
lbp_feature_cache = {}
cache_path = "%s/cache" % Project.project_path

if __name__ == '__main__':
    level = logging.DEBUG
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    train_num = -1
    test_num = 100

    start_time = datetime.datetime.now()
    logging.info('start program---------------------')
    logging.info("loading feature cache now")
    feature_list = []
    vgg_feature_path = "%s/%s" % (cache_path, "vgg_feature.pickle")
    logging.info("load vgg featuer from %s" % vgg_feature_path)
    vgg_feature = load_feature_from_pickle(vgg_feature_path)
    if vgg_feature is not None:
        feature_list.append(vgg_feature)
        key = vgg_feature.keys()[0]
        logging.info("vgg_feature[%s][0]=%s" % (key, vgg_feature[key][0]))
    else:
        logging.warning("fail to vgg_feature")

    hog_feature_cache, lbp_feature_cache = load_cache()
    

    logging.info("%s features in feature_list" % len(feature_list))

    logging.info("load feature cache end")
    logging.info("load train data feature now")
    train_path_list, train_x_feature, train_y, validation_path_list, validation_x, validation_y = load_train_validation_feature(Project.train_img_folder_path, hog_feature_cache, lbp_feature_cache, feature_list, train_num)
    logging.info("extract train data feature done")

    logging.info("start to train the model")

    logging.debug("len of train_x_feature = %d" % len(train_x_feature))
    logging.debug("len of train_x_feature[0] = %d" % len(train_x_feature[0]))

    logging.debug("len of train_y = %d" % len(train_y))

    Project.predict_model.fit(x_train=train_x_feature, y_train=train_y
                              , x_validation=validation_x, y_validation=validation_y)

    logging.info("train the model done")

    logging.info("load test feature now")
    test_img_names, test_x_feature = load_test_feature(Project.test_img_folder_path, hog_feature_cache, lbp_feature_cache, feature_list, test_num)

    logging.info("load test feature done")

    if Project.save_cache:
        logging.info("saving feature cache now")
        save_cache(hog_feature_cache, lbp_feature_cache)
        logging.info("save feature cache end")
    else:
        logging.info("skip saving the feature cache")

    del hog_feature_cache
    del lbp_feature_cache
    
    logging.info("start to do validation")

    validation_result = Project.predict_model.predict(validation_x) 
    report = classification_report(validation_result, validation_y)
    logging.info("the validation report:\n %s" % report)

    validation_pro = Project.predict_model.predict_proba(validation_x) 
    logloss_val =  log_loss(validation_y, validation_pro)

    logging.info("validation logloss is %.3f" % logloss_val)
    logging.info("done validation")

    logging.info("start predict test data")
    predict_result = Project.predict_model.predict_proba(test_x_feature)
    logging.info("predict test data done")

    logging.info("start to generate the final file used to submit")
    generate_result_file(test_img_names[:len(predict_result)], predict_result)
    logging.info("generated the final file used to submit")

    end_time = datetime.datetime.now()
    logging.info('total running time: %.2f second' % (end_time - start_time).seconds)
    logging.info('end program---------------------')
