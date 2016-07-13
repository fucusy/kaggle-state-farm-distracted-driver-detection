__author__ = 'fucus'

import sys
sys.path.append('../')

import config
from keras.models import model_from_yaml
import os
import numpy as np
from scipy.misc import imread, imresize
import logging
from keras.utils.np_utils import to_categorical



def save_model(model, weight_path, structure_path=''):
    """
    save model to file system
    :param model, the model
    :param weight_path, the weight path file you want, required
    :param structure_path, the structure  path file you want, optional
    """
    model_string = model.to_yaml()
    if structure_path == '':
        structure_path = weight_path + ".yaml"
    open(structure_path, 'w').write(model_string)
    model.save_weights(weight_path, overwrite=True)

def load_model(weight_path, structure_path=''):
    """
    load the keras model, from your saved model

    :return: uncompile model
    """
    if structure_path == '':
        structure_path = weight_path + ".yaml"
    model = model_from_yaml(open(structure_path).read())
    model.load_weights(weight_path)
    return model

def load_image_path_list(path):
    """

    :param path: the test image folder
    :return:
    """
    list_path = os.listdir(path)
    result = ["%s/%s" % (path, x) for x in list_path if x.endswith("jpg")]
    return np.array(result)


def load_train_image_path_list_and_label(train_path):
    label_list = []
    result_list = []
    for x in range(10):
        sub_folder = 'c%d' % x
        path = "%s/%s" % (train_path, sub_folder)
        result = load_image_path_list(path)
        label_list += [x] * len(result)
        result_list += list(result)
    return np.array(result_list), np.array(label_list)

####  preprocess function

def resize_and_mean(image, size=(224, 224), mean=(103.939, 116.779, 123.68)):
    """
    :param image:
    :param size:
    :param mean:
    :return:
    """
    img_resized = imresize(image, size)
    img_resized = img_resized.transpose((2, 0, 1))

    for c in range(3):
        img_resized[c, :, :] = img_resized[c, :, :] - mean[c]
    return img_resized
def image_preprocess(image, mean):
    """

    :param image:
    :param mean: the mean img computed by  data_tool.compute_mean_image
    :return:
    """
    # image -= mean
    image = image.transpose((2, 0, 1))
    image = image.astype(float)
    # image /= 256.0
    return image

def load_test_data_set(test_image_path, for_cnn=True):
    test_image_list = load_image_path_list(test_image_path)
    return DataSet(test_image_list, for_cnn=for_cnn)


def load_data(train_folder, test_folder, for_cnn=True):
    """

    :param train_folder:
    :param test_folder:
    :return: three DataSet structure include train data, validation data, test data
    """
    test_data = load_test_data_set(test_folder, for_cnn=for_cnn)
    train_data, validation_data = load_train_validation_data_set(train_folder, for_cnn=for_cnn)

    return train_data, validation_data, test_data


def get_image_to_person(file_path=config.Project.driver_img_list_path):
    """
    :param file_path: the driver list .cvs file path
    :return: a image_id to person dictionary, image_id means the name remove suffix like '.jpg'
    """

    image_id_to_person = {}
    count = 0

    for line in open(file_path):

        # ignore first line
        if count == 0:
            count += 1
            continue

        count += 1
        split_line = line.rstrip('\n').split(',')
        if len(split_line) == 3:
            person = split_line[0]
            driver_type = split_line[1]
            image_id = split_line[2].split('.')[0]
            image_id_to_person[image_id] = person
    return image_id_to_person


def load_train_validation_data_set(path, validation_split=0.2, to_category=True, for_cnn=True):
    """
    param: validation_spilt, how many percent driver to be vlidation
    param: to_category, if it's true, the result image label will be a 10 length vector
    param: for_cnn, if it's true, the returned DataSet will do transpose, and sub mean RBG value, else returned DataSet do not

    return a tuple of dataset contain train data set and validation data set
    """
    if type(path) is list:
        logging.debug("train validation data from multi-directory %s" % ",".join(path))
        image_list = []
        image_label = []
        for p in path:
            image_list_tmp, image_label_tmp = load_train_image_path_list_and_label(p)
            image_list += list(image_list_tmp)
            image_label += list(image_label_tmp)

    else:
        logging.debug("train validation data from %s" % path)
        image_list, image_label = load_train_image_path_list_and_label(path)

    img_to_person = get_image_to_person()
    driver_list = sorted(set(img_to_person.values()))
    train_driver_index_end = int(len(driver_list) * (1-validation_split))
    train_driver_list = driver_list[:train_driver_index_end]
    validation_driver_list = driver_list[train_driver_index_end:]

    logging.info("load train data from driver %s" % ",".join(train_driver_list))
    logging.info("load validation data from driver %s" % ",".join(validation_driver_list))

    train_image_list = []
    train_image_label = []

    validation_image_list = []
    validation_image_label = []

    for i in range(len(image_list)):
        image_id = os.path.basename(image_list[i]).split('.')[0]

        if img_to_person[image_id] in train_driver_list:
            train_image_list.append(image_list[i])
            train_image_label.append(image_label[i])
        else:
            validation_image_list.append(image_list[i])
            validation_image_label.append(image_label[i])

    return DataSet(train_image_list, train_image_label, to_category, for_cnn=for_cnn), DataSet(validation_image_list, validation_image_label, to_category, for_cnn=for_cnn)



class DataSet(object):
    def __init__(self,
               images_path_list, image_label_list=None, to_category=True, for_cnn=True):
        """

        :param images_path_list: numpy.array or list
        :param labels: numpy.array or list
        :return:
        """
        if type(images_path_list) is list:
            images_path_list = np.array(images_path_list)

        if image_label_list is not None:
            self._one_images_label = np.array(image_label_list)
            if to_category:
                image_label_list = to_categorical(np.array(image_label_list), 10)
            else:
                image_label_list = np.array(image_label_list)

        self._num_examples = images_path_list.shape[0]
        self._images_path = images_path_list
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._for_cnn = for_cnn
        self.mean_img = imread(config.Data.mean_image_file_name)
        if image_label_list is not None:
            random = 2016
            np.random.seed(random)
            permut = np.random.permutation(len(images_path_list))
            self._images_path = images_path_list[permut]
            self._images_label = image_label_list[permut]
            self._one_images_label = self._one_images_label[permut]
    @property
    def image_path_list(self):
        return self._images_path

    @property
    def image_label_list(self):
        return self._images_label

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def get_image_label(self, to_cate=True):
        if to_cate:
            return self._images_label
        else:
            return self._one_images_label

    def count(self):
        return self._num_examples

    def set_index_in_epoch(self, index):
        self._index_in_epoch = index
    def reset_index(self):
        self.set_index_in_epoch(0)

    def image_path_list_to_image_pic_list(self, image_path_list):
        image_pic_list = []
        for image_path in image_path_list:
            im = imread(image_path)
            if self._for_cnn:
                im = image_preprocess(im, self.mean_img)
            image_pic_list.append(im)
        return np.array(image_pic_list)

    def have_next(self):
        return self._index_in_epoch < self._num_examples
    def load_all_image(self, need_label=False):
        index_in_epoch = self._index_in_epoch
        self.reset_index()
        result = self.next_fragment(self.num_examples, need_label)
        self.set_index_in_epoch(index_in_epoch)
        return result

    def next_fragment(self, fragment_size, need_label=False):
        start = self._index_in_epoch
        end = min(self._index_in_epoch + fragment_size, self._num_examples)
        self._index_in_epoch = end
        image_list = self.image_path_list_to_image_pic_list(self._images_path[start:end])
        image_path = [os.path.basename(x) for x in self._images_path[start:end]]
        if need_label and self._images_label is not None:
            return image_list, self._images_label[start:end], image_path
        else:
            return image_list, image_path

if __name__ == '__main__':
    level = logging.INFO
    FORMAT = '%(asctime)-12s[%(levelname)s] %(message)s'
    logging.basicConfig(level=level, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
    train, validation, test = load_data(config.Project.train_img_folder_path, config.Project.test_img_folder_path)
    print(train.image_label_list)
    while train.have_next():
        img_list, img_label, _ = train.next_fragment(2, need_label=True)
        print(img_list)
        print(img_label)
        break
