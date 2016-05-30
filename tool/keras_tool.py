__author__ = 'fucus'

import sys
sys.path.append('../')
import config
from keras.models import model_from_json
import os
import numpy as np
from scipy.misc import imread, imresize
from keras.utils.np_utils import to_categorical



def save_model(model, cross=''):
    json_string = model.to_json()
    cache_path = "%s/cache" % config.Project.project_path
    if not os.path.isdir(cache_path):
        os.mkdir(cache_path)
    model_filename = 'architecture_' + cross + '.json'
    weights_filename = 'model_weights_' + cross + '.h5'
    open(os.path.join(cache_path, model_filename), 'w').write(json_string)
    model.save_weights(os.path.join(cache_path, weights_filename), overwrite=True)

def load_model_from_file(cross, loss='categorical_crossentropy', optimizer='adagrad'):
    """
    load the keras model, from your saved model

    :param network_structure_path:
    :param weight_path:
    :param loss:
    :param optimizer:
    :return:
    """
    model_filename = 'architecture_' + cross + '.json'
    weights_filename = 'model_weights_' + cross + '.h5'
    cache_path = "%s/cache" % config.Project.project_path
    model = model_from_json(open(os.path.join(cache_path, model_filename)).read())
    model.load_weights(os.path.join(cache_path, weights_filename))
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def load_test_image_path_list(path):
    """

    :param path: the test image folder
    :return:
    """
    list_path = os.listdir(path)
    result = ["%s/%s" % (path, x) for x in list_path if x.endswith("jpg")]
    return np.array(result)

def load_train_image_path_list_and_label(path):
    label_list = []
    result_list = []
    for x in range(10):
        sub_folder = 'c%d' % x
        path = "%s/%s" % (config.Project.train_img_folder_path, sub_folder)
        result = load_test_image_path_list(path)
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

def load_test_data_set(test_image_path):
    test_image_list = load_test_image_path_list(test_image_path)
    return DataSet(test_image_list)

def load_train_data_set(path):
    image_list, image_label = load_train_image_path_list_and_label(path)
    return DataSet(image_list, image_label)

class DataSet(object):
    def __init__(self,
               images_path_list, image_label_list=None):
        """

        :param images_path_list: numpy.array
        :param labels: numpy.array
        :return:
        """

        self._num_examples = images_path_list.shape[0]
        self._images_path = images_path_list
        self._epochs_completed = 0
        self._index_in_epoch = 0
        if image_label_list is not None:
            random = 2016
            np.random.seed(random)
            permut = np.random.permutation(len(images_path_list))
            self._images_path = images_path_list[permut]
            self._images_label = image_label_list[permut]

    

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

    def set_index_in_epoch(self, index):
        self._index_in_epoch = index
    def reset_index(self):
        self.set_index_in_epoch(0)

    def image_path_list_to_image_pic_list(self, image_path_list, preprocess_func=resize_and_mean):
        image_pic_list = []
        for image_path in image_path_list:
            im = imread(image_path)
            if preprocess_func is not None:
                im = preprocess_func(im)
            image_pic_list.append(im)
        return np.array(image_pic_list)

    def have_next(self):
        return self._index_in_epoch < self._num_examples
    def load_all_image(self, need_label=False):
        index_in_epoch = self._index_in_epoch
        self.reset_index()
        result = self.next_batch(self.num_examples, need_label)
        self.set_index_in_epoch(index_in_epoch)
        return result

    def next_batch(self, batch_size, need_label=False):
        start = self._index_in_epoch
        end = min(self._index_in_epoch + batch_size, self._num_examples)
        self._index_in_epoch = end
        image_list = self.image_path_list_to_image_pic_list(self._images_path[start:end])
        image_path = [os.path.basename(x) for x in self._images_path[start:end]]
        if need_label and self._images_label is not None:
            return image_list, self._images_label[start:end], image_path
        else:
            return image_list, image_path

if __name__ == '__main__':
    data_set = load_train_data_set(config.Project.train_img_folder_path)
    print(data_set.image_label_list)
    while data_set.have_next():
        img_list, img_label, _ = data_set.next_batch(2, need_label=True)
        print(img_list)
        print(img_label)
        break