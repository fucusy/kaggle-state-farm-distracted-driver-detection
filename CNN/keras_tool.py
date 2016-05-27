__author__ = 'fucus'

from keras.models import model_from_json
import os
import numpy as np
from scipy.misc import imread, imresize




def load_model_from_file(network_structure_path, weight_path, loss='categorical_crossentropy', optimizer='adagrad'):
    """
    load the keras model, from your saved model

    :param network_structure_path:
    :param weight_path:
    :param loss:
    :param optimizer:
    :return:
    """
    model = model_from_json(network_structure_path)
    model.load_weights(weight_path)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


def load_test_image_path_list(path):
    """

    :param path: the test image folder
    :return:
    """
    list_path = os.listdir(path)
    result = [x for x in list_path if x.endswith("jpg")]
    return result


####  preprocess function

def resize_and_mean(image, size=(224, 224), mean=(103.939, 116.779, 123.68)):
    """
    :param image:
    :param size:
    :param mean:
    :return:
    """
    img_resized = imresize(image, size)
    for c in range(3):
        img_resized[c, :, :] = img_resized[c, :, :] - mean[c]
    return img_resized

def load_test_data_set(test_image_path):
    test_image_list = load_test_image_path_list(test_image_path)
    return DataSet(test_image_list)



class DataSet(object):

    def __init__(self,
               images_path_list):
        """

        :param images_path_list: numpy.array
        :param labels: numpy.array
        :return:
s        """

        self._num_examples = images_path_list.shape[0]
        self._images_path = images_path_list
        self._epochs_completed = 0
        self._index_in_epoch = 0


    @property
    def image_path_list(self):
        return self._images_path

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

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

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        end = min(self._index_in_epoch + batch_size, self._num_examples)
        self._index_in_epoch = end
        image_list = self.image_path_list_to_image_pic_list(self._images_path[start:end])
        image_path = [os.path.basename(x) for x in self._images_path[start:end]]
        return image_list, image_path
