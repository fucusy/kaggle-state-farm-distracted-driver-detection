from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import  Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D
from keras.optimizers import SGD
from keras.models import model_from_json
from CNN.keras_tool import *

img_rows, img_cols, color_type = 224, 224, 3

def save_model(model, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    model_filename = 'architecture_' + cross + '.json'
    weights_filename = 'model_weights_' + cross + '.h5'
    open(os.path.join('cache', model_filename), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weights_filename), overwrite=True)


def read_model(cross=''):
    model_filename = 'architecture_'+ cross + '.json'
    weights_filename = 'model_weights_' + cross + '.h5'
    model = model_from_json(open(os.path.join('cache', model_filename)).read())
    model.load_weights(os.path.join('cache', weights_filename))
    return model

def VGG_16(weights_path=None):
    # standard VGG16 network architecture
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
