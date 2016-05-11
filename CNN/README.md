# kaggle_distractedDrivers
A project built for kaggle competition [distracted drivers detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection).

### trainModel.py
The wrapper of training models.

### trainCNN_KERAS.py
The file defining functions of 1) inferring network architecture by using the layers in KERAS toolbox, 2) defining optimizer and training settings.

### testModel.py
The wrapper of testing models.

### testCNN_KERAS.py
The file defining functions of testing settings.

### DataReader_KERAS.py
The file defining functions of reading data from the disc and return training or testing data for KERAS toolbox.

### DataAugmentation.py
The file defining functions for data augmentation, including 'mirror', 'rotate', 'resize' and so on.
