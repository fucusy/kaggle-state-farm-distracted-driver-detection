# kaggle Distracted Drivers
A project built for kaggle competition [distracted drivers detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection).

# 环境配置

## environment

* python 3.5
* python library:
    in the requirements.md
    to install pip packages use : `pip3.5 install -r pip_packages.txt`
    install tenserflow(tensorflow==0.8.0) from [here](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#pip-installation)
* Linux or unix System, may be windows
    
## how to run

### run machine learning method  

1. move `config.tmpl.py` to `config.py`, you can use *unix command line `mv config.tmpl.py config.py`, and update the variable in this file 
2. run the `main.py` by `python3.5 main.py`

### run cnn method vgg_try_karea network
1. move `config.tmpl.py` to `config.py`, you can use *unix command line `mv config.tmpl.py config.py`, and update the variable in this file
2. change your dir to `CNN/vgg_try_karea`, then train model by running `python3.5 train_model.py`, evaluate model by running `python3.5 evaluate_model.py`
, predict test set by running `python3.5 predict_model.py`


## result

1. in the result folder, you can see some file end with `.csv`
2. the cache folder, you can see cache file

#Idea



#实验结果记录

| submit date | name      | of-los | on-los |feature                  | model   | other trick                                   | comments                           |
| ----------  |--------   | ---    |------  |-------------------------|---------|-----------------------------------------------|----------                          |
| 2016-05-13  | liu zheng |   -    | 4.4707 |   6 conv layers cnn     | cnn     | mirror,rotate,resize 64x64                    | what a shame....                   |
| 2016-05-15  | chenqiang |   -    | 14.*   |   9600 hog feature      | forest  | no                                            | it must be over-fitting            |
| 2016-05-15  | chenqiang |   -    | 2.3025 |    all 0.1              |         | no                                            | base line                          |
| 2016-05-19  | chenqiang |   -    | 1.6647 |   9600 hog feature      | forest  | forest with probability                       | still have a huge space to improve |
| 2016-06-01  | chenqiang |   1.37 | 1.59   |   vgg fine-tuning       | cnn     | replace last layer 1000 node, to 10 node.     | still have a huge space to improve |
| 2016-06-01  | chenqiang |   1.09 | 1.34   |   vgg fine-tuning       | cnn     | replace last layer 1000 node, to 10 node.     | still have a huge space to improve |
| 2016-06-01  | chenqiang |   0.10 | 1.26   |   vgg fine-tuning       | cnn     | strange, only 2 epoch                         | still have a huge space to improve |
| 2016-06-01  | chenqiang |   0.03 | 1.41   |   vgg fine-tuning       | cnn     | strange, 12 epoch                             | over-fitting                       |
| 2016-06-01  | chenqiang |   0.9  | 1.28   |   vgg fine-tuning       | cnn-vgg | add one softmax  layer to vgg                 |                                    |
  
