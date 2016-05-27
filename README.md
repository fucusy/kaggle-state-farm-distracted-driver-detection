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
1. move `config.tmpl.py` to `config.py`, you can use *unix command line `mv config.tmpl.py config.py`, and update the variable in this file 
2. run the `main.py` by `python3.5 main.py`

## result

1. in the result folder, you can see some file end with `.csv`
2. the cache folder, you can see cache file

#Idea



#实验结果记录

| submit date | name      | offlogloss | off f1-score    | online  logloss |   compare   |feature                  | model   | other trick                                   | comments                           |
| ----------  |--------   | ---        |----             |---------        | ------------|-------------------------|---------|-----------------------------------------------|----------                          |
| 2016-05-13  | liu zheng |   -        | -               |    4.47070      |             |   6 conv layers cnn     | cnn     | mirror,rotate,resize 64x64                    | what a shame....                   |
| 2016-05-15  | chenqiang |   -        |  -              |     14.*        |   *_*       |   9600 hog feature      | forest  | no                                            | it must be over-fitting            |
| 2016-05-15  | chenqiang |   -        |  -              |     2.30259     |   *_*       |    all 0.1              |         | no                                            | base line                          |
| 2016-05-19  | chenqiang |   -        |  0.9            |     1.66477     |             |   9600 hog feature      | forest  | forest with probability                       | still have a huge space to improve |

