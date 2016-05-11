# kaggle Distracted Drivers
A project built for kaggle competition [distracted drivers detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection).

# 环境配置

## environment

* python 3.5
* python library:
    in the requirements.md
    to install pip packages use : `pip3.5 install -r pip_packages.txt`
* Linux or unix System, may be windows
    
## how to run
1. move `config.tmpl.py` to `config.py`, you can use *unix command line `mv config.tmpl.py config.py`, and update the variable in this file 



#Idea



#实验结果记录

| submit date | name | offline |          | online  |   compare  |feature                  | model                   | other trick                                   | comments |
| ---------- |-------- | --------|---------|---------|------------|-------------------------|-------------------------|-----------------------------------------------|----------|
| 2016-05-11  | chenqiang | 0.47447 |  0       | 0.47571 |    0       |  query_in_title etc     | RandomForestRegressor   | remove stop words                             | example |
