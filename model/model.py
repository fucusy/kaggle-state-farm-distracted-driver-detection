#encoding=utf8

from sklearn.ensemble import RandomForestClassifier
from model.base_model import Model

class RandomForestClassification(Model):

    def __init__(self):
        Model.__init__(self)
        self.model = RandomForestClassifier(n_estimators = 500, n_jobs = 3, random_state = 2016, verbose = 1)

    def fit(self, x_train, y_train, need_transform_label=False):
        self.model.fit(x_train, y_train)

    def predict(self, x_test, need_transform_label=False):
        return self.model.predict(x_test)