#encoding=utf8

from sklearn.ensemble import RandomForestClassifier
from model.base_model import Model
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import classification_report
import logging



class RandomForestClassification(Model):

    def __init__(self):
        Model.__init__(self)
        self.model = RandomForestClassifier(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 1)

    def fit(self, x_train, y_train, need_transform_label=False):
        train_pred = cross_val_predict(self.model, x_train, y_train, cv=3)
        report = classification_report(y_train, train_pred)
        logging.info("the cross validation report:\n %s" % report)
        self.model.fit(x_train, y_train)

    def predict(self, x_test, need_transform_label=False):
        return self.model.predict(x_test)