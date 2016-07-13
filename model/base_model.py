from sklearn import grid_search
from sklearn.metrics import make_scorer, log_loss
from keras.utils.np_utils import to_categorical
import numpy as np

class Model(object):

    def __init__(self):
        self.model = None

    def fit(self, x_train, y_train, x_test):
        pass

    def predict(self, x_test):
        pass

    def grid_search_fit_(self, clf, param_grid, x_train, y_train, cv=2):
        model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, verbose=20, scoring=make_scorer(log_loss))
        print("y_train = %s" % y_train)
        model.fit(x_train, y_train)
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        # f = "%s/%s/model.dump.pickle" % (config.project.project_path, sys.argv[1])
        # pickle.dump(model, f)
        return model        
