from sklearn.naive_bayes import MultinomialNB
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

class MultinomialNBOptuna(MultinomialNB):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('MultinomialNB_')
        #self.classes_ = np.unique(y.astype(int))

        self.alpha = trial.suggest_loguniform(self.name + "alpha", 1e-2, 100)
        self.fit_prior = trial.suggest_categorical(self.name + "fit_prior", [True, False])

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('MultinomialNB_')
        space_gen.generate_number(self.name + "alpha", 1, depending_node=depending_node, low=1e-2, high=100, is_log=True)
        space_gen.generate_cat(self.name + "fit_prior", [True, False], True, depending_node=depending_node)

    def set_weight(self, custom_weight):
        self.custom_weight = custom_weight

    def fit(self, X, y=None, sample_weight=None):
        if hasattr(self, 'custom_weight'):
            return super().fit(np.abs(X), y, sample_weight=compute_sample_weight(class_weight=self.custom_weight, y=y))
        else:
            return super().fit(np.abs(X), y)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        return super().partial_fit(np.abs(X), y, classes=classes, sample_weight=sample_weight)

    def predict_proba(self, X):
        return super().predict_proba(np.abs(X))

    def predict_log_proba(self, X):
        return super().predict_log_proba(np.abs(X))

    def predict(self, X):
        return super().predict(np.abs(X))