from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.wrapper.ClassifierWrapper import \
    calculate_class_weight


class AdaBoostClassifierOptuna(AdaBoostClassifier):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('AdaBoostClassifier_')
        self.n_estimators = trial.suggest_int(self.name + "n_estimators", 50, 500, log=False)
        self.learning_rate = trial.suggest_loguniform(self.name + "learning_rate", 0.01, 2)
        self.algorithm = trial.suggest_categorical(self.name + "algorithm", ["SAMME.R", "SAMME"])
        self.max_depth = trial.suggest_int(self.name + "max_depth", 1, 10, log=False)
        self.base_estimator = DecisionTreeClassifier(max_depth=self.max_depth)
        self.classes_ = np.unique(y.astype(int))

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('AdaBoostClassifier_')
        space_gen.generate_number(self.name + "n_estimators", 50, depending_node=depending_node, low=50, high=500, is_float=False)
        space_gen.generate_number(self.name + "learning_rate", 0.1, depending_node=depending_node, low=0.01, high=2, is_log=True)
        space_gen.generate_cat(self.name + "algorithm", ["SAMME.R", "SAMME"], "SAMME.R", depending_node=depending_node)
        space_gen.generate_number(self.name + "max_depth", 1, depending_node=depending_node, low=1, high=10, is_float=False)

    def set_weight(self, custom_weight):
        self.custom_weight = custom_weight

    def fit(self, X, y=None, sample_weight=None):
        if hasattr(self, 'custom_weight'):
            class_weight = 'balanced'
            if self.custom_weight != 0.5:
                class_weight = calculate_class_weight(y, self.custom_weight)
            return super().fit(X, y, sample_weight=compute_sample_weight(class_weight=class_weight, y=y))
        else:
            return super().fit(X, y)

