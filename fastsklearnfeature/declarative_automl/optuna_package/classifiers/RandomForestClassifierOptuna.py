from sklearn.ensemble import RandomForestClassifier
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.utils.class_weight import compute_sample_weight
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.wrapper.ClassifierWrapper import \
    calculate_class_weight

class RandomForestClassifierOptuna(RandomForestClassifier):

    def init_hyperparameters(self, trial, X, y, name_space=None):
        self.name = id_name('RandomForestClassifier_')

        if type(name_space) != type(None):
            self.name += '_' + name_space

        self.criterion = trial.suggest_categorical(self.name + "criterion", ["gini", "entropy"])
        self.max_features = trial.suggest_uniform(self.name + "max_features", 0., 1.)
        self.max_depth = None
        self.min_samples_split = trial.suggest_int(self.name + "min_samples_split", 2, 20, log=False)
        self.min_samples_leaf = trial.suggest_int(self.name + "min_samples_leaf", 1, 20, log=False)
        self.min_weight_fraction_leaf = 0.
        self.max_leaf_nodes = None
        self.min_impurity_decrease = 0.0
        self.bootstrap = trial.suggest_categorical(self.name + "bootstrap", [True, False])
        #self.classes_ = 2
        self.n_jobs = 1

        #hyperopt config
        #self.n_estimators = trial.suggest_int(self.name + "n_estimators", 10, 512, log=True)
        self.n_estimators = trial.suggest_int(self.name + "n_estimators", 10, 100, log=True)

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('RandomForestClassifier_')

        space_gen.generate_cat(self.name + "criterion", ["gini", "entropy"], "gini", depending_node=depending_node)
        space_gen.generate_number(self.name + "max_features", 0.5, depending_node=depending_node, low=0., high=1.)
        space_gen.generate_number(self.name + "min_samples_split", 2, depending_node=depending_node, low=2, high=20, is_float=False)
        space_gen.generate_number(self.name + "min_samples_leaf", 1, depending_node=depending_node, low=1, high=20, is_float=False)
        space_gen.generate_cat(self.name + "bootstrap", [True, False], True, depending_node=depending_node)

        space_gen.generate_number(self.name + "n_estimators", 100, depending_node=depending_node, low=10, high=100, is_float=False, is_log=True)

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