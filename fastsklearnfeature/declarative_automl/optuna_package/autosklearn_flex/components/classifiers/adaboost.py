from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.wrapper.ClassifierWrapper import \
    calculate_class_weight



class AdaBoostClassifierOptuna(AdaBoostClassifier):

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('AdaBoostClassifier_')
        space_gen.generate_number(self.name + "n_estimators", 50, depending_node=depending_node, low=50, high=500, is_float=False)
        space_gen.generate_number(self.name + "learning_rate", 0.1, depending_node=depending_node, low=0.01, high=2, is_log=True)
        space_gen.generate_cat(self.name + "algorithm", ["SAMME.R", "SAMME"], "SAMME.R", depending_node=depending_node)
        space_gen.generate_number(self.name + "max_depth", 1, depending_node=depending_node, low=1, high=10, is_float=False)
