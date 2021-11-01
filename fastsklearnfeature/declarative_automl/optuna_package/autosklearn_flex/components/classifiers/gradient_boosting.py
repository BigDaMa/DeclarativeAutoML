from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class HistGradientBoostingClassifierOptuna(HistGradientBoostingClassifier):

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('HistGradientBoostingClassifier_')

        space_gen.generate_number(self.name + "learning_rate", 0.1, depending_node=depending_node, low=0.01, high=1, is_log=True)
        space_gen.generate_number(self.name + "min_samples_leaf", 20, depending_node=depending_node, low=1, high=200, is_float=False, is_log=True)
        space_gen.generate_number(self.name + "max_leaf_nodes", 31, depending_node=depending_node, low=3, high=2047, is_float=False, is_log=True)
        space_gen.generate_number(self.name + "l2_regularization", 1E-10, depending_node=depending_node, low=1E-10, high=1, is_log=True)
        space_gen.generate_cat(self.name + "early_stop", ["off", "train", "valid"], "off", depending_node=depending_node)
        space_gen.generate_number(self.name + "n_iter_no_change", 10, depending_node=depending_node, low=1, high=20, is_float=False)
        space_gen.generate_number(self.name + "validation_fraction", 0.1, depending_node=depending_node, low=0.01, high=0.4)

