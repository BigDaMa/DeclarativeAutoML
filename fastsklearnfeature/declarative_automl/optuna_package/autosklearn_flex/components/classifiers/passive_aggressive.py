from sklearn.linear_model import PassiveAggressiveClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.utils.class_weight import compute_sample_weight
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.wrapper.ClassifierWrapper import \
    calculate_class_weight

class PassiveAggressiveOptuna(PassiveAggressiveClassifier):

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('PassiveAggressive_')

        space_gen.generate_number(self.name + "C", 1.0, depending_node=depending_node, low=1e-5, high=10, is_log=True)
        space_gen.generate_cat(self.name + "loss", ["hinge", "squared_hinge"], "hinge", depending_node=depending_node)
        space_gen.generate_number(self.name + "tol", 1e-4, depending_node=depending_node, low=1e-5, high=1e-1, is_log=True)
        space_gen.generate_cat(self.name + "average", ['False', 'True'], 'False', depending_node=depending_node)




