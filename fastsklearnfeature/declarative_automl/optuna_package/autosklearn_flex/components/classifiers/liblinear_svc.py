from sklearn.svm import LinearSVC
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.utils.class_weight import compute_sample_weight
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.wrapper.ClassifierWrapper import \
    calculate_class_weight

class LinearSVCOptuna(LinearSVC):

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('LinearSVC_')
        space_gen.generate_cat(self.name + "penalty", ["l1", "l2"], "l2", depending_node=depending_node)
        space_gen.generate_cat(self.name + "loss", ["hinge", "squared_hinge"], "squared_hinge", depending_node=depending_node)
        space_gen.generate_number(self.name + "tol", 1e-4, depending_node=depending_node, low=1e-5, high=1e-1, is_log=True)
        space_gen.generate_number(self.name + "C", 1.0, depending_node=depending_node, low=0.03125, high=32768, is_log=True)

