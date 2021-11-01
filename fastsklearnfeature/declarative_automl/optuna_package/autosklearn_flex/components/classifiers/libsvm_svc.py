from sklearn.svm import SVC
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.utils.class_weight import compute_sample_weight
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.wrapper.ClassifierWrapper import \
    calculate_class_weight

class SVCOptuna(SVC):


    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('SVC_')

        space_gen.generate_number(self.name + "C", 1.0, depending_node=depending_node, low=0.03125, high=32768, is_log=True)
        category_kernel = space_gen.generate_cat(self.name + "kernel", ["rbf", "poly", "sigmoid"], "rbf", depending_node=depending_node)

        space_gen.generate_number(self.name + "degree", 3, depending_node=category_kernel[1], low=2, high=5, is_float=False)
        space_gen.generate_number(self.name + "gamma", 0.1, depending_node=depending_node, low=3.0517578125e-05, high=8, is_log=True)
        space_gen.generate_number(self.name + "coef0", 0, depending_node=depending_node, low=-1, high=1) #todo: fix if we change to graph
        space_gen.generate_cat(self.name + "shrinking", ['True', 'False'], 'True', depending_node=depending_node)
        space_gen.generate_number(self.name + "tol", 1e-3, depending_node=depending_node, low=1e-5, high=1e-1, is_log=True)

