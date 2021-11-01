from sklearn.linear_model import SGDClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.utils.class_weight import compute_sample_weight
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.wrapper.ClassifierWrapper import \
    calculate_class_weight

class SGDClassifierOptuna(SGDClassifier):

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('SGDClassifier_')

        space_gen.generate_cat(self.name + "loss", ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"], "log", depending_node=depending_node)
        space_gen.generate_cat(self.name + "penalty", ["l1", "l2", "elasticnet"], "l2", depending_node=depending_node)
        space_gen.generate_number(self.name + "alpha", 0.0001, depending_node=depending_node, low=1e-7, high=1e-1, is_log=True)
        space_gen.generate_number(self.name + "l1_ratio", 0.15, depending_node=depending_node, low=1e-9, high=1, is_log=True)
        space_gen.generate_number(self.name + "tol", 1e-4, depending_node=depending_node, low=1e-5, high=1e-1, is_log=True)
        space_gen.generate_number(self.name + "epsilon", 1e-4, depending_node=depending_node, low=1e-5, high=1e-1, is_log=True)
        category_learn = space_gen.generate_cat(self.name + "learning_rate", ["optimal", "invscaling", "constant"], "invscaling", depending_node=depending_node)

        space_gen.generate_number(self.name + "eta0", 0.01, depending_node=depending_node, low=1e-7, high=1e-1, is_log=True) #fix if we change to graph
        space_gen.generate_number(self.name + "power_t", 0.5, depending_node=category_learn[1], low=1e-5, high=1)
        space_gen.generate_cat(self.name + "average", ['False', 'True'], 'False', depending_node=depending_node)

