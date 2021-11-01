from sklearn.neighbors import KNeighborsClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class KNeighborsClassifierOptuna(KNeighborsClassifier):

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('KNeighborsClassifier_')
        space_gen.generate_number(self.name + "n_neighbors", 1, depending_node=depending_node, low=1, high=100, is_float=False, is_log=True)
        space_gen.generate_cat(self.name + "weights", ["uniform", "distance"], "uniform", depending_node=depending_node)
        space_gen.generate_cat(self.name + "p", [1, 2], 2, depending_node=depending_node)
