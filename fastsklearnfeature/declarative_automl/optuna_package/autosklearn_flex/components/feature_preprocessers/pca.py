from sklearn.decomposition import PCA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class PCAOptuna(PCA):
    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('PCA_')

        space_gen.generate_number(self.name + "keep_variance", 0.9999, depending_node=depending_node, low=0.5, high=0.9999)
        space_gen.generate_cat(self.name + "whiten", ['False', 'True'], 'False', depending_node=depending_node)
