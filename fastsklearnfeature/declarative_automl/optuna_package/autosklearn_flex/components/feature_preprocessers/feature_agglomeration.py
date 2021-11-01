from sklearn.cluster import FeatureAgglomeration
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import categorical
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class FeatureAgglomerationOptuna(FeatureAgglomeration):
    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('FeatureAgglomeration_')

        space_gen.generate_number(self.name + 'n_clusters', 25, depending_node=depending_node, low=2, high=400, is_float=False, is_log=False)
        space_gen.generate_cat(self.name + "linkage", ["ward", "complete", "average"], "ward", depending_node=depending_node)
        space_gen.generate_cat(self.name + "affinity", ["euclidean", "manhattan", "cosine"], "euclidean", depending_node=depending_node)
        space_gen.generate_cat(self.name + "pooling_func", ["mean", "median", "max"], "mean", depending_node=depending_node)


