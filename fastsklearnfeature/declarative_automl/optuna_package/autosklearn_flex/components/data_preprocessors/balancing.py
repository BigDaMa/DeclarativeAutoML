from sklearn.decomposition import FastICA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.base import BaseEstimator, TransformerMixin

class Balancing(TransformerMixin):
    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('Balancing_')

        space_gen.generate_cat(self.name + 'strategy', ['none', 'weighting'], 'none', depending_node=depending_node)



