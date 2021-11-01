from sklearn.decomposition import FastICA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.base import BaseEstimator, TransformerMixin

class MinorityCoalescer(TransformerMixin):
    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('MinorityCoalescer_')
        space_gen.generate_number(self.name + 'minimum_fraction', 0.01, depending_node=depending_node, low=0.0001, high=0.5,
                                  is_float=True, is_log=True)




