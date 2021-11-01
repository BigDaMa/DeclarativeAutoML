from sklearn.decomposition import FastICA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.base import BaseEstimator, TransformerMixin

class SelectRegressionRates(TransformerMixin):
    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('SelectRegressionRates_')
