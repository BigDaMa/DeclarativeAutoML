from sklearn.decomposition import FastICA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.base import BaseEstimator, TransformerMixin

class QuantileTransformerComponent(TransformerMixin):
    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('QuantileTransformerComponent_')

        space_gen.generate_number(self.name + 'n_quantiles', 1000, depending_node=depending_node, low=10, high=2000,
                                  is_float=False, is_log=False)
        space_gen.generate_cat(self.name + "output_distribution", ["uniform", "normal"], "uniform",
                               depending_node=depending_node)




