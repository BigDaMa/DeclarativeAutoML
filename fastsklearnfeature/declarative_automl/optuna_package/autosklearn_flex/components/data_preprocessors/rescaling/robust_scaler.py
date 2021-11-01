from sklearn.decomposition import FastICA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.base import BaseEstimator, TransformerMixin

class RobustScalerComponent(TransformerMixin):
    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('RobustScalerComponent_')

        space_gen.generate_number(self.name + 'q_min', 0.25, depending_node=depending_node, low=0.001, high=0.3,
                                  is_float=True, is_log=False)
        space_gen.generate_number(self.name + 'q_max', 0.75, depending_node=depending_node, low=0.7, high=0.999,
                                  is_float=True, is_log=False)




