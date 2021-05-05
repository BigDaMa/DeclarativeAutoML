from imblearn.over_sampling import RandomOverSampler
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
import numpy as np

class RandomOverSamplerOptuna(RandomOverSampler):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('RandomOverSampler_')
        self.random_state = 0

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('RandomOverSampler_')