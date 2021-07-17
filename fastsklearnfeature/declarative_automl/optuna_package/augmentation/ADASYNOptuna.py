from imblearn.over_sampling import ADASYN
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
import numpy as np

class ADASYNOptuna(ADASYN):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('ADASYN_')
        self.random_state = 0
        self.n_jobs = 1

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('ADASYN_')