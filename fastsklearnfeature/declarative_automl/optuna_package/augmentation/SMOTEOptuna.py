from imblearn.over_sampling import SMOTE
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
import numpy as np

class SMOTEOptuna(SMOTE):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('SMOTE_')
        self.random_state = 0
        self.n_jobs = 1

        self.k_neighbors = trial.suggest_int(self.name + "n_neighbors", 1, min([100, X.shape[0]]), log=True)

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('SMOTE_')

        space_gen.generate_number(self.name + "n_neighbors", 5, depending_node=depending_node, low=1, high=100,
                                  is_float=False, is_log=True)

    def _fit_resample(self, X, y):
        self.k_neighbors = max(min(np.min(np.unique(y, return_counts=True)[1]), self.k_neighbors)-1, 1)
        return super()._fit_resample(X, y)