from sklearn.kernel_approximation import Nystroem
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class NystroemOptuna(Nystroem):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('Nystroem_')

        self.kernel = trial.suggest_categorical(self.name + 'kernel', ['poly', 'rbf', 'sigmoid', 'cosine', 'chi2'])
        self.n_components = trial.suggest_int(self.name + "n_components", 50, 10000, log=True)
        self.gamma = trial.suggest_loguniform(self.name + "gamma", 3.0517578125e-05, 8)

        if self.kernel == 'poly':
            self.degree = trial.suggest_int(self.name + 'degree', 2, 5, log=False)

        if self.kernel == "poly" or self.kernel == "sigmoid":
            self.coef0 = trial.suggest_uniform(self.name + "coef0", -1, 1)

        self.sparse = False
        self.n_jobs = 1

    def fit(self, X, y=None):
        self.n_components = min(len(X), self.n_components)
        return super().fit(X=X, y=y)

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('Nystroem_')

        category_kernel = space_gen.generate_cat(self.name + 'kernel', ['poly', 'rbf', 'sigmoid', 'cosine', 'chi2'], 'rbf', depending_node=depending_node)
        space_gen.generate_number(self.name + "n_components", 100, depending_node=depending_node, low=50, high=10000, is_float=False, is_log=True)
        space_gen.generate_number(self.name + "gamma", 0.1, depending_node=depending_node, low=3.0517578125e-05, high=8, is_log=True)
        space_gen.generate_number(self.name + 'degree', 3, depending_node=category_kernel[0], low=2, high=5, is_float=False)
        space_gen.generate_number(self.name + "coef0", 0, depending_node=depending_node, low=-1, high=1) # todo: fix once it is a graph

