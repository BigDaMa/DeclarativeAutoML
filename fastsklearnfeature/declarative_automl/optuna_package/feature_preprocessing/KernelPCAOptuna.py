from sklearn.decomposition import KernelPCA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class KernelPCAOptuna(KernelPCA):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('KernelPCA_')

        self.n_components_fraction = trial.suggest_uniform(self.name + 'n_components_fraction', 0.0, 1.0)
        self.kernel = trial.suggest_categorical(self.name + 'kernel', ['poly', 'rbf', 'sigmoid', 'cosine'])
        self.gamma = trial.suggest_loguniform(self.name + "gamma", 3.0517578125e-05, 8)

        if self.kernel == 'poly':
            self.degree = trial.suggest_int(self.name + 'degree', 2, 5, log=False)

        if self.kernel == "poly" or self.kernel == "sigmoid":
            self.coef0 = trial.suggest_uniform(self.name + "coef0", -1, 1)

        self.remove_zero_eig = True

        self.sparse = False

    def fit(self, X, y=None):
        self.n_components = max(1, int(self.n_components_fraction * X.shape[1]))
        #print('ncom: ' + str(self.n_components))
        return super().fit(X=X, y=y)

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('KernelPCA_')

        space_gen.generate_number(self.name + "n_components_fraction", 0.5, depending_node=depending_node)
        category_kernel = space_gen.generate_cat(self.name + 'kernel', ['poly', 'rbf', 'sigmoid', 'cosine'], 'rbf', depending_node=depending_node)
        space_gen.generate_number(self.name + "gamma", 1.0, depending_node=depending_node)
        space_gen.generate_number(self.name + 'degree', 3, depending_node=category_kernel[0])
        space_gen.generate_number(self.name + "coef0", 0, depending_node=depending_node) # todo: fix once it is a graph

