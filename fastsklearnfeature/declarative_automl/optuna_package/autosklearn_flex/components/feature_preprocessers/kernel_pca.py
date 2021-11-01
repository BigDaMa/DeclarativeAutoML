from sklearn.decomposition import KernelPCA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class KernelPCAOptuna(KernelPCA):
    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('KernelPCA_')

        space_gen.generate_number(self.name + "n_components", 100, depending_node=depending_node, low=10, high=2000, is_log=False, is_float=False)
        category_kernel = space_gen.generate_cat(self.name + 'kernel', ['poly', 'rbf', 'sigmoid', 'cosine'], 'rbf', depending_node=depending_node)
        space_gen.generate_number(self.name + "gamma", 1.0, depending_node=depending_node, low=3.0517578125e-05, high=8, is_log=True)
        space_gen.generate_number(self.name + 'degree', 3, depending_node=category_kernel[0], low=2, high=5, is_float=False)
        space_gen.generate_number(self.name + "coef0", 0, depending_node=depending_node, low=-1, high=1) # todo: fix once it is a graph

