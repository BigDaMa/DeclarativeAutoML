from sklearn.kernel_approximation import Nystroem
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class NystroemOptuna(Nystroem):

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('Nystroem_')

        category_kernel = space_gen.generate_cat(self.name + 'kernel', ['poly', 'rbf', 'sigmoid', 'cosine', 'chi2'], 'rbf', depending_node=depending_node)
        space_gen.generate_number(self.name + "n_components", 100, depending_node=depending_node, low=50, high=10000, is_float=False, is_log=True)
        space_gen.generate_number(self.name + "gamma", 0.1, depending_node=depending_node, low=3.0517578125e-05, high=8, is_log=True)
        space_gen.generate_number(self.name + 'degree', 3, depending_node=category_kernel[0], low=2, high=5, is_float=False)
        space_gen.generate_number(self.name + "coef0", 0, depending_node=depending_node, low=-1, high=1) # todo: fix once it is a graph

