from sklearn.kernel_approximation import RBFSampler
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class RBFSamplerOptuna(RBFSampler):
    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('RBFSampler_')

        space_gen.generate_number(self.name + "gamma", 1.0, depending_node=depending_node, low=3.0517578125e-05, high=8, is_log=True)
        space_gen.generate_number(self.name + "n_components", 100, depending_node=depending_node, low=50, high=10000, is_float=False, is_log=True)

