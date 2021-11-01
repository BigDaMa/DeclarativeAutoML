from sklearn.decomposition import TruncatedSVD
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class TruncatedSVDOptuna(TruncatedSVD):
    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('TruncatedSVD_')
        space_gen.generate_number(self.name + 'target_dim', 128, depending_node=depending_node, low=10, high=256, is_log=False, is_float=False)



