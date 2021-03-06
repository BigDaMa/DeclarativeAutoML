from sklearn.preprocessing import QuantileTransformer
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class QuantileTransformerOptuna(QuantileTransformer):
    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('QuantileTransformer_')

        self.n_quantiles = trial.suggest_int(self.name + 'n_quantiles', 10, 1000)
        self.output_distribution = trial.suggest_categorical(self.name + 'output_distribution', ['uniform', 'normal'])

    def fit(self, X, y):
        self.n_quantiles = min(X.shape[0], self.n_quantiles)
        return super().fit(X=X, y=y)

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('QuantileTransformer_')

        space_gen.generate_number(self.name + 'n_quantiles', 1000, depending_node=depending_node, low=10, high=1000, is_float=False)
        space_gen.generate_cat(self.name + 'output_distribution', ['uniform', 'normal'], 'uniform', depending_node=depending_node)
