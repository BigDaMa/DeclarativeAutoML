from sklearn.preprocessing import PolynomialFeatures
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class PolynomialFeaturesOptuna(PolynomialFeatures):
    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('PolynomialFeatures_')

        space_gen.generate_number(self.name + "degree", 2, depending_node=depending_node, low=2, high=3, is_float=False)
        space_gen.generate_cat(self.name + "interaction_only", ['False', 'True'], 'False', depending_node=depending_node)
        space_gen.generate_cat(self.name + "include_bias", ['True', 'False'], 'True', depending_node=depending_node)


