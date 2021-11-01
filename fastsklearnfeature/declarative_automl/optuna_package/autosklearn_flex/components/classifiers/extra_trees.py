from sklearn.ensemble import ExtraTreesClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class ExtraTreesClassifierOptuna(ExtraTreesClassifier):

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('ExtraTreesClassifier_')
        space_gen.generate_cat(self.name + "criterion", ["gini", "entropy"], "gini", depending_node=depending_node)
        space_gen.generate_number(self.name + 'max_features', 0.5, depending_node=depending_node, low=0., high=1.)
        space_gen.generate_number(self.name + "min_samples_split", 2, depending_node=depending_node, low=2, high=20, is_float=False)
        space_gen.generate_number(self.name + "min_samples_leaf", 1, depending_node=depending_node, low=1, high=20, is_float=False)
        space_gen.generate_cat(self.name + "bootstrap", ['True', 'False'], 'False', depending_node=depending_node)

