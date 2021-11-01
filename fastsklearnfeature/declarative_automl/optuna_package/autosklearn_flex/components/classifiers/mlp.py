from sklearn.neural_network import MLPClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class MLPOptuna(MLPClassifier):

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('MLPClassifier_')

        space_gen.generate_number(self.name + "hidden_layer_depth", 1, depending_node=depending_node, low=1, high=3,
                                  is_float=False, is_log=False)
        space_gen.generate_number(self.name + "num_nodes_per_layer", 32, depending_node=depending_node, low=16, high=264,
                                  is_float=False, is_log=True)
        space_gen.generate_cat(self.name + "activation", ['tanh', 'relu'], 'relu',
                               depending_node=depending_node)
        space_gen.generate_number(self.name + "alpha", 1e-4, depending_node=depending_node, low=1e-7,
                                  high=1e-1,
                                  is_float=True, is_log=True)
        space_gen.generate_number(self.name + "learning_rate_init", 1e-3, depending_node=depending_node, low=1e-4,
                                  high=0.5,
                                  is_float=True, is_log=True)
        space_gen.generate_cat(self.name + "early_stopping", ["valid", "train"], "valid",
                               depending_node=depending_node)

