from sklearn.neural_network import MLPClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class MLPClassifierOptuna(MLPClassifier):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('MLPClassifier_')

        self.n_iter_no_change = 32
        self.validation_fraction = 0.1
        self.tol = 1e-4
        self.solver = "adam"
        self.batch_size = "auto"
        self.shuffle = True
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8

        hidden_layer_depth = trial.suggest_int(self.name + "hidden_layer_depth", 1, 3, log=False)
        num_nodes_per_layer = trial.suggest_int(self.name + "num_nodes_per_layer", 16, 264, log=True)
        self.hidden_layer_sizes = tuple(
            num_nodes_per_layer for _ in range(hidden_layer_depth)
        )

        self.activation = trial.suggest_categorical(self.name + "activation", ["tanh", "relu"])
        self.alpha = trial.suggest_loguniform(self.name + "alpha", 1e-7, 1e-1)
        self.learning_rate_init = trial.suggest_loguniform(self.name + "learning_rate_init", 1e-4, 0.5)
        self.early_stopping = trial.suggest_categorical(self.name + "early_stopping", ["valid", "train"])

        if self.early_stopping == "train":
            self.validation_fraction = 0.0
            self.early_stopping = False
        else:
            self.early_stopping = True


        self.max_iter = 2

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('MLPClassifier_')

        space_gen.generate_number(self.name + "hidden_layer_depth", 1, depending_node=depending_node, low=1, high=3, is_log=False, is_float=False)
        space_gen.generate_number(self.name + "num_nodes_per_layer", 32, depending_node=depending_node, low=16, high=264,
                                  is_log=True, is_float=False)
        space_gen.generate_cat(self.name + "activation", ["tanh", "relu"],
                               "relu", depending_node=depending_node)
        space_gen.generate_number(self.name + "alpha", 1e-4, depending_node=depending_node, low=1e-7, high=1e-1,
                                  is_log=True, is_float=True)
        space_gen.generate_number(self.name + "learning_rate_init", 1e-3, depending_node=depending_node, low=1e-4, high=0.5,
                                  is_log=True, is_float=True)
        space_gen.generate_cat(self.name + "early_stopping", ["valid", "train"],
                               "valid", depending_node=depending_node)


    def custom_iterative_fit(self, X, y=None, sample_weight=None, number_steps=2):
        self.max_iter = number_steps
        self.fit(X, y)

    def get_max_steps(self):
        return 512
