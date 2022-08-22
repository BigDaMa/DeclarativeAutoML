from sklearn.linear_model import PassiveAggressiveClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from autosklearn.pipeline.implementations.util import softmax

class PassiveAggressiveOptuna(PassiveAggressiveClassifier):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('PassiveAggressive_')
        self.C = trial.suggest_loguniform(self.name + "C", 1e-5, 10)
        self.fit_intercept = True
        self.loss = trial.suggest_categorical(self.name + "loss", ["hinge", "squared_hinge"])
        self.tol = trial.suggest_loguniform(self.name + "tol", 1e-5, 1e-1)
        self.average = trial.suggest_categorical(self.name + "average", [False, True])

        self.max_iter = 2
        self.n_jobs = 1

    def set_weight(self, custom_weight):
        self.class_weight = custom_weight

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('PassiveAggressive_')

        space_gen.generate_number(self.name + "C", 1.0, depending_node=depending_node, low=1e-5, high=10, is_log=True)
        space_gen.generate_cat(self.name + "loss", ["hinge", "squared_hinge"], "hinge", depending_node=depending_node)
        space_gen.generate_number(self.name + "tol", 1e-4, depending_node=depending_node, low=1e-5, high=1e-1, is_log=True)
        space_gen.generate_cat(self.name + "average", [False, True], False, depending_node=depending_node)

    def custom_iterative_fit(self, X, y=None, sample_weight=None, number_steps=2):
        self.max_iter = number_steps
        try:
            lr = "pa1" if self.loss == "hinge" else "pa2"
            self._partial_fit(
                X,
                y,
                alpha=1.0,
                C=self.C,
                loss="hinge",
                learning_rate=lr,
                max_iter=self.max_iter,
                classes=None,
                sample_weight=sample_weight,
                coef_init=None,
                intercept_init=None,
            )
        except:
            self.fit(X,y)

    def get_max_steps(self):
        return 1024

    def predict_proba(self, X):
        df = self.decision_function(X)
        return softmax(df)
