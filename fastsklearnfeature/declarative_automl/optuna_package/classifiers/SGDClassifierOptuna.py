from sklearn.linear_model import SGDClassifier
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.utils.class_weight import compute_sample_weight
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.wrapper.ClassifierWrapper import \
    calculate_class_weight

class SGDClassifierOptuna(SGDClassifier):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('SGDClassifier_')
        self.n_jobs = 1
        self.loss = trial.suggest_categorical(self.name + "loss", ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"])
        self.penalty = trial.suggest_categorical(self.name + "penalty", ["l1", "l2", "elasticnet"])
        self.alpha = trial.suggest_loguniform(self.name + "alpha", 1e-7, 1e-1)
        self.l1_ratio = trial.suggest_loguniform(self.name + "l1_ratio", 1e-9, 1)
        self.fit_intercept = True
        self.tol = trial.suggest_loguniform(self.name + "tol", 1e-5, 1e-1)
        self.epsilon = trial.suggest_loguniform(self.name + "epsilon", 1e-5, 1e-1)
        self.learning_rate = trial.suggest_categorical(self.name + "learning_rate", ["optimal", "invscaling", "constant"])

        if self.learning_rate == "constant" or self.learning_rate == "invscaling" or self.learning_rate == "adaptive":
            self.eta0 = trial.suggest_loguniform(self.name + "eta0", 1e-7, 1e-1)

        if self.learning_rate == "invscaling":
            self.power_t = trial.suggest_uniform(self.name + "power_t", 1e-5, 1)
        self.average = trial.suggest_categorical(self.name + "average", [False, True])

        self.max_iter = trial.suggest_int(self.name + "max_iter", 10, 1024)

        #todo: add conditional parameters

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('SGDClassifier_')

        space_gen.generate_cat(self.name + "loss", ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"], "log", depending_node=depending_node)
        space_gen.generate_cat(self.name + "penalty", ["l1", "l2", "elasticnet"], "l2", depending_node=depending_node)
        space_gen.generate_number(self.name + "alpha", 0.0001, depending_node=depending_node, low=1e-7, high=1e-1, is_log=True)
        space_gen.generate_number(self.name + "l1_ratio", 0.15, depending_node=depending_node, low=1e-9, high=1, is_log=True)
        space_gen.generate_number(self.name + "tol", 1e-4, depending_node=depending_node, low=1e-5, high=1e-1, is_log=True)
        space_gen.generate_number(self.name + "epsilon", 1e-4, depending_node=depending_node, low=1e-5, high=1e-1, is_log=True)
        category_learn = space_gen.generate_cat(self.name + "learning_rate", ["optimal", "invscaling", "constant"], "invscaling", depending_node=depending_node)

        space_gen.generate_number(self.name + "eta0", 0.01, depending_node=depending_node, low=1e-7, high=1e-1, is_log=True) #fix if we change to graph
        space_gen.generate_number(self.name + "power_t", 0.5, depending_node=category_learn[1], low=1e-5, high=1)
        space_gen.generate_cat(self.name + "average", [False, True], False, depending_node=depending_node)

        space_gen.generate_number(self.name + "max_iter", 1000, depending_node=depending_node, low=10, high=1024, is_float=False)

    def set_weight(self, custom_weight):
        self.custom_weight = custom_weight

    def fit(self, X, y=None, sample_weight=None):
        if hasattr(self, 'custom_weight'):
            return super().fit(X, y, sample_weight=compute_sample_weight(class_weight=self.custom_weight, y=y))
        else:
            return super().fit(X, y)