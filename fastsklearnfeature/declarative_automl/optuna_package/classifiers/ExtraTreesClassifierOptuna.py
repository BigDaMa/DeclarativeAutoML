from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.utils.class_weight import compute_sample_weight
from autosklearn.pipeline.implementations.util import convert_multioutput_multiclass_to_multilabel


class ExtraTreesClassifierOptuna(ExtraTreesClassifier):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('ExtraTreesClassifier_')
        self.criterion = trial.suggest_categorical(self.name + "criterion", ["gini", "entropy"])
        self.max_features = trial.suggest_uniform(self.name + "max_features", 0., 1.)
        self.min_samples_split = trial.suggest_int(self.name + "min_samples_split", 2, 20, log=False)
        self.min_samples_leaf = trial.suggest_int(self.name + "min_samples_leaf", 1, 20, log=False)
        self.max_depth = None
        self.min_weight_fraction_leaf = 0.
        self.max_leaf_nodes = None
        self.min_impurity_decrease = 0.0
        self.bootstrap = trial.suggest_categorical(self.name + "bootstrap", [True, False])
        self.classes_ = np.unique(y.astype(int))
        self.n_jobs = 1

        # hyperopt config
        self.n_estimators = 1

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('ExtraTreesClassifier_')
        space_gen.generate_cat(self.name + "criterion", ["gini", "entropy"], "gini", depending_node=depending_node)
        space_gen.generate_number(self.name + 'max_features', 0.5, depending_node=depending_node, low=0., high=1.)
        space_gen.generate_number(self.name + "min_samples_split", 2, depending_node=depending_node, low=2, high=20, is_float=False)
        space_gen.generate_number(self.name + "min_samples_leaf", 1, depending_node=depending_node, low=1, high=20, is_float=False)
        space_gen.generate_cat(self.name + "bootstrap", [True, False], False, depending_node=depending_node)

    def set_weight(self, custom_weight):
        self.custom_weight = custom_weight

    def fit(self, X, y=None, sample_weight=None):
        if hasattr(self, 'custom_weight'):
            return super().fit(X, y, sample_weight=compute_sample_weight(class_weight=self.custom_weight, y=y))
        else:
            return super().fit(X, y)

    def custom_iterative_fit(self, X, y=None, sample_weight=None, number_steps=2):
        self.n_estimators = number_steps
        return self.fit(X, y=y, sample_weight=sample_weight)

    def get_max_steps(self):
        return 512

    def predict_proba(self, X):
        probas = super().predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas