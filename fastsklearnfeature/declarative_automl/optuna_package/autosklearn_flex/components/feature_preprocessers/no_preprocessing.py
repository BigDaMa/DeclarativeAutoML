from sklearn.svm import LinearSVC
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.utils.class_weight import compute_sample_weight
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.wrapper.ClassifierWrapper import \
    calculate_class_weight

class NoPreprocessing(LinearSVC):

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('NoPreprocessing_')
