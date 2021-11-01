from sklearn.naive_bayes import GaussianNB
from sklearn.utils.class_weight import compute_sample_weight
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.wrapper.ClassifierWrapper import \
    calculate_class_weight

class GaussianNBOptuna(GaussianNB):

    def generate_hyperparameters(self, space_gen, depending_node=None):
        pass
