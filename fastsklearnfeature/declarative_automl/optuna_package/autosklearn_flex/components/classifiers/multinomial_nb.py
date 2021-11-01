from sklearn.naive_bayes import MultinomialNB
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.utils.class_weight import compute_sample_weight
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.wrapper.ClassifierWrapper import \
    calculate_class_weight

class MultinomialNBOptuna(MultinomialNB):

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('MultinomialNB_')
        space_gen.generate_number(self.name + "alpha", 1, depending_node=depending_node, low=1e-2, high=100, is_log=True)
        space_gen.generate_cat(self.name + "fit_prior", ['True', 'False'], 'True', depending_node=depending_node)
