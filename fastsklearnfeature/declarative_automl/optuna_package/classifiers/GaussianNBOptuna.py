from sklearn.naive_bayes import GaussianNB
from sklearn.utils.class_weight import compute_sample_weight
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.wrapper.ClassifierWrapper import \
    calculate_class_weight

class GaussianNBOptuna(GaussianNB):

    def init_hyperparameters(self, trial, X, y):
        #self.classes_ = np.unique(y.astype(int))
        pass

    def generate_hyperparameters(self, space_gen, depending_node=None):
        pass

    def set_weight(self, custom_weight):
        self.custom_weight = custom_weight

    def fit(self, X, y=None, sample_weight=None):
        if hasattr(self, 'custom_weight'):
            class_weight = 'balanced'
            if self.custom_weight != 0.5:
                class_weight = calculate_class_weight(y, self.custom_weight)
            return super().fit(X, y, sample_weight=compute_sample_weight(class_weight=class_weight, y=y))
        else:
            return super().fit(X, y)