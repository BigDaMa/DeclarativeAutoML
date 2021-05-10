from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.utils.class_weight import compute_sample_weight
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.wrapper.ClassifierWrapper import \
    calculate_class_weight

class QuadraticDiscriminantAnalysisOptuna(QuadraticDiscriminantAnalysis):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('QuadraticDiscriminantAnalysis_')
        #self.classes_ = np.unique(y.astype(int))

        self.reg_param = trial.suggest_uniform(self.name + 'reg_param', 0.0, 1.0)

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('QuadraticDiscriminantAnalysis_')
        space_gen.generate_number(self.name + 'reg_param', 0.0, depending_node=depending_node, low=0.0, high=1.0)

