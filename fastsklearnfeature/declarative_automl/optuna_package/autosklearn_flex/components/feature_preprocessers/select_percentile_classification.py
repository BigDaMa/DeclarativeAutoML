from sklearn.decomposition import FastICA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.base import BaseEstimator, TransformerMixin

class SelectPercentileClassification(TransformerMixin):
    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('SelectPercentileClassification_')

        space_gen.generate_cat(self.name + "score_func", ["chi2", "f_classif", "mutual_info"], "chi2", depending_node=depending_node)
        space_gen.generate_number(self.name + 'percentile', 50, depending_node=depending_node, low=1, high=99, is_float=False, is_log=False)









