from sklearn.decomposition import FastICA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from sklearn.base import BaseEstimator, TransformerMixin

class SelectClassificationRates(TransformerMixin):
    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('SelectClassificationRates_')

        space_gen.generate_number(self.name + 'alpha', 0.1, depending_node=depending_node, low=0.01, high=0.5, is_float=True, is_log=False)
        space_gen.generate_cat(self.name + "score_func", ["chi2", "f_classif", "mutual_info_classif"], "chi2", depending_node=depending_node)
        space_gen.generate_cat(self.name + "mode", ["fpr", "fdr", "fwe"], "fpr", depending_node=depending_node)











