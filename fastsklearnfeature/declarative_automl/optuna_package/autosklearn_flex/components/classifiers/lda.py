from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class LDAOptuna(LinearDiscriminantAnalysis):

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('LinearDiscriminantAnalysisClassifier_')

        space_gen.generate_cat(self.name + "shrinkage", ["None", "auto", "manual"], "None", depending_node=depending_node)
        space_gen.generate_number(self.name + "shrinkage_factor", 0.5, depending_node=depending_node, low=0., high=1.,
                                  is_float=True, is_log=False)
        space_gen.generate_number(self.name + "tol", 0.5, depending_node=depending_node, low=1e-5, high=1e-1,
                                  is_float=True, is_log=True)
