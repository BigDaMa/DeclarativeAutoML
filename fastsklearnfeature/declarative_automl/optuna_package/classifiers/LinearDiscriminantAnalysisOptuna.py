from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from autosklearn.pipeline.implementations.util import softmax

class LinearDiscriminantAnalysisOptuna(LinearDiscriminantAnalysis):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('LinearDiscriminantAnalysis_')

        self.tol = trial.suggest_loguniform(self.name + 'tol', 1e-5, 1e-1)
        shrinkage_choice = trial.suggest_categorical(self.name + "shrinkage_choice", ["None", "auto", "manual"])

        if shrinkage_choice == 'None':
            self.shrinkage = None
            self.solver = "svd"
        elif shrinkage_choice == "auto":
            self.shrinkage = "auto"
            self.solver = "lsqr"
        elif shrinkage_choice == "manual":
            self.shrinkage = trial.suggest_uniform(self.name + "shrinkage_factor", 0.0, 1.0)
            self.solver = "lsqr"

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('LinearDiscriminantAnalysis_')

        space_gen.generate_number(self.name + 'tol', 1e-4, depending_node=depending_node, low=1e-5, high=1e-1, is_log=True)
        category_shrinkage = space_gen.generate_cat(self.name + "shrinkage_choice", ["None", "auto", "manual"], "None", depending_node=depending_node)
        space_gen.generate_number(self.name + 'shrinkage_factor', 0.5, depending_node=category_shrinkage[2], low=0.0, high=1.0)


    def predict_proba(self, X):
        df = super().predict_proba(X)
        return softmax(df)

