from sklearn.decomposition import FastICA
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class FastICAOptuna(FastICA):
    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('FastICA_')

        category_whiten = space_gen.generate_cat(self.name + 'whiten', ['False', 'True'], 'False',
                                                 depending_node=depending_node)

        space_gen.generate_number(self.name + 'n_components', 100, depending_node=category_whiten[1], low=10, high=2000, is_float=False,is_log=False)
        space_gen.generate_cat(self.name + 'algorithm', ['parallel', 'deflation'], 'parallel', depending_node=depending_node)
        space_gen.generate_cat(self.name + 'fun', ['logcosh', 'exp', 'cube'], 'logcosh', depending_node=depending_node)



