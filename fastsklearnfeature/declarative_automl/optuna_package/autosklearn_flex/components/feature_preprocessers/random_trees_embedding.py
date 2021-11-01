from sklearn.ensemble import RandomTreesEmbedding
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name

class RandomTreesEmbeddingOptuna(RandomTreesEmbedding):
    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('RandomTreesEmbedding_')

        space_gen.generate_number(self.name + "n_estimators", 10, depending_node=depending_node, low=10, high=100, is_float=False, is_log=True)
        space_gen.generate_number(self.name + "max_depth", 5, depending_node=depending_node, low=2, high=10, is_float=False)
        space_gen.generate_number(self.name + "min_samples_split", 2, depending_node=depending_node, low=2, high=20, is_float=False)
        space_gen.generate_number(self.name + "min_samples_leaf", 1, depending_node=depending_node, low=1, high=20, is_float=False)
        space_gen.generate_cat(self.name + "bootstrap", ['True', 'False'], 'False', depending_node=depending_node)


