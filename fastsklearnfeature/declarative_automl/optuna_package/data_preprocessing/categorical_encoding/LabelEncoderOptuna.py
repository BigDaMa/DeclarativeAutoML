from sklearn.base import BaseEstimator, TransformerMixin
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
import numpy as np

class LabelEncoderOptuna(TransformerMixin, BaseEstimator):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('LabelEncoder_')

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('LabelEncoder_')

    def fit(self, X, y=None):
        self.key2id_per_column=[]
        for column_i in range(X.shape[1]):
            key2id = {}
            key_list, id_list = np.unique(X[:, [column_i]], return_index=True)
            id_list += 1
            for key_i in range(len(key_list)):
                key2id[key_list[key_i]] = id_list[key_i]
            self.key2id_per_column.append(key2id)
        return self

    def transform(self, X):
        X_encoded = np.zeros(X.shape)
        for col in range(X_encoded.shape[1]):
            for row in range(X_encoded.shape[0]):
                e = X[row, col]
                if e in self.key2id_per_column[col]:
                    X_encoded[row, col] = self.key2id_per_column[col][e]
        return X_encoded


