from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import copy
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import data2features


def stuff2list(my_list):
    new_list = []
    for list_i in my_list:
        for element_i in list_i.flatten():
            new_list.append(element_i)
    return new_list



class FeatureTransformations(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.id2features = {}


    def get_metadata_features(self, dataset_id):
        if dataset_id in self.id2features:
            return self.id2features[dataset_id]
        else:
            X_train, _, y_train, _, categorical_indicator, attribute_names = get_data(dataset_id, randomstate=42)
            metafeature_values = data2features(X_train, y_train, categorical_indicator)
            return metafeature_values

    def transform(self, X, cs):
        _hyperparameter_idx = cs._hyperparameter_idx

        metafeature_names = ['ClassEntropy', 'ClassProbabilityMax', 'ClassProbabilityMean', 'ClassProbabilityMin',
                             'ClassProbabilitySTD',
                             'DatasetRatio', 'InverseDatasetRatio', 'LogDatasetRatio', 'LogInverseDatasetRatio',
                             'LogNumberOfFeatures',
                             'LogNumberOfInstances', 'NumberOfCategoricalFeatures', 'NumberOfClasses',
                             'NumberOfFeatures',
                             'NumberOfFeaturesWithMissingValues', 'NumberOfInstances',
                             'NumberOfInstancesWithMissingValues',
                             'NumberOfMissingValues', 'NumberOfNumericFeatures',
                             'PercentageOfFeaturesWithMissingValues',
                             'PercentageOfInstancesWithMissingValues', 'PercentageOfMissingValues',
                             'RatioNominalToNumerical',
                             'RatioNumericalToNominal', 'SymbolsMax', 'SymbolsMean', 'SymbolsMin', 'SymbolsSTD',
                             'SymbolsSum']

        # data
        feature_array = []

        for i in range(X.shape[0]):
            d_id = int(X[i, _hyperparameter_idx['dataset_id']])
            data_id = int(cs._hyperparameters['dataset_id'].choices[d_id])
            metafeatures = self.get_metadata_features(data_id)

            #products
            product_cvs = np.multiply(X[i, _hyperparameter_idx['global_cv']], X[i, _hyperparameter_idx['global_number_cv']])
            product_hold_out_test = np.multiply(np.multiply(X[i, _hyperparameter_idx['hold_out_fraction']], metafeatures [:, metafeature_names.index('NumberOfInstances')]), X[i, _hyperparameter_idx['sample_fraction']])
            product_hold_out_training = np.multiply(np.multiply((1.0 - X[i, _hyperparameter_idx['hold_out_fraction']]), metafeatures [:, metafeature_names.index('NumberOfInstances')]), X[i, _hyperparameter_idx['sample_fraction']])
            product_sampled_data = np.multiply(metafeatures [:, metafeature_names.index('NumberOfInstances')], X[i, _hyperparameter_idx['sample_fraction']])

            number_of_evaluations = np.divide(X[i, _hyperparameter_idx['global_evaluation_time_constraint']], X[i, _hyperparameter_idx['global_search_time_constraint']])

            #logs
            log_global_search_time_constraint = np.log(X[i, _hyperparameter_idx['global_search_time_constraint']])
            log_global_evaluation_time_constraint = np.log(X[i, _hyperparameter_idx['global_evaluation_time_constraint']])
            log_global_memory_constraint = np.log(X[i, _hyperparameter_idx['global_memory_constraint']])
            log_privacy = np.log(X[i, _hyperparameter_idx['privacy_constraint']])
            log_sampled_instances = np.log(product_sampled_data)

            new_list = stuff2list([X[i,:],
                              product_cvs.reshape((1, 1)),
                              product_hold_out_test.reshape((1, 1)),
                              product_hold_out_training.reshape((1, 1)),
                              product_sampled_data.reshape((1, 1)),
                              number_of_evaluations.reshape((1, 1)),
                              log_global_search_time_constraint.reshape((1, 1)),
                              log_global_evaluation_time_constraint.reshape((1, 1)),
                              log_global_memory_constraint.reshape((1, 1)),
                              log_privacy.reshape((1, 1)),
                              log_sampled_instances.reshape((1, 1)),
                              metafeatures])
            feature_array.append(new_list)

        return np.array(feature_array, dtype=float)


    def get_new_feature_names(self, feature_names):
        self.feature_names_new = copy.deepcopy(feature_names)
        self.feature_names_new.append('product_cvs')
        self.feature_names_new.append('hold_out_test_instances')
        self.feature_names_new.append('hold_out_training_instances')
        self.feature_names_new.append('sampled_instances')
        self.feature_names_new.append('number_of_evaluations')
        self.feature_names_new.append('log_search_time_constraint')
        self.feature_names_new.append('log_evaluation_time_constraint')
        self.feature_names_new.append('log_search_memory_constraint')
        self.feature_names_new.append('log_privacy_constraint')
        self.feature_names_new.append('log_sampled_instances')
        return self.feature_names_new


    def fit(self, X, y=None):
        return self