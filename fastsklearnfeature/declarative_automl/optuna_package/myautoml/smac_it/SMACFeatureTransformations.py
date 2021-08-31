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


def extract_value(x, cs, name):
    h_param = cs._hyperparameters[name]
    h_param_id = cs._hyperparameter_idx[name]
    vector_val = x[h_param_id]
    real_value = h_param._transform(vector_val)
    return real_value


class FeatureTransformations(BaseEstimator, TransformerMixin):

    def get_metadata_features(self, dataset_id, x, cs, catch_exception=False):
        metafeature_values = np.zeros((1, 29))
        try:
            X_train, _, y_train, _, categorical_indicator, attribute_names = get_data(dataset_id, randomstate=42)

            '''
            random_feature_selection = x['random_feature_selection']
            rand_feature_ids = np.arange(X_train.shape[1])
            np.random.seed(my_random_seed)
            np.random.shuffle(rand_feature_ids)
            number_of_sampled_features = int(X_train.shape[1] * random_feature_selection)
    
            X_train = X_train[:, rand_feature_ids[0:number_of_sampled_features]]
            X_test = X_test[:, rand_feature_ids[0:number_of_sampled_features]]
            categorical_indicator = np.array(categorical_indicator)[rand_feature_ids[0:number_of_sampled_features]]
            attribute_names = np.array(attribute_names)[rand_feature_ids[0:number_of_sampled_features]]
            '''

            # sampling with class imbalancing
            my_random_seed = 41
            class_labels = np.unique(y_train)

            ids_class0 = np.array((y_train == class_labels[0]).nonzero()[0])
            ids_class1 = np.array((y_train == class_labels[1]).nonzero()[0])

            np.random.seed(my_random_seed)
            np.random.shuffle(ids_class0)
            np.random.seed(my_random_seed)
            np.random.shuffle(ids_class1)

            if extract_value(x, cs, 'unbalance_data'):
                fraction_ids_class0 = extract_value(x, cs, 'fraction_ids_class0')
                fraction_ids_class1 = extract_value(x, cs, 'fraction_ids_class1')
            else:
                sampling_factor_train_only = extract_value(x, cs, 'sampling_factor_train_only')
                fraction_ids_class0 = sampling_factor_train_only
                fraction_ids_class1 = sampling_factor_train_only

            number_class0 = int(fraction_ids_class0 * len(ids_class0))
            number_class1 = int(fraction_ids_class1 * len(ids_class1))

            all_sampled_training_ids = []
            all_sampled_training_ids.extend(ids_class0[0:number_class0])
            all_sampled_training_ids.extend(ids_class1[0:number_class1])

            X_train = X_train[all_sampled_training_ids, :]
            y_train = y_train[all_sampled_training_ids]

            metafeature_values = data2features(X_train, y_train, categorical_indicator)

            return metafeature_values
        except:
            if catch_exception:
                return metafeature_values
            else:
                raise Exception('here happened something')

    def transform(self, X, cs, metafeatures_pre=None, catch_exception=False):
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
            if type(metafeatures_pre) == type(None):
                d_id = int(X[i, _hyperparameter_idx['dataset_id']])
                data_id = int(cs._hyperparameters['dataset_id'].choices[d_id])
                metafeatures = self.get_metadata_features(data_id, X[i,:], cs, catch_exception)
            else:
                metafeatures = metafeatures_pre

            #products
            product_cvs = np.multiply(X[i, _hyperparameter_idx['global_cv']], X[i, _hyperparameter_idx['global_number_cv']])
            product_hold_out_test = np.multiply(np.multiply(X[i, _hyperparameter_idx['hold_out_fraction']], metafeatures [:, metafeature_names.index('NumberOfInstances')]), X[i, _hyperparameter_idx['sample_fraction']])
            product_hold_out_training = np.multiply(np.multiply((1.0 - X[i, _hyperparameter_idx['hold_out_fraction']]), metafeatures [:, metafeature_names.index('NumberOfInstances')]), X[i, _hyperparameter_idx['sample_fraction']])
            product_sampled_data = np.multiply(metafeatures [:, metafeature_names.index('NumberOfInstances')], X[i, _hyperparameter_idx['sample_fraction']])

            number_of_evaluations = np.divide(X[i, _hyperparameter_idx['global_evaluation_time_constraint']], X[i, _hyperparameter_idx['global_search_time_constraint']])

            '''
            #logs
            log_global_search_time_constraint = np.log(X[i, _hyperparameter_idx['global_search_time_constraint']])
            log_global_evaluation_time_constraint = np.log(X[i, _hyperparameter_idx['global_evaluation_time_constraint']])
            log_global_memory_constraint = np.log(X[i, _hyperparameter_idx['global_memory_constraint']])
            log_privacy = np.log(X[i, _hyperparameter_idx['privacy_constraint']])
            log_sampled_instances = np.log(product_sampled_data)
            '''

            new_list = stuff2list([X[i,:],
                              product_cvs.reshape((1, 1)),
                              product_hold_out_test.reshape((1, 1)),
                              product_hold_out_training.reshape((1, 1)),
                              product_sampled_data.reshape((1, 1)),
                              number_of_evaluations.reshape((1, 1)),
                              #log_global_search_time_constraint.reshape((1, 1)),
                              #log_global_evaluation_time_constraint.reshape((1, 1)),
                              #log_global_memory_constraint.reshape((1, 1)),
                              #log_privacy.reshape((1, 1)),
                              #log_sampled_instances.reshape((1, 1)),
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
        #self.feature_names_new.append('log_search_time_constraint')
        #self.feature_names_new.append('log_evaluation_time_constraint')
        #self.feature_names_new.append('log_search_memory_constraint')
        #self.feature_names_new.append('log_privacy_constraint')
        #self.feature_names_new.append('log_sampled_instances')
        return self.feature_names_new


    def fit(self, X, y=None):
        return self