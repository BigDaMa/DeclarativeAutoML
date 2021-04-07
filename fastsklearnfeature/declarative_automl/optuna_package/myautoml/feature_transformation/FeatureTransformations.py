from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import copy

class FeatureTransformations(BaseEstimator, TransformerMixin):

    def transform(self, X, feature_names):
        #products
        product_cvs = np.multiply(X[:, feature_names.index('global_cv')], X[:, feature_names.index('global_number_cv')])
        product_hold_out_test = np.multiply(np.multiply(X[:, feature_names.index('hold_out_fraction')], X[:, feature_names.index('NumberOfInstances')]), X[:, feature_names.index('sample_fraction')])
        product_hold_out_training = np.multiply(np.multiply((1.0 - X[:, feature_names.index('hold_out_fraction')]), X[:, feature_names.index('NumberOfInstances')]), X[:, feature_names.index('sample_fraction')])
        product_sampled_data = np.multiply(X[:, feature_names.index('NumberOfInstances')], X[:, feature_names.index('sample_fraction')])

        number_of_evaluations = np.divide(X[:, feature_names.index('global_evaluation_time_constraint')], X[:, feature_names.index('global_search_time_constraint')])

        #logs
        log_global_search_time_constraint = np.log(X[:, feature_names.index('global_search_time_constraint')])
        log_global_evaluation_time_constraint = np.log(X[:, feature_names.index('global_evaluation_time_constraint')])
        log_global_memory_constraint = np.log(X[:, feature_names.index('global_memory_constraint')])
        log_privacy = np.log(X[:, feature_names.index('privacy')])
        log_sampled_instances = np.log(product_sampled_data)

        has_privacy_constraint = X[:, feature_names.index('privacy')] < 1000
        has_evaluation_time_constraint = X[:, feature_names.index('global_evaluation_time_constraint')] < X[:, feature_names.index('global_search_time_constraint')]


        return np.hstack((X,
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
                          has_privacy_constraint.reshape((1, 1)),
                          has_evaluation_time_constraint.reshape((1, 1))
                        ))


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
        self.feature_names_new.append('has_privacy_constraint')
        self.feature_names_new.append('has_evaluation_time_constraint')
        return self.feature_names_new


    def fit(self, X, y=None):
        return self