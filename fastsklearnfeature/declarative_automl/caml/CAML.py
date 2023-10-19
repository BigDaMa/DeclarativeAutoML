import numpy as np
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import ifNull
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.ensemble.AutoEnsembleSuccessive import \
            MyAutoML as AutoEnsembleML

class CAML:
    def __init__(self, time_search_budget=10*60,
                 scorer=make_scorer(balanced_accuracy_score),
                 training_time_limit=None,
                 inference_time_limit=None,
                 pipeline_size_limit=None,
                 fairness_limit=None,
                 fairness_group_id=None):
        self.time_search_budget = time_search_budget
        self.scorer = scorer
        self.training_time_limit = training_time_limit
        self.inference_time_limit = inference_time_limit
        self.pipeline_size_limit = pipeline_size_limit
        self.fairness_limit = fairness_limit
        self.fairness_group_id = fairness_group_id

        self.random_configs = np.array(pickle.load(open('randomconfigs.p', "rb")))

        my_list_constraints = ['global_search_time_constraint',
                               'global_evaluation_time_constraint',
                               'global_memory_constraint',
                               'global_cv',
                               'global_number_cv',
                               'privacy',
                               'hold_out_fraction',
                               'sample_fraction',
                               'training_time_constraint',
                               'inference_time_constraint',
                               'pipeline_size_constraint',
                               'fairness_constraint',
                               'use_ensemble',
                               'use_incremental_data',
                               'shuffle_validation'
                               ]

        self.metafeature_names_new = ['ClassEntropy', 'NumSymbols', 'SymbolsSum', 'SymbolsSTD', 'SymbolsMean', 'SymbolsMax',
                                 'SymbolsMin', 'ClassOccurences', 'ClassProbabilitySTD', 'ClassProbabilityMean',
                                 'ClassProbabilityMax', 'ClassProbabilityMin', 'InverseDatasetRatio', 'DatasetRatio',
                                 'RatioNominalToNumerical', 'RatioNumericalToNominal', 'NumberOfCategoricalFeatures',
                                 'NumberOfNumericFeatures', 'MissingValues', 'NumberOfMissingValues',
                                 'NumberOfFeaturesWithMissingValues', 'NumberOfInstancesWithMissingValues',
                                 'NumberOfFeatures', 'NumberOfClasses', 'NumberOfInstances', 'LogInverseDatasetRatio',
                                 'LogDatasetRatio', 'PercentageOfMissingValues',
                                 'PercentageOfFeaturesWithMissingValues',
                                 'PercentageOfInstancesWithMissingValues', 'LogNumberOfFeatures',
                                 'LogNumberOfInstances']

        _, self.feature_names = get_feature_names(my_list_constraints)

        self.model_success_list = []
        self.model_success_list.append(pickle.load(open(
            'my_great_model_compare_scaled' + str(14) + "_" + str(
                0.1) + ".p", "rb")))
    
    
    def fit(self, X, y, categorical_indicator):

        metafeature_values_hold = data2features(X, y, categorical_indicator)

        # adjust metadata features
        search_time_frozen = self.time_search_budget
        for metadata_i in range(len(self.metafeature_names_new)):
            param_id = self.feature_names.index(self.metafeature_names_new[metadata_i])
            self.random_configs[:, param_id] = np.ones((len(self.random_configs))) * metafeature_values_hold[0, metadata_i]

        # adjust constraints
        self.random_configs[:, self.feature_names.index('global_search_time_constraint')] = np.ones(
            (len(self.random_configs))) * search_time_frozen
        self.random_configs[:, self.feature_names.index('global_evaluation_time_constraint')] = np.ones(
            (len(self.random_configs))) * int(0.1 * search_time_frozen)
        self.random_configs[:, self.feature_names.index('training_time_constraint')] = np.ones(
            (len(self.random_configs))) * ifNull(self.training_time_limit, constant_value=search_time_frozen)
        self.random_configs[:, self.feature_names.index('inference_time_constraint')] = np.ones(
            (len(self.random_configs))) * ifNull(self.inference_time_limit, constant_value=60)
        self.random_configs[:, self.feature_names.index('pipeline_size_constraint')] = np.ones(
            (len(self.random_configs))) * ifNull(self.pipeline_size_limit, constant_value=350000000)
        self.random_configs[:, self.feature_names.index('fairness_constraint')] = np.ones((len(self.random_configs))) * ifNull(
            self.fairness_limit, constant_value=0.0)

        # adjust transformed constraints
        hold_out_fraction = self.feature_names.index('hold_out_fraction')
        self.random_configs[:, self.feature_names.index('hold_out_test_instances')] = np.multiply(
            np.multiply(hold_out_fraction, self.random_configs[:, self.feature_names.index('NumberOfInstances')]),
            self.random_configs[:, self.feature_names.index('sample_fraction')])
        self.random_configs[:, self.feature_names.index('hold_out_training_instances')] = np.multiply(
            np.multiply((1.0 - hold_out_fraction), self.random_configs[:, self.feature_names.index('NumberOfInstances')]),
            self.random_configs[:, self.feature_names.index('sample_fraction')])
        self.random_configs[:, self.feature_names.index('sampled_instances')] = self.random_configs[:,
                                                                      self.feature_names.index('NumberOfInstances')]
        self.random_configs[:, self.feature_names.index('log_search_time_constraint')] = np.ones(
            (len(self.random_configs))) * np.log(search_time_frozen)
        self.random_configs[:, self.feature_names.index('log_evaluation_time_constraint')] = np.ones(
            (len(self.random_configs))) * np.log(int(0.1 * search_time_frozen))
        self.random_configs[:, self.feature_names.index('log_sampled_instances')] = np.log(
            self.random_configs[:, self.feature_names.index('NumberOfInstances')])

        predictions = np.zeros((len(self.random_configs),))
        for model_success_i in range(len(self.model_success_list)):
            model_success = self.model_success_list[model_success_i]
            predictions_curr = model_success.predict_proba(self.random_configs)[:, 1]
            predictions += predictions_curr

        best_id = np.argmax(predictions)

        features = self.random_configs[best_id]

        gen = SpaceGenerator()
        self.space = gen.generate_params()
        self.space.prune_space_from_features(features, self.feature_names)

        evaluation_time = int(features[self.feature_names.index('global_evaluation_time_constraint')])

        cv = 1
        number_of_cvs = 1
        hold_out_fraction = features[self.feature_names.index('hold_out_fraction')]

        sample_fraction = 1.0

        ensemble_size = 50
        if int(features[self.feature_names.index('use_ensemble')]) == 0:
            ensemble_size = 1

        # use_incremental_data = True
        use_incremental_data = int(features[self.feature_names.index('use_incremental_data')]) == 1

        shuffle_validation = int(features[self.feature_names.index('shuffle_validation')]) == 1

        self.search = AutoEnsembleML(cv=cv,
                                number_of_cvs=number_of_cvs,
                                n_jobs=1,
                                evaluation_budget=evaluation_time,
                                time_search_budget=search_time_frozen,
                                space=self.space,
                                main_memory_budget_gb=500.0,
                                differential_privacy_epsilon=None,
                                hold_out_fraction=hold_out_fraction,
                                sample_fraction=sample_fraction,
                                training_time_limit=self.training_time_limit,
                                inference_time_limit=self.inference_time_limit,
                                pipeline_size_limit=self.pipeline_size_limit,
                                fairness_limit=self.fairness_limit,
                                fairness_group_id=self.fairness_group_id,
                                max_ensemble_models=ensemble_size,
                                use_incremental_data=use_incremental_data,
                                shuffle_validation=shuffle_validation
                                )


        self.search.fit(X, y, categorical_indicator=categorical_indicator, scorer=self.scorer)

        try:
            self.search.ensemble(X, y)
        except:
            pass

    def predict(self, X):
        return self.search.ensemble_predict(X)

    def print_space(self):
        from anytree import RenderTree

        for pre, _, node in RenderTree(self.space.parameter_tree):
            if node.status:
                print("%s%s" % (pre, node.name))

if __name__ == "__main__":
    import openml
    import sklearn

    dataset = openml.datasets.get_dataset(31)  # 51

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, stratify=y, train_size=0.6)

    caml = CAML(time_search_budget=60, inference_time_limit=0.001)
    caml.fit(X_train, y_train, categorical_indicator)

    caml.print_space()

    y_pred = caml.predict(X_test)
    print('test balanced accuracy: ' + str(balanced_accuracy_score(y_test, y_pred)))
