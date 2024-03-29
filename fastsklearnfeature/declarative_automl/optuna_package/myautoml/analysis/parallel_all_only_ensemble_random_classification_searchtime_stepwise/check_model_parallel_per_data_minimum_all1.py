import optuna
from sklearn.metrics import make_scorer
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import utils_run_AutoML_ensemble_from_features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import ifNull
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.util_classes_new import ConstraintEvaluation, ConstraintRun, space2str
from anytree import RenderTree
import argparse
import openml
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.my_global_vars as mp_global
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import getpass

openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'

my_scorer = make_scorer(balanced_accuracy_score)

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="OpenML datatset ID", type=int)
parser.add_argument("--outputname", "-o", help="Name of the output file")
args = parser.parse_args()
print(args.dataset)

#args.dataset = 448
#args.outputname = 'testtest'


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

metafeature_names_new = ['ClassEntropy', 'NumSymbols', 'SymbolsSum', 'SymbolsSTD', 'SymbolsMean', 'SymbolsMax',
                         'SymbolsMin', 'ClassOccurences', 'ClassProbabilitySTD', 'ClassProbabilityMean',
                         'ClassProbabilityMax', 'ClassProbabilityMin', 'InverseDatasetRatio', 'DatasetRatio',
                         'RatioNominalToNumerical', 'RatioNumericalToNominal', 'NumberOfCategoricalFeatures',
                         'NumberOfNumericFeatures', 'MissingValues', 'NumberOfMissingValues',
                         'NumberOfFeaturesWithMissingValues', 'NumberOfInstancesWithMissingValues',
                         'NumberOfFeatures', 'NumberOfClasses', 'NumberOfInstances', 'LogInverseDatasetRatio',
                         'LogDatasetRatio', 'PercentageOfMissingValues', 'PercentageOfFeaturesWithMissingValues',
                         'PercentageOfInstancesWithMissingValues', 'LogNumberOfFeatures', 'LogNumberOfInstances']

_, feature_names = get_feature_names(my_list_constraints)

random_configs = np.array(pickle.load(open('/home/' + getpass.getuser() + '/data/my_temp/randomconfigs.p', "rb")))


print(args)

memory_budget = 500.0
privacy = None

for test_holdout_dataset_id in [args.dataset]:

    X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data('data', randomstate=42, task_id=test_holdout_dataset_id)
    metafeature_values_hold = data2features(X_train_hold, y_train_hold, categorical_indicator_hold)

    #X_surrogate = pickle.load(open('/home/neutatz/data/my_temp/felix_X_compare_scaled.p', "rb"))
    #y_surrogate = pickle.load(open('/home/neutatz/data/my_temp/felix_y_compare_scaled.p', "rb"))


    dynamic_approach = []

    new_constraint_evaluation_dynamic_all = []

    training_time_limit = None
    inference_time_limit = None
    pipeline_size_limit = None
    fairness_limit = None

    for days_surrogate in [7,8,9,10,12,13,14]:#[1, 2, 3, 4, 5, 6, 7]:#[1, 5, 10, 60]:#range(1, 6):

        #model_success = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=1)
        #model_success.fit(X_surrogate[0: int((days_surrogate / 7.0) * len(X_surrogate)), :], y_surrogate[0: int((days_surrogate / 7.0) * len(X_surrogate))])

        model_success_list = []
        # discrete_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        discrete_list = [0.1]
        for discrete in discrete_list:
            model_success_list.append(pickle.load(open(
                '/home/' + getpass.getuser() + '/data/my_temp/my_great_model_compare_scaled' + str(days_surrogate) + "_" + str(
                    discrete) + ".p", "rb")))

        current_dynamic = []

        minutes_to_search = 5
        search_time_frozen = minutes_to_search * 60

        new_constraint_evaluation_dynamic = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                 constraint={'days_surrogate': days_surrogate},
                                                                 system_def='dynamic')
        for repeat in range(10):

            ##fix random_configs

            # adjust metadata features
            for metadata_i in range(len(metafeature_names_new)):
                param_id = feature_names.index(metafeature_names_new[metadata_i])
                random_configs[:, param_id] = np.ones((len(random_configs))) * metafeature_values_hold[0, metadata_i]

            # adjust constraints
            random_configs[:, feature_names.index('global_search_time_constraint')] = np.ones(
                (len(random_configs))) * search_time_frozen
            random_configs[:, feature_names.index('global_evaluation_time_constraint')] = np.ones(
                (len(random_configs))) * int(0.1 * search_time_frozen)
            random_configs[:, feature_names.index('training_time_constraint')] = np.ones(
                (len(random_configs))) * ifNull(training_time_limit, constant_value=search_time_frozen)
            random_configs[:, feature_names.index('inference_time_constraint')] = np.ones(
                (len(random_configs))) * ifNull(inference_time_limit, constant_value=60)
            random_configs[:, feature_names.index('pipeline_size_constraint')] = np.ones(
                (len(random_configs))) * ifNull(pipeline_size_limit, constant_value=350000000)
            random_configs[:, feature_names.index('fairness_constraint')] = np.ones((len(random_configs))) * ifNull(
                fairness_limit, constant_value=0.0)

            # adjust transformed constraints
            hold_out_fraction = feature_names.index('hold_out_fraction')
            random_configs[:, feature_names.index('hold_out_test_instances')] = np.multiply(
                np.multiply(hold_out_fraction, random_configs[:, feature_names.index('NumberOfInstances')]),
                random_configs[:, feature_names.index('sample_fraction')])
            random_configs[:, feature_names.index('hold_out_training_instances')] = np.multiply(
                np.multiply((1.0 - hold_out_fraction), random_configs[:, feature_names.index('NumberOfInstances')]),
                random_configs[:, feature_names.index('sample_fraction')])
            random_configs[:, feature_names.index('sampled_instances')] = random_configs[:,
                                                                          feature_names.index('NumberOfInstances')]
            random_configs[:, feature_names.index('log_search_time_constraint')] = np.ones(
                (len(random_configs))) * np.log(search_time_frozen)
            random_configs[:, feature_names.index('log_evaluation_time_constraint')] = np.ones(
                (len(random_configs))) * np.log(int(0.1 * search_time_frozen))
            random_configs[:, feature_names.index('log_sampled_instances')] = np.log(
                random_configs[:, feature_names.index('NumberOfInstances')])

            predictions = np.zeros((len(random_configs),))
            for model_success_i in range(len(model_success_list)):
                model_success = model_success_list[model_success_i]
                print(model_success.classes_)
                discrete = discrete_list[model_success_i]
                predictions_curr = model_success.predict_proba(random_configs)[:, 1]
                print(predictions_curr.shape)
                predictions += predictions_curr

            best_id = np.argmax(predictions)
            print('best_id: ' + str(best_id))
            print('max:' + str(np.max(predictions)))
            print('avg:' + str(np.average(predictions)))

            try:
                result = None
                search_dynamic = None
                if True:  # mp_global.study_prune.best_trial.value > 0.5: #TODO
                    result, search_dynamic, space = utils_run_AutoML_ensemble_from_features(X_train=X_train_hold,
                                                                                            X_test=X_test_hold,
                                                                                            y_train=y_train_hold,
                                                                                            y_test=y_test_hold,
                                                                                            categorical_indicator=categorical_indicator_hold,
                                                                                            my_scorer=my_scorer,
                                                                                            search_time=search_time_frozen,
                                                                                            memory_limit=memory_budget,
                                                                                            privacy_limit=privacy,
                                                                                            features=random_configs[
                                                                                                best_id],
                                                                                            feature_names=feature_names
                                                                                            )

                new_constraint_evaluation_dynamic.append(
                    ConstraintRun(space_str=space2str(space.parameter_tree), params=random_configs[best_id],
                                  test_score=result, estimated_score=0.0))
            except:
                result = 0
                new_constraint_evaluation_dynamic.append(
                    ConstraintRun(space_str='', params=random_configs[best_id], test_score=result, estimated_score=0.0))

            print("test result: " + str(result))
            current_dynamic.append(result)

            print('dynamic: ' + str(current_dynamic))

        dynamic_approach.append(current_dynamic)

        new_constraint_evaluation_dynamic_all.append(new_constraint_evaluation_dynamic)

        print('dynamic: ' + str(dynamic_approach))


    results_dict_log = {}
    results_dict_log['dynamic'] = new_constraint_evaluation_dynamic_all

    pickle.dump(results_dict_log, open('/home/neutatz/data/automl_runs/log_' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))

    results_dict = {}
    results_dict['dynamic'] = dynamic_approach

    pickle.dump(results_dict, open('/home/neutatz/data/automl_runs/' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))