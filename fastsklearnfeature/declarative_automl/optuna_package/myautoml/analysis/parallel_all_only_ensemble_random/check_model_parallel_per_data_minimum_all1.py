import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.ensemble.AutoEnsembleSuccessive import \
    MyAutoML as AutoEnsembleML

import optuna
from sklearn.metrics import make_scorer
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import optimize_accuracy_under_minimal_sample_ensemble
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import utils_run_AutoML_ensemble_from_features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import ifNull
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.util_classes_new import ConstraintEvaluation, ConstraintRun, space2str
from anytree import RenderTree
import argparse
import openml
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.my_global_vars as mp_global
from sklearn.metrics import balanced_accuracy_score
import time
from optuna.samplers import NSGAIISampler
import getpass
import numpy as np

openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/' + getpass.getuser() + '/phd2/cache_openml'

my_scorer = make_scorer(balanced_accuracy_score)

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="OpenML datatset ID", type=int)
parser.add_argument("--outputname", "-o", help="Name of the output file")
args = parser.parse_args()
print(args.dataset)

#args.dataset = 168794
#args.outputname = 'testtest'



print(args)

memory_budget = 500.0
privacy = None

for test_holdout_dataset_id in [args.dataset]:

    X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data('data', randomstate=42, task_id=test_holdout_dataset_id)
    metafeature_values_hold = data2features(X_train_hold, y_train_hold, categorical_indicator_hold)

    model_success = pickle.load(open('/home/' + getpass.getuser() + '/data/my_temp/my_great_model_compare_scaled.p', "rb"))
    #random_configs = np.array(pickle.load(open('/home/' + getpass.getuser() + '/data/my_temp/randomconfigs.p', "rb"))['random_configs'])
    random_configs = np.array(pickle.load(open('/home/' + getpass.getuser() + '/data/my_temp/randomconfigs.p', "rb")))

    #model_success = pickle.load(open('/home/felix/phd2/dec_automl/okt25_al_opt_models/my_great_model_compare_scaled7.p', "rb"))
    #random_configs = np.array(pickle.load(open('/home/felix/phd2/dec_automl/oct25_random_configs/randomconfigs.p', "rb"))['random_configs'])

    #random_configs = np.array(pickle.load(open('/home/felix/phd2/dec_automl/okt22_al_1week_m4/felix_X_compare_scaled.p', "rb")))

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

    #plot_most_important_features(model, feature_names, k=len(feature_names))

    dynamic_approach = []

    new_constraint_evaluation_dynamic_all = []

    training_time_limit = None
    inference_time_limit = None
    pipeline_size_limit = None
    fairness_limit = None

    #for minutes_to_search in [10, 30, 1*60, 5*60, 10*60, 60*60]:#[1, 5]:#range(1, 6):
    for minutes_to_search in [10, 30, 1*60, 5*60, 10*60, 60*60]:

        current_dynamic = []

        search_time_frozen = minutes_to_search #* 60

        new_constraint_evaluation_dynamic = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                 constraint={'search_time': minutes_to_search},
                                                                 system_def='dynamic')
        for repeat in range(10):

            start_probing = time.time()

            ##fix random_configs

            #adjust metadata features
            for metadata_i in range(len(metafeature_names_new)):
                param_id = feature_names.index(metafeature_names_new[metadata_i])
                random_configs[:, param_id] = np.ones((len(random_configs))) * metafeature_values_hold[0, metadata_i]

            #adjust constraints
            random_configs[:, feature_names.index('global_search_time_constraint')] = np.ones((len(random_configs))) * search_time_frozen
            random_configs[:, feature_names.index('global_evaluation_time_constraint')] = np.ones((len(random_configs))) * int(0.1 * search_time_frozen)
            random_configs[:, feature_names.index('training_time_constraint')] = np.ones((len(random_configs))) * ifNull(training_time_limit, constant_value=search_time_frozen)
            random_configs[:, feature_names.index('inference_time_constraint')] = np.ones((len(random_configs))) * ifNull(inference_time_limit, constant_value=60)
            random_configs[:, feature_names.index('pipeline_size_constraint')] = np.ones((len(random_configs))) * ifNull(pipeline_size_limit, constant_value=350000000)
            random_configs[:, feature_names.index('fairness_constraint')] = np.ones((len(random_configs))) * ifNull(fairness_limit, constant_value=0.0)

            #adjust transformed constraints
            hold_out_fraction = feature_names.index('hold_out_fraction')
            random_configs[:, feature_names.index('hold_out_test_instances')] = np.multiply(np.multiply(hold_out_fraction, random_configs[:, feature_names.index('NumberOfInstances')]), random_configs[:, feature_names.index('sample_fraction')])
            random_configs[:, feature_names.index('hold_out_training_instances')] =  np.multiply(np.multiply((1.0 - hold_out_fraction), random_configs[:, feature_names.index('NumberOfInstances')]), random_configs[:, feature_names.index('sample_fraction')])
            random_configs[:, feature_names.index('sampled_instances')] = random_configs[:, feature_names.index('NumberOfInstances')]
            random_configs[:, feature_names.index('log_search_time_constraint')] = np.ones((len(random_configs))) * np.log(search_time_frozen)
            random_configs[:, feature_names.index('log_evaluation_time_constraint')] = np.ones((len(random_configs))) * np.log(int(0.1 * search_time_frozen))
            random_configs[:, feature_names.index('log_sampled_instances')]= np.log(random_configs[:, feature_names.index('NumberOfInstances')])



            best_id = np.argmax(model_success.predict(random_configs))
            print('max:' + str(np.max(model_success.predict(random_configs))))

            space = None
            try:
                result = None
                search_dynamic = None
                if True:#mp_global.study_prune.best_trial.value > 0.5: #TODO
                    result, search_dynamic, space = utils_run_AutoML_ensemble_from_features(X_train=X_train_hold,
                                                                 X_test=X_test_hold,
                                                                 y_train=y_train_hold,
                                                                 y_test=y_test_hold,
                                                                 categorical_indicator=categorical_indicator_hold,
                                                                 my_scorer=my_scorer,
                                                                 search_time=search_time_frozen,
                                                                 memory_limit=memory_budget,
                                                                 privacy_limit=privacy,
                                                                 features=random_configs[best_id],
                                                                 feature_names=feature_names
                                                 )

                new_constraint_evaluation_dynamic.append(ConstraintRun(space_str=space2str(space.parameter_tree), params=random_configs[best_id], test_score=result, estimated_score=0.0))
            except:
                result = 0
                new_constraint_evaluation_dynamic.append(ConstraintRun(space_str='', params=random_configs[best_id], test_score=result, estimated_score=0.0))

            print("test result: " + str(result))
            current_dynamic.append(result)

            print('dynamic: ' + str(current_dynamic))

        dynamic_approach.append(current_dynamic)

        new_constraint_evaluation_dynamic_all.append(new_constraint_evaluation_dynamic)

        print('dynamic: ' + str(dynamic_approach))


    results_dict_log = {}
    results_dict_log['dynamic'] = new_constraint_evaluation_dynamic_all

    pickle.dump(results_dict_log, open('/home/' + getpass.getuser() + '/data/automl_runs/log_' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))

    results_dict = {}
    results_dict['dynamic'] = dynamic_approach

    pickle.dump(results_dict, open('/home/' + getpass.getuser() + '/data/automl_runs/' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))