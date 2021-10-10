from autosklearn.metalearning.metafeatures.metafeatures import calculate_all_metafeatures_with_labels
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.MyAutoMLProcessClassBalance import MyAutoML

import optuna
import time
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import openml
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
import copy
from optuna.trial import FrozenTrial
from anytree import RenderTree
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle
import operator
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.feature_transformation.FeatureTransformations import FeatureTransformations


from fastsklearnfeature.declarative_automl.optuna_package.myautoml.feature_transformation.FeatureTransformationsNew import FeatureTransformations as FT

import multiprocessing as mp
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.my_global_vars as mp_global


metafeature_names_new = ['ClassEntropy', 'ClassProbabilityMax', 'ClassProbabilityMean', 'ClassProbabilityMin', 'ClassProbabilitySTD',
     'DatasetRatio', 'InverseDatasetRatio', 'LogDatasetRatio', 'LogInverseDatasetRatio', 'LogNumberOfFeatures',
     'LogNumberOfInstances', 'NumberOfCategoricalFeatures', 'NumberOfClasses', 'NumberOfFeatures',
     'NumberOfFeaturesWithMissingValues', 'NumberOfInstances', 'NumberOfInstancesWithMissingValues',
     'NumberOfMissingValues', 'NumberOfNumericFeatures', 'PercentageOfFeaturesWithMissingValues',
     'PercentageOfInstancesWithMissingValues', 'PercentageOfMissingValues', 'RatioNominalToNumerical',
     'RatioNumericalToNominal', 'SymbolsMax', 'SymbolsMean', 'SymbolsMin', 'SymbolsSTD', 'SymbolsSum']




mgen = SpaceGenerator()
mspace = mgen.generate_params()

my_list = list(mspace.name2node.keys())
my_list.sort()

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(mp.pool.Pool):
    Process = NoDaemonProcess


def get_feature_names(my_list_constraints=None):
    feature_names = copy.deepcopy(my_list)
    feature_names.extend(copy.deepcopy(my_list_constraints))
    feature_names.extend(copy.deepcopy(metafeature_names_new))

    feature_names_new = FeatureTransformations().get_new_feature_names(feature_names)
    return feature_names, feature_names_new

def get_feature_names2(my_list_constraints=None):
    feature_names = copy.deepcopy(my_list)

    new_list_names_f = list()
    mspace.generate_additional_features_v2_name(start=True, sum_list=new_list_names_f)
    feature_names.extend(copy.deepcopy(new_list_names_f))
    feature_names.extend(copy.deepcopy(my_list_constraints))
    feature_names.extend(copy.deepcopy(metafeature_names_new))

    feature_names_new = FeatureTransformations().get_new_feature_names(feature_names)
    return feature_names, feature_names_new

def get_feature_names_new(my_list_constraints=None):
    feature_names = []
    feature_names.extend(copy.deepcopy(my_list_constraints))
    feature_names.extend(copy.deepcopy(metafeature_names_new))

    feature_names_new = FT().get_new_feature_names(feature_names)
    return feature_names, feature_names_new

def data2features(X_train, y_train, categorical_indicator):
    metafeatures = calculate_all_metafeatures_with_labels(X_train, y_train, categorical=categorical_indicator,
                                                          dataset_name='data')

    metafeature_values = np.zeros((1, len(metafeature_names_new)))
    for m_i in range(len(metafeature_names_new)):
        try:
            metafeature_values[0, m_i] = metafeatures[metafeature_names_new[m_i]].value
        except:
            pass
    return metafeature_values


def get_data(data_id, randomstate=42, task_id=None):
    task = None
    if type(task_id) != type(None):
        task = openml.tasks.get_task(task_id)
        data_id = task.get_dataset().dataset_id

    dataset = openml.datasets.get_dataset(dataset_id=data_id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array",
        target=dataset.default_target_attribute
    )

    X_train = None
    X_test = None
    y_train = None
    y_test = None
    if type(task_id) != type(None):
        train_indices, test_indices = task.get_train_test_split_indices()
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
    else:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,
                                                                                    y,
                                                                                    random_state=randomstate,
                                                                                    stratify=y,
                                                                                    train_size=0.66)

    return X_train, X_test, y_train, y_test, categorical_indicator, attribute_names


def plot_most_important_features(rf_random, names_features, title='importance', verbose=True, k=25):

    assert len(rf_random.feature_importances_) == len(names_features), 'mismatch'

    importances = {}
    for name_i in range(len(names_features)):
        importances[names_features[name_i]] = rf_random.feature_importances_[name_i]

    sorted_x = sorted(importances.items(), key=operator.itemgetter(1), reverse=True)

    labels = []
    score = []
    t = 0
    for key, value in sorted_x:
        labels.append(key)
        score.append(value)
        print(key + ': ' + str(value))
        t += 1
        if t == k:
            break

    ind = np.arange(len(score))
    plt.bar(ind, score, align='center', alpha=0.5)
    #plt.yticks(ind, labels)

    plt.xticks(ind, labels, rotation='vertical')

    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.6)

    if verbose:
        plt.show()
    else:
        plt.savefig('/tmp/feature_importance.png')
        plt.clf()

def merge_features(my_list_constraints_values, metafeature_values):
    tuple_constraints = np.zeros((1, len(my_list_constraints_values)))
    t = 0
    for constraint_i in range(len(my_list_constraints_values)):
        tuple_constraints[t, constraint_i] = my_list_constraints_values[constraint_i]
    return np.hstack((tuple_constraints, metafeature_values))

def space2features(space, my_list_constraints_values, metafeature_values):
    tuple_param = np.zeros((1, len(my_list)))
    tuple_constraints = np.zeros((1, len(my_list_constraints_values)))
    t = 0
    for parameter_i in range(len(my_list)):
        tuple_param[t, parameter_i] = space.name2node[my_list[parameter_i]].status

    for constraint_i in range(len(my_list_constraints_values)):
        tuple_constraints[t, constraint_i] = my_list_constraints_values[constraint_i] #current_trial.params[my_list_constraints[constraint_i]]

    return np.hstack((tuple_param, tuple_constraints, metafeature_values))

def space2features_v2(space, my_list_constraints_values, metafeature_values):
    tuple_param = np.zeros((1, len(my_list)))
    tuple_constraints = np.zeros((1, len(my_list_constraints_values)))
    t = 0
    for parameter_i in range(len(my_list)):
        tuple_param[t, parameter_i] = space.name2node[my_list[parameter_i]].status

    new_list = list()
    space.generate_additional_features_v2(start=True, sum_list=new_list)
    tuple_param_sum = np.zeros((1, len(new_list)))
    for parameter_sum_i in range(len(new_list)):
        tuple_param_sum[t, parameter_sum_i] = new_list[parameter_sum_i]

    for constraint_i in range(len(my_list_constraints_values)):
        tuple_constraints[t, constraint_i] = my_list_constraints_values[constraint_i] #current_trial.params[my_list_constraints[constraint_i]]

    return np.hstack((tuple_param, tuple_param_sum, tuple_constraints, metafeature_values))


def predict_range(model, X):
    y_pred = model.predict(X)

    #y_pred[y_pred > 1.0] = 1.0
    #y_pred[y_pred < 0.0] = 0.0
    return y_pred

def ifNull(value, constant_value=0):
    if type(value) == type(None):
        return constant_value
    else:
        return value








def generate_features(trial, metafeature_values_hold, search_time,
                                        memory_limit=10,
                                        privacy_limit=None,
                                        evaluation_time=None,
                                        hold_out_fraction=None,
                                        training_time_limit=None,
                                        inference_time_limit=None,
                                        pipeline_size_limit=None,
                                        tune_space=False,
                                        save_data=True
                                        ):
    try:
        gen = SpaceGenerator()
        space = gen.generate_params()
        if tune_space:
            space.sample_parameters(trial)

        if type(evaluation_time) == type(None):
            evaluation_time = search_time
            if trial.suggest_categorical('use_evaluation_time_constraint', [True, False]):
                evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)
        else:
            trial.set_user_attr('evaluation_time', evaluation_time)

        # how many cvs should be used
        cv = 1
        number_of_cvs = 1
        if type(hold_out_fraction) == type(None):
            hold_out_fraction = None
            if trial.suggest_categorical('use_hold_out', [True, False]):
                hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0, 1)
            else:
                cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
                number_of_cvs = 1
                if trial.suggest_categorical('use_multiple_cvs', [True, False]):
                    number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)
        else:
            trial.set_user_attr('hold_out_fraction', hold_out_fraction)


        sample_fraction = 1.0
        if trial.suggest_categorical('use_sampling', [True, False]):
            sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)



        my_list_constraints_values = [search_time,
                                      evaluation_time,
                                      memory_limit,
                                      cv,
                                      number_of_cvs,
                                      ifNull(privacy_limit, constant_value=1000),
                                      ifNull(hold_out_fraction),
                                      sample_fraction,
                                      ifNull(training_time_limit, constant_value=search_time),
                                      ifNull(inference_time_limit, constant_value=60),
                                      ifNull(pipeline_size_limit, constant_value=350000000),
                                      ]

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
                               'pipeline_size_constraint']

        features = space2features(space, my_list_constraints_values, metafeature_values_hold)
        feature_names, _ = get_feature_names(my_list_constraints)
        features = FeatureTransformations().fit(features).transform(features, feature_names=feature_names)

        if save_data:
            trial.set_user_attr('features', features)

        if not save_data:
            return features, space
        else:
            return features
    except Exception as e:
        return None


def generate_features_minimum(trial, metafeature_values_hold, search_time,
                                        memory_limit=10,
                                        privacy_limit=None,
                                        evaluation_time=None,
                                        hold_out_fraction=None,
                                        training_time_limit=None,
                                        inference_time_limit=None,
                                        pipeline_size_limit=None,
                                        tune_space=False,
                                        save_data=True
                                        ):
    try:
        gen = SpaceGenerator()
        space = gen.generate_params()


        evaluation_time = int(0.1 * search_time)
        if trial.suggest_categorical('use_evaluation_time_constraint', [False]):
            evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)
        trial.set_user_attr('evaluation_time', evaluation_time)

        # how many cvs should be used
        cv = 1
        number_of_cvs = 1
        if type(hold_out_fraction) == type(None):
            hold_out_fraction = None
            if trial.suggest_categorical('use_hold_out', [True]):
                hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0, 1)
            else:
                cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
                number_of_cvs = 1
                if trial.suggest_categorical('use_multiple_cvs', [True, False]):
                    number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)
        else:
            trial.set_user_attr('hold_out_fraction', hold_out_fraction)


        sample_fraction = 1.0
        if trial.suggest_categorical('use_sampling', [False]):
            sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)



        my_list_constraints_values = [search_time,
                                      evaluation_time,
                                      memory_limit,
                                      cv,
                                      number_of_cvs,
                                      ifNull(privacy_limit, constant_value=1000),
                                      ifNull(hold_out_fraction),
                                      sample_fraction,
                                      ifNull(training_time_limit, constant_value=search_time),
                                      ifNull(inference_time_limit, constant_value=60),
                                      ifNull(pipeline_size_limit, constant_value=350000000),
                                      ]

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
                               'pipeline_size_constraint']

        features = space2features(space, my_list_constraints_values, metafeature_values_hold)
        feature_names, _ = get_feature_names(my_list_constraints)
        features = FeatureTransformations().fit(features).transform(features, feature_names=feature_names)

        if save_data:
            trial.set_user_attr('features', features)

        if not save_data:
            return features, space
        else:
            return features
    except Exception as e:
        return None


def generate_features_minimum_sample(trial, metafeature_values_hold, search_time,
                                        memory_limit=10,
                                        privacy_limit=None,
                                        evaluation_time=None,
                                        hold_out_fraction=None,
                                        training_time_limit=None,
                                        inference_time_limit=None,
                                        pipeline_size_limit=None,
                                        tune_space=False,
                                        save_data=True,
                                        tune_eval_time=False,
                                        tune_val_fraction=False,
                                        tune_cv=False
                                        ):
    try:
        gen = SpaceGenerator()
        space = gen.generate_params()

        if tune_space:
            if trial.suggest_categorical('use_space_search_param', [True]):
                space.sample_parameters(trial)


        evaluation_time = int(0.1 * search_time)
        cat_eval_list = [False]
        if tune_eval_time:
            cat_eval_list = [False, True]
        if trial.suggest_categorical('use_evaluation_time_constraint', cat_eval_list):
            evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)
        trial.set_user_attr('evaluation_time', evaluation_time)

        # how many cvs should be used
        cv = 1
        number_of_cvs = 1
        if type(hold_out_fraction) == type(None):

            cat_holdout_list = [True]
            if tune_cv:
                cat_holdout_list = [True, False]

            hold_out_fraction = None
            if trial.suggest_categorical('use_hold_out', cat_holdout_list):
                if tune_val_fraction:
                    hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0.0, 1.0)
                else:
                    hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0.33, 0.33)
            else:
                cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
                number_of_cvs = 1
                if trial.suggest_categorical('use_multiple_cvs', [True, False]):
                    number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)
        else:
            trial.set_user_attr('hold_out_fraction', hold_out_fraction)


        sample_fraction = 1.0
        if trial.suggest_categorical('use_sampling', [True]):
            sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)

        hold_out_fraction_feature = hold_out_fraction
        if cv * number_of_cvs > 1:
            hold_out_fraction_feature = (100.0 / cv) / 100.0

        my_list_constraints_values = [search_time,
                                      evaluation_time,
                                      memory_limit,
                                      cv,
                                      number_of_cvs,
                                      ifNull(privacy_limit, constant_value=1000),
                                      hold_out_fraction_feature,
                                      sample_fraction,
                                      ifNull(training_time_limit, constant_value=search_time),
                                      ifNull(inference_time_limit, constant_value=60),
                                      ifNull(pipeline_size_limit, constant_value=350000000),
                                      ]

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
                               'pipeline_size_constraint']

        features = space2features(space, my_list_constraints_values, metafeature_values_hold)
        feature_names, _ = get_feature_names(my_list_constraints)
        features = FeatureTransformations().fit(features).transform(features, feature_names=feature_names)

        if save_data:
            trial.set_user_attr('features', features)

        if not save_data:
            return features, space
        else:
            return features
    except Exception as e:
        return None


def optimize_accuracy_under_minimal(trial, metafeature_values_hold, search_time, model_success,
                                        memory_limit=10,
                                        privacy_limit=None,
                                        evaluation_time=None,
                                        hold_out_fraction=None,
                                        training_time_limit=None,
                                        inference_time_limit=None,
                                        pipeline_size_limit=None,
                                        tune_space=False
                                        ):
    features, space = generate_features_minimum(trial, metafeature_values_hold, search_time,
                      memory_limit=memory_limit,
                      privacy_limit=privacy_limit,
                      evaluation_time=evaluation_time,
                      hold_out_fraction=hold_out_fraction,
                      training_time_limit=training_time_limit,
                      inference_time_limit=inference_time_limit,
                      pipeline_size_limit=pipeline_size_limit,
                      tune_space=tune_space,
                      save_data=False
                      )

    success_val = predict_range(model_success, features)

    if trial.number == 0 or success_val > mp_global.study_prune.best_trial.value:
        trial.set_user_attr('space', copy.deepcopy(space))

    return success_val

def optimize_accuracy_under_minimal_sample(trial, metafeature_values_hold, search_time, model_success,
                                        memory_limit=10,
                                        privacy_limit=None,
                                        evaluation_time=None,
                                        hold_out_fraction=None,
                                        training_time_limit=None,
                                        inference_time_limit=None,
                                        pipeline_size_limit=None,
                                        tune_space=False,
                                        tune_eval_time=False,
                                        tune_val_fraction=False,
                                        tune_cv=False

                                        ):
    features, space = generate_features_minimum_sample(trial, metafeature_values_hold, search_time,
                      memory_limit=memory_limit,
                      privacy_limit=privacy_limit,
                      evaluation_time=evaluation_time,
                      hold_out_fraction=hold_out_fraction,
                      training_time_limit=training_time_limit,
                      inference_time_limit=inference_time_limit,
                      pipeline_size_limit=pipeline_size_limit,
                      tune_space=tune_space,
                      save_data=False,
                      tune_eval_time=tune_eval_time,
                      tune_val_fraction=tune_val_fraction,
                      tune_cv=tune_cv
                      )

    success_val = predict_range(model_success, features)

    if trial.number == 0 or success_val > mp_global.study_prune.best_trial.value:
        trial.set_user_attr('space', copy.deepcopy(space))

    return success_val


def optimize_accuracy_under_constraints2(trial, metafeature_values_hold, search_time, model_success,
                                        memory_limit=10,
                                        privacy_limit=None,
                                        evaluation_time=None,
                                        hold_out_fraction=None,
                                        training_time_limit=None,
                                        inference_time_limit=None,
                                        pipeline_size_limit=None,
                                        tune_space=False
                                        ):
    features, space = generate_features(trial, metafeature_values_hold, search_time,
                      memory_limit=memory_limit,
                      privacy_limit=privacy_limit,
                      evaluation_time=evaluation_time,
                      hold_out_fraction=hold_out_fraction,
                      training_time_limit=training_time_limit,
                      inference_time_limit=inference_time_limit,
                      pipeline_size_limit=pipeline_size_limit,
                      tune_space=tune_space,
                      save_data=False
                      )

    success_val = predict_range(model_success, features)

    try:
        if success_val > mp_global.study_prune.best_trial.value:
            trial.set_user_attr('space', copy.deepcopy(space))
    except:
        pass

    return success_val

def batched_objective(x, model_success):
    print(x)
    return predict_range(model_success, x)


def optimize_accuracy_under_constraints_weights(trial, metafeature_values_hold, search_time, model_weights,
                                        memory_limit=10,
                                        privacy_limit=None,
                                        training_time_limit=None,
                                        inference_time_limit=None,
                                        pipeline_size_limit=None
                                        ):
    try:
        model_weight = 0
        if trial.suggest_categorical('use_model_weight', [True, False]):
            model_weight = trial.suggest_loguniform('model_weight', 0.0000001, 1000)

        number_trials = trial.suggest_int('number_trials', 10, 500, log=False)



        my_list_constraints_values = [search_time,
                                      memory_limit,
                                      ifNull(privacy_limit, constant_value=1000),
                                      ifNull(training_time_limit, constant_value=search_time),
                                      ifNull(inference_time_limit, constant_value=60),
                                      ifNull(pipeline_size_limit, constant_value=350000000),
                                      model_weight,
                                      number_trials
                                      ]

        my_list_constraints = ['global_search_time_constraint',
                               'global_memory_constraint',
                               'privacy',
                               'training_time_constraint',
                               'inference_time_constraint',
                               'pipeline_size_constraint',
                               'model_weight',
                               'number_trials']

        features = merge_features(my_list_constraints_values, metafeature_values_hold)
        feature_names, _ = get_feature_names_new(my_list_constraints)
        features = FT().fit(features).transform(features, feature_names=feature_names)
        trial.set_user_attr('features', features)

        return predict_range(model_weights, features)
    except Exception as e:
        print(str(e) + 'except dataset _ accuracy: ' + '\n\n')
        return 0.0


def generate_parameters(trial, total_search_time_minutes, my_openml_datasets, sample_data=True):
    # which constraints to use
    search_time = trial.suggest_int('global_search_time_constraint', 1, max(10, total_search_time_minutes), log=False) #* 60

    # how much time for each evaluation
    evaluation_time = search_time
    if trial.suggest_categorical('use_evaluation_time_constraint', [True, False]):
        evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)

    # how much memory is allowed
    memory_limit = 10
    if trial.suggest_categorical('use_search_memory_constraint', [True, False]):
        memory_limit = trial.suggest_loguniform('global_memory_constraint', 0.00000000000001, 10)

    # how much privacy is required
    privacy_limit = None
    if trial.suggest_categorical('use_privacy_constraint', [True, False]):
        privacy_limit = trial.suggest_loguniform('privacy_constraint', 0.0001, 10)

    training_time_limit = search_time
    if trial.suggest_categorical('use_training_time_constraint', [True, False]):
        training_time_limit = trial.suggest_loguniform('training_time_constraint', 0.005, search_time)

    inference_time_limit = 60
    if trial.suggest_categorical('use_inference_time_constraint', [True, False]):
        inference_time_limit = trial.suggest_loguniform('inference_time_constraint', 0.0004, 60)

    pipeline_size_limit = 350000000
    if trial.suggest_categorical('use_pipeline_size_constraint', [True, False]):
        pipeline_size_limit = trial.suggest_loguniform('pipeline_size_constraint', 2000, 350000000)


    # how many cvs should be used
    cv = 1
    number_of_cvs = 1
    hold_out_fraction = None
    if trial.suggest_categorical('use_hold_out', [True, False]):
        hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0, 1)
    else:
        cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
        number_of_cvs = 1
        if trial.suggest_categorical('use_multiple_cvs', [True, False]):
            number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)

    sample_fraction = 1.0
    if trial.suggest_categorical('use_sampling', [True, False]):
        sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)

    dataset_id = None
    if sample_data:
        dataset_id = trial.suggest_categorical('dataset_id', my_openml_datasets)

    return search_time, evaluation_time, memory_limit, privacy_limit, training_time_limit, inference_time_limit, pipeline_size_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, dataset_id




def generate_parameters_2constraints(trial, total_search_time_minutes, my_openml_datasets, sample_data=True):
    # which constraints to use
    search_time = trial.suggest_int('global_search_time_constraint', 1, max(10, total_search_time_minutes), log=False) #* 60

    # how much time for each evaluation
    evaluation_time = search_time
    if trial.suggest_categorical('use_evaluation_time_constraint', [True, False]):
        evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)

    # how much memory is allowed
    memory_limit = 10
    if trial.suggest_categorical('use_search_memory_constraint', [False]):
        memory_limit = trial.suggest_loguniform('global_memory_constraint', 0.00000000000001, 10)

    # how much privacy is required
    privacy_limit = None
    if trial.suggest_categorical('use_privacy_constraint', [False]):
        privacy_limit = trial.suggest_loguniform('privacy_constraint', 0.0001, 10)

    training_time_limit = search_time
    if trial.suggest_categorical('use_training_time_constraint', [False]):
        training_time_limit = trial.suggest_loguniform('training_time_constraint', 0.005, search_time)

    inference_time_limit = 60
    if trial.suggest_categorical('use_inference_time_constraint', [False]):
        inference_time_limit = trial.suggest_loguniform('inference_time_constraint', 0.0004, 60)

    pipeline_size_limit = 350000000
    if trial.suggest_categorical('use_pipeline_size_constraint', [True, False]):
        pipeline_size_limit = trial.suggest_loguniform('pipeline_size_constraint', 2000, 350000000)


    # how many cvs should be used
    cv = 1
    number_of_cvs = 1
    hold_out_fraction = None
    if trial.suggest_categorical('use_hold_out', [True, False]):
        hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0, 1)
    else:
        cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
        number_of_cvs = 1
        if trial.suggest_categorical('use_multiple_cvs', [True, False]):
            number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)

    sample_fraction = 1.0
    if trial.suggest_categorical('use_sampling', [True, False]):
        sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)

    dataset_id = None
    if sample_data:
        dataset_id = trial.suggest_categorical('dataset_id', my_openml_datasets)

    return search_time, evaluation_time, memory_limit, privacy_limit, training_time_limit, inference_time_limit, pipeline_size_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, dataset_id




def generate_parameters_minimal(trial, total_search_time_minutes, my_openml_datasets, sample_data=True):
    # which constraints to use
    search_time = trial.suggest_int('global_search_time_constraint', 10, max(10, total_search_time_minutes), log=False) #* 60

    # how much time for each evaluation
    evaluation_time = int(0.1 * search_time)
    if trial.suggest_categorical('use_evaluation_time_constraint', [False]):
        evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)

    # how much memory is allowed
    memory_limit = 10
    if trial.suggest_categorical('use_search_memory_constraint', [False]):
        memory_limit = trial.suggest_loguniform('global_memory_constraint', 0.00000000000001, 10)

    # how much privacy is required
    privacy_limit = None
    if trial.suggest_categorical('use_privacy_constraint', [False]):
        privacy_limit = trial.suggest_loguniform('privacy_constraint', 0.0001, 10)

    training_time_limit = search_time
    if trial.suggest_categorical('use_training_time_constraint', [False]):
        training_time_limit = trial.suggest_loguniform('training_time_constraint', 0.005, search_time)

    inference_time_limit = 60
    if trial.suggest_categorical('use_inference_time_constraint', [False]):
        inference_time_limit = trial.suggest_loguniform('inference_time_constraint', 0.0004, 60)

    pipeline_size_limit = 350000000
    if trial.suggest_categorical('use_pipeline_size_constraint', [False]):
        pipeline_size_limit = trial.suggest_loguniform('pipeline_size_constraint', 2000, 350000000)


    # how many cvs should be used
    cv = 1
    number_of_cvs = 1
    hold_out_fraction = None
    if trial.suggest_categorical('use_hold_out', [True]):
        hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0, 1)
    else:
        cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
        number_of_cvs = 1
        if trial.suggest_categorical('use_multiple_cvs', [True, False]):
            number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)

    sample_fraction = 1.0
    if trial.suggest_categorical('use_sampling', [False]):
        sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)

    dataset_id = None
    if sample_data:
        dataset_id = trial.suggest_categorical('dataset_id', my_openml_datasets)

    return search_time, evaluation_time, memory_limit, privacy_limit, training_time_limit, inference_time_limit, pipeline_size_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, dataset_id




def generate_parameters_minimal_sample(trial, total_search_time_minutes, my_openml_datasets, sample_data=True):
    # which constraints to use
    search_time = trial.suggest_int('global_search_time_constraint', 10, max(10, total_search_time_minutes), log=False) #* 60

    # how much time for each evaluation
    evaluation_time = int(0.1 * search_time)
    if trial.suggest_categorical('use_evaluation_time_constraint', [False]):
        evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)

    # how much memory is allowed
    memory_limit = 10
    if trial.suggest_categorical('use_search_memory_constraint', [False]):
        memory_limit = trial.suggest_loguniform('global_memory_constraint', 0.00000000000001, 10)

    # how much privacy is required
    privacy_limit = None
    if trial.suggest_categorical('use_privacy_constraint', [False]):
        privacy_limit = trial.suggest_loguniform('privacy_constraint', 0.0001, 10)

    training_time_limit = search_time
    if trial.suggest_categorical('use_training_time_constraint', [False]):
        training_time_limit = trial.suggest_loguniform('training_time_constraint', 0.005, search_time)

    inference_time_limit = 60
    if trial.suggest_categorical('use_inference_time_constraint', [False]):
        inference_time_limit = trial.suggest_loguniform('inference_time_constraint', 0.0004, 60)

    pipeline_size_limit = 350000000
    if trial.suggest_categorical('use_pipeline_size_constraint', [False]):
        pipeline_size_limit = trial.suggest_loguniform('pipeline_size_constraint', 2000, 350000000)


    # how many cvs should be used
    cv = 1
    number_of_cvs = 1
    hold_out_fraction = None
    if trial.suggest_categorical('use_hold_out', [True]):
        hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0.33, 0.33)
    else:
        cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
        number_of_cvs = 1
        if trial.suggest_categorical('use_multiple_cvs', [True, False]):
            number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)

    sample_fraction = 1.0
    if trial.suggest_categorical('use_sampling', [True]):
        sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)

    dataset_id = None
    if sample_data:
        dataset_id = trial.suggest_categorical('dataset_id', my_openml_datasets)

    return search_time, evaluation_time, memory_limit, privacy_limit, training_time_limit, inference_time_limit, pipeline_size_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, dataset_id

def generate_parameters_minimal_sample_eval_time(trial, total_search_time_minutes, my_openml_datasets, sample_data=True):
    # which constraints to use
    search_time = trial.suggest_int('global_search_time_constraint', 10, max(10, total_search_time_minutes), log=False) #* 60

    # how much time for each evaluation
    evaluation_time = int(0.1 * search_time)
    if trial.suggest_categorical('use_evaluation_time_constraint', [False, True]):
        evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)

    # how much memory is allowed
    memory_limit = 10
    if trial.suggest_categorical('use_search_memory_constraint', [False]):
        memory_limit = trial.suggest_loguniform('global_memory_constraint', 0.00000000000001, 10)

    # how much privacy is required
    privacy_limit = None
    if trial.suggest_categorical('use_privacy_constraint', [False]):
        privacy_limit = trial.suggest_loguniform('privacy_constraint', 0.0001, 10)

    training_time_limit = search_time
    if trial.suggest_categorical('use_training_time_constraint', [False]):
        training_time_limit = trial.suggest_loguniform('training_time_constraint', 0.005, search_time)

    inference_time_limit = 60
    if trial.suggest_categorical('use_inference_time_constraint', [False]):
        inference_time_limit = trial.suggest_loguniform('inference_time_constraint', 0.0004, 60)

    pipeline_size_limit = 350000000
    if trial.suggest_categorical('use_pipeline_size_constraint', [False]):
        pipeline_size_limit = trial.suggest_loguniform('pipeline_size_constraint', 2000, 350000000)


    # how many cvs should be used
    cv = 1
    number_of_cvs = 1
    hold_out_fraction = None
    if trial.suggest_categorical('use_hold_out', [True]):
        hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0.33, 0.33)
    else:
        cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
        number_of_cvs = 1
        if trial.suggest_categorical('use_multiple_cvs', [True, False]):
            number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)

    sample_fraction = 1.0
    if trial.suggest_categorical('use_sampling', [True]):
        sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)

    dataset_id = None
    if sample_data:
        dataset_id = trial.suggest_categorical('dataset_id', my_openml_datasets)

    return search_time, evaluation_time, memory_limit, privacy_limit, training_time_limit, inference_time_limit, pipeline_size_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, dataset_id


def generate_parameters_minimal_sample_val_fraction(trial, total_search_time_minutes, my_openml_datasets, sample_data=True):
    # which constraints to use
    search_time = trial.suggest_int('global_search_time_constraint', 10, max(10, total_search_time_minutes), log=False) #* 60

    # how much time for each evaluation
    evaluation_time = int(0.1 * search_time)
    if trial.suggest_categorical('use_evaluation_time_constraint', [False]):
        evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)

    # how much memory is allowed
    memory_limit = 10
    if trial.suggest_categorical('use_search_memory_constraint', [False]):
        memory_limit = trial.suggest_loguniform('global_memory_constraint', 0.00000000000001, 10)

    # how much privacy is required
    privacy_limit = None
    if trial.suggest_categorical('use_privacy_constraint', [False]):
        privacy_limit = trial.suggest_loguniform('privacy_constraint', 0.0001, 10)

    training_time_limit = search_time
    if trial.suggest_categorical('use_training_time_constraint', [False]):
        training_time_limit = trial.suggest_loguniform('training_time_constraint', 0.005, search_time)

    inference_time_limit = 60
    if trial.suggest_categorical('use_inference_time_constraint', [False]):
        inference_time_limit = trial.suggest_loguniform('inference_time_constraint', 0.0004, 60)

    pipeline_size_limit = 350000000
    if trial.suggest_categorical('use_pipeline_size_constraint', [False]):
        pipeline_size_limit = trial.suggest_loguniform('pipeline_size_constraint', 2000, 350000000)


    # how many cvs should be used
    cv = 1
    number_of_cvs = 1
    hold_out_fraction = None
    if trial.suggest_categorical('use_hold_out', [True]):
        hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0.0, 1.0)
    else:
        cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
        number_of_cvs = 1
        if trial.suggest_categorical('use_multiple_cvs', [True, False]):
            number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)

    sample_fraction = 1.0
    if trial.suggest_categorical('use_sampling', [True]):
        sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)

    dataset_id = None
    if sample_data:
        dataset_id = trial.suggest_categorical('dataset_id', my_openml_datasets)

    return search_time, evaluation_time, memory_limit, privacy_limit, training_time_limit, inference_time_limit, pipeline_size_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, dataset_id


def generate_parameters_minimal_sample_cv(trial, total_search_time_minutes, my_openml_datasets, sample_data=True):
    # which constraints to use
    search_time = trial.suggest_int('global_search_time_constraint', 10, max(10, total_search_time_minutes), log=False) #* 60

    # how much time for each evaluation
    evaluation_time = int(0.1 * search_time)
    if trial.suggest_categorical('use_evaluation_time_constraint', [False]):
        evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)

    # how much memory is allowed
    memory_limit = 10
    if trial.suggest_categorical('use_search_memory_constraint', [False]):
        memory_limit = trial.suggest_loguniform('global_memory_constraint', 0.00000000000001, 10)

    # how much privacy is required
    privacy_limit = None
    if trial.suggest_categorical('use_privacy_constraint', [False]):
        privacy_limit = trial.suggest_loguniform('privacy_constraint', 0.0001, 10)

    training_time_limit = search_time
    if trial.suggest_categorical('use_training_time_constraint', [False]):
        training_time_limit = trial.suggest_loguniform('training_time_constraint', 0.005, search_time)

    inference_time_limit = 60
    if trial.suggest_categorical('use_inference_time_constraint', [False]):
        inference_time_limit = trial.suggest_loguniform('inference_time_constraint', 0.0004, 60)

    pipeline_size_limit = 350000000
    if trial.suggest_categorical('use_pipeline_size_constraint', [False]):
        pipeline_size_limit = trial.suggest_loguniform('pipeline_size_constraint', 2000, 350000000)


    # how many cvs should be used
    cv = 1
    number_of_cvs = 1
    hold_out_fraction = None
    if trial.suggest_categorical('use_hold_out', [True, False]):
        hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0.33, 0.33)
    else:
        cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
        number_of_cvs = 1
        if trial.suggest_categorical('use_multiple_cvs', [True, False]):
            number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)

    sample_fraction = 1.0
    if trial.suggest_categorical('use_sampling', [True]):
        sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)

    dataset_id = None
    if sample_data:
        dataset_id = trial.suggest_categorical('dataset_id', my_openml_datasets)

    return search_time, evaluation_time, memory_limit, privacy_limit, training_time_limit, inference_time_limit, pipeline_size_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, dataset_id





def optimize_accuracy_under_constraints(trial, metafeature_values_hold, search_time, model,
                                        memory_limit=10,
                                        privacy_limit=None,
                                        evaluation_time=None,
                                        hold_out_fraction=None):
    try:
        gen = SpaceGenerator()
        space = gen.generate_params()
        space.sample_parameters(trial)

        trial.set_user_attr('space', copy.deepcopy(space))

        if type(evaluation_time) == type(None):
            evaluation_time = search_time
            if trial.suggest_categorical('use_evaluation_time_constraint', [True, False]):
                evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)
        else:
            trial.set_user_attr('evaluation_time', evaluation_time)

        # how many cvs should be used
        cv = 1
        number_of_cvs = 1
        if type(hold_out_fraction) == type(None):
            hold_out_fraction = None
            if trial.suggest_categorical('use_hold_out', [True, False]):
                hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0, 1)
            else:
                cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
                number_of_cvs = 1
                if trial.suggest_categorical('use_multiple_cvs', [True, False]):
                    number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)
        else:
            trial.set_user_attr('hold_out_fraction', hold_out_fraction)


        sample_fraction = 1.0
        #if trial.suggest_categorical('use_sampling', [True, False]):
        #    sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)



        my_list_constraints_values = [search_time,
                                      evaluation_time,
                                      memory_limit,
                                      cv,
                                      number_of_cvs,
                                      ifNull(privacy_limit, constant_value=1000),
                                      ifNull(hold_out_fraction),
                                      sample_fraction]

        features = space2features(space, my_list_constraints_values, metafeature_values_hold)
        feature_names, _ = get_feature_names()
        features = FeatureTransformations().fit(features).transform(features, feature_names=feature_names)
        trial.set_user_attr('features', features)

        return predict_range(model, features)
    except Exception as e:
        print(str(e) + 'except dataset _ accuracy: ' + '\n\n')
        return 0.0

def utils_run_AutoML(trial, X_train=None, X_test=None, y_train=None, y_test=None, categorical_indicator=None, my_scorer=None,
               search_time=None,
               memory_limit=None,
               privacy_limit=None,
               training_time_limit=None,
               inference_time_limit=None,
               pipeline_size_limit=None,
               space=None
               ):
    if type(None) == type(space):
        space = trial.user_attrs['space']

    print(trial.params)

    if 'evaluation_time' in trial.user_attrs:
        evaluation_time = trial.user_attrs['evaluation_time']
    else:
        evaluation_time = search_time
        if 'global_evaluation_time_constraint' in trial.params:
            evaluation_time = trial.params['global_evaluation_time_constraint']

    cv = 1
    number_of_cvs = 1
    if 'hold_out_fraction' in trial.user_attrs:
        hold_out_fraction = trial.user_attrs['hold_out_fraction']
    else:
        hold_out_fraction = None
        if 'global_cv' in trial.params:
            cv = trial.params['global_cv']
            if 'global_number_cv' in trial.params:
                number_of_cvs = trial.params['global_number_cv']
        if 'hold_out_fraction' in trial.params:
            hold_out_fraction = trial.params['hold_out_fraction']

    sample_fraction = 1.0
    if 'sample_fraction' in trial.params:
        sample_fraction = trial.params['sample_fraction']

    search = MyAutoML(cv=cv,
                      number_of_cvs=number_of_cvs,
                      n_jobs=1,
                      evaluation_budget=evaluation_time,
                      time_search_budget=search_time,
                      space=space,
                      main_memory_budget_gb=memory_limit,
                      differential_privacy_epsilon=privacy_limit,
                      hold_out_fraction=hold_out_fraction,
                      sample_fraction=sample_fraction,
                      training_time_limit=training_time_limit,
                      inference_time_limit=inference_time_limit,
                      pipeline_size_limit=pipeline_size_limit
                      )
    search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=my_scorer)

    best_pipeline = search.get_best_pipeline()

    test_score = 0.0
    if type(best_pipeline) != type(None):
        test_score = my_scorer(search.get_best_pipeline(), X_test, y_test)


    return test_score, search

def show_progress(search, X_test, y_test, scorer):
    times = []
    validation_scores = []
    best_scores = []
    current_best = 0
    real_scores = []
    current_real = 0.0

    for t in search.study.trials:
        try:
            current_time = t.user_attrs['time_since_start']
            current_val = t.value
            if current_val < 0:
                current_val = 0
            times.append(current_time)
            validation_scores.append(current_val)

            current_pipeline = None
            try:
                current_pipeline = t.user_attrs['pipeline']
            except:
                pass

            if type(current_pipeline) != type(None):
                current_real = scorer(current_pipeline, X_test, y_test)

            real_scores.append(copy.deepcopy(current_real))

            if current_val > current_best:
                current_best = current_val
            best_scores.append(copy.deepcopy(current_best))

        except:
            pass

    import matplotlib.pyplot as plt
    plt.plot(times, best_scores, color='red')
    plt.plot(times, real_scores, color='blue')
    plt.show()
    print(times)
    print(best_scores)
    print(real_scores)