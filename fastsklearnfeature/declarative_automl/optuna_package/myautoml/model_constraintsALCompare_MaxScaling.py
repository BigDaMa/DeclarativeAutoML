from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.MyAutoMLProcessClassBalance import MyAutoML
import optuna
import time
from sklearn.metrics import make_scorer
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
import copy
from optuna.trial import FrozenTrial
from anytree import RenderTree
from sklearn.ensemble import RandomForestRegressor
from optuna.samplers import RandomSampler
import pickle
import heapq
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.mp_global_vars as mp_glob
from sklearn.metrics import f1_score
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.feature_transformation.FeatureTransformations import FeatureTransformations
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import space2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import MyPool
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import ifNull
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import generate_parameters
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV

def predict_range(model, X):
    y_pred = model.predict(X)
    return y_pred

#test data

test_holdout_dataset_id = [1134, 1495, 41147, 316, 1085, 1046, 1111, 55, 1116, 448, 1458, 162, 1101, 1561, 1061, 1506, 1235, 4135, 151, 51, 41138, 40645, 1510, 1158, 312, 38, 52, 1216, 41007, 1130]

my_scorer = make_scorer(f1_score)


total_search_time = 5*60#60*60#60*60#10 * 60

my_openml_datasets = [3, 4, 13, 15, 24, 25, 29, 31, 37, 38, 40, 43, 44, 49, 50, 51, 52, 53, 55, 56, 59, 151, 152, 153, 161, 162, 164, 172, 179, 310, 311, 312, 316, 333, 334, 335, 336, 337, 346, 444, 446, 448, 450, 451, 459, 461, 463, 464, 465, 466, 467, 470, 472, 476, 479, 481, 682, 683, 747, 803, 981, 993, 1037, 1038, 1039, 1040, 1042, 1045, 1046, 1048, 1049, 1050, 1053, 1054, 1055, 1056, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1071, 1073, 1075, 1085, 1101, 1104, 1107, 1111, 1112, 1114, 1116, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1169, 1216, 1235, 1236, 1237, 1238, 1240, 1412, 1441, 1442, 1443, 1444, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1455, 1458, 1460, 1461, 1462, 1463, 1464, 1467, 1471, 1473, 1479, 1480, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1494, 1495, 1496, 1498, 1502, 1504, 1506, 1507, 1510, 1511, 1547, 1561, 1562, 1563, 1564, 1597, 4134, 4135, 4154, 4329, 4534, 23499, 40536, 40645, 40646, 40647, 40648, 40649, 40650, 40660, 40665, 40666, 40669, 40680, 40681, 40690, 40693, 40701, 40705, 40706, 40710, 40713, 40714, 40900, 40910, 40922, 40999, 41005, 41007, 41138, 41142, 41144, 41145, 41146, 41147, 41150, 41156, 41158, 41159, 41160, 41161, 41162, 41228, 41430, 41521, 41538, 41976, 42172, 42477]
for t_v in test_holdout_dataset_id:
    my_openml_datasets.remove(t_v)

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

feature_names, feature_names_new = get_feature_names(my_list_constraints)


def run_AutoML(trial, X_train=None, X_test=None, y_train=None, y_test=None, categorical_indicator=None):
    space = None
    search_time = None
    if not 'space' in trial.user_attrs:
        # which hyperparameters to use
        gen = SpaceGenerator()
        space = gen.generate_params()
        space.sample_parameters(trial)

        trial.set_user_attr('space', copy.deepcopy(space))

        search_time, evaluation_time, memory_limit, privacy_limit, training_time_limit, inference_time_limit, pipeline_size_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, dataset_id = generate_parameters(trial, total_search_time, my_openml_datasets)

    else:
        space = trial.user_attrs['space']

        print(trial.params)

        #make this a hyperparameter
        search_time = trial.params['global_search_time_constraint']

        evaluation_time = search_time
        if 'global_evaluation_time_constraint' in trial.params:
            evaluation_time = trial.params['global_evaluation_time_constraint']

        memory_limit = 10
        if 'global_memory_constraint' in trial.params:
            memory_limit = trial.params['global_memory_constraint']

        privacy_limit = None
        if 'privacy_constraint' in trial.params:
            privacy_limit = trial.params['privacy_constraint']

        training_time_limit = search_time
        if 'training_time_constraint' in trial.params:
            training_time_limit = trial.params['training_time_constraint']

        inference_time_limit = 60
        if 'inference_time_constraint' in trial.params:
            inference_time_limit = trial.params['inference_time_constraint']

        pipeline_size_limit = 350000000
        if 'pipeline_size_constraint' in trial.params:
            pipeline_size_limit = trial.params['pipeline_size_constraint']

        cv = 1
        number_of_cvs = 1
        hold_out_fraction = None
        if 'global_cv' in trial.params:
            cv = trial.params['global_cv']
            if 'global_number_cv' in trial.params:
                number_of_cvs = trial.params['global_number_cv']
        else:
            hold_out_fraction = trial.params['hold_out_fraction']

        sample_fraction = 1.0
        if 'sample_fraction' in trial.params:
            sample_fraction = trial.params['sample_fraction']

        if 'dataset_id' in trial.params:
            dataset_id = trial.params['dataset_id'] #get same random seed
        else:
            dataset_id = 31

    for pre, _, node in RenderTree(space.parameter_tree):
        if node.status == True:
            print("%s%s" % (pre, node.name))

    if type(X_train) == type(None):

        my_random_seed = int(time.time())
        if 'data_random_seed' in trial.user_attrs:
            my_random_seed = trial.user_attrs['data_random_seed']

        X_train, X_test, y_train, y_test, categorical_indicator, attribute_names = get_data(dataset_id, randomstate=my_random_seed)

        if not isinstance(trial, FrozenTrial):
            my_list_constraints_values = [search_time,
                                          evaluation_time,
                                          memory_limit, cv,
                                          number_of_cvs,
                                          ifNull(privacy_limit, constant_value=1000),
                                          ifNull(hold_out_fraction),
                                          sample_fraction,
                                          training_time_limit,
                                          inference_time_limit,
                                          pipeline_size_limit]

            metafeature_values = data2features(X_train, y_train, categorical_indicator)
            features = space2features(space, my_list_constraints_values, metafeature_values)
            features = FeatureTransformations().fit(features).transform(features, feature_names=feature_names)
            trial.set_user_attr('features', features)


    dynamic_params = []
    static_params = []
    for random_i in range(5):
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
                          pipeline_size_limit=pipeline_size_limit)

        test_score = 0.0
        try:
            search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=my_scorer)

            best_pipeline = search.get_best_pipeline()
            if type(best_pipeline) != type(None):
                test_score = my_scorer(search.get_best_pipeline(), X_test, y_test)
        except:
            pass
        dynamic_params.append(test_score)

        # default params
        gen_new = SpaceGenerator()
        space_new = gen_new.generate_params()
        for pre, _, node in RenderTree(space_new.parameter_tree):
            if node.status == True:
                print("%s%s" % (pre, node.name))

        search_static = MyAutoML(n_jobs=1,
                          time_search_budget=search_time,
                          space=space_new,
                          evaluation_budget=int(0.1 * search_time),
                          main_memory_budget_gb=memory_limit,
                          differential_privacy_epsilon=privacy_limit,
                          hold_out_fraction=0.33,
                          training_time_limit=training_time_limit,
                          inference_time_limit=inference_time_limit,
                          pipeline_size_limit=pipeline_size_limit
                          )

        try:
            best_result = search_static.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=my_scorer)
            test_score_default = my_scorer(search_static.get_best_pipeline(), X_test, y_test)
        except:
            test_score_default = 0.0
        static_params.append(test_score_default)

    #this was a bad idea because it is not comparable across scenarios
    #comparison = np.mean(dynamic_params) - np.mean(static_params)

    #conservativ cost function
    all_results = copy.deepcopy(dynamic_params)
    all_results.extend(static_params)

    if np.max(all_results) <= 0.0:
        comparison = 0.0
    else:
        comparison = float(np.quantile(dynamic_params, 0.1)) / float(np.max(all_results))

    #less conservativ
    #but this has too little signal
    '''
    if np.max([np.mean(static_params), np.mean(dynamic_params)]) <= 0.0:
        comparison = 1.0
    else:
        comparison = np.mean(dynamic_params) / np.max([np.mean(static_params), np.mean(dynamic_params)])
    '''

    return comparison, search


def run_AutoML_global(trial_id):
    comparison, current_search = run_AutoML(mp_glob.my_trials[trial_id])

    # get all best validation scores and evaluate them
    feature_list = []
    target_list = []

    feature_list.append(copy.deepcopy(mp_glob.my_trials[trial_id].user_attrs['features']))
    target_list.append(comparison)

    return {'feature_l': feature_list,
            'target_l': target_list,
            'loss': np.square(target_list[-1] - mp_glob.my_trials[trial_id].user_attrs['predicted_target']),
            'group_l': copy.deepcopy(mp_glob.my_trials[trial_id].params['dataset_id'])}


def run_AutoML_score_only(trial, X_train=None, X_test=None, y_train=None, y_test=None, categorical_indicator=None):
    test_score, _ = run_AutoML(trial, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, categorical_indicator=categorical_indicator)
    return test_score


def optimize_uncertainty(trial):
    try:
        gen = SpaceGenerator()
        space = gen.generate_params()
        space.sample_parameters(trial)

        trial.set_user_attr('space', copy.deepcopy(space))

        search_time, evaluation_time, memory_limit, privacy_limit, training_time_limit, inference_time_limit, pipeline_size_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, dataset_id = generate_parameters(trial, total_search_time, my_openml_datasets)

        my_random_seed = int(time.time())

        X_train, X_test, y_train, y_test, categorical_indicator, attribute_names = get_data(dataset_id,
                                                                                            randomstate=my_random_seed)

        trial.set_user_attr('data_random_seed', my_random_seed)

        #add metafeatures of data
        my_list_constraints_values = [search_time,
                                      evaluation_time,
                                      memory_limit,
                                      cv,
                                      number_of_cvs,
                                      ifNull(privacy_limit, constant_value=1000),
                                      ifNull(hold_out_fraction),
                                      sample_fraction,
                                      training_time_limit,
                                      inference_time_limit,
                                      pipeline_size_limit]

        metafeature_values = data2features(X_train, y_train, categorical_indicator)
        features = space2features(space, my_list_constraints_values, metafeature_values)
        features = FeatureTransformations().fit(features).transform(features, feature_names=feature_names)

        trial.set_user_attr('features', features)

        trial.set_user_attr('predicted_target', model.predict(features))

        predictions = []
        for tree in range(model.n_estimators):
            predictions.append(predict_range(model.estimators_[tree], features))

        stddev_pred = np.std(np.matrix(predictions).transpose(), axis=1)

        return stddev_pred[0]
    except Exception as e:
        print(str(e) + 'except dataset _ uncertainty: ' + str(dataset_id) + '\n\n')
        return -1 * np.inf



#cold start - random sampling
study = optuna.create_study(direction='maximize', sampler=RandomSampler(seed=42))
study.optimize(run_AutoML_score_only, n_trials=4, n_jobs=1)

print('done')

#generate training data
all_trials = study.get_trials()
X_meta = []
y_meta = []
group_meta = []

for t in range(len(study.get_trials())):
    current_trial = all_trials[t]
    if current_trial.value >= 0 and current_trial.value <= 1.0:
        y_meta.append(current_trial.value)
    else:
        y_meta.append(0.0)
    X_meta.append(current_trial.user_attrs['features'])
    group_meta.append(current_trial.params['dataset_id'])

X_meta = np.vstack(X_meta)
print(X_meta.shape)


pruned_accuray_results = []

verbose = False

cv_over_time = []

topk = 20
while True:

    assert X_meta.shape[1] == len(feature_names_new), 'error'

    model = None
    if len(np.unique(group_meta)) > topk:
        gkf = GroupKFold(n_splits=topk)
        cross_val = GridSearchCV(RandomForestRegressor(), param_grid={'n_estimators': [1000]}, cv=gkf, refit=True,
                                 scoring='r2', n_jobs=topk)
        cross_val.fit(X_meta, y_meta, groups=group_meta)
        model = cross_val.best_estimator_
        cv_over_time.append(cross_val.best_score_)

        with open('/tmp/my_great_model_compare_scaled.p', "wb") as pickle_model_file:
            pickle.dump(model, pickle_model_file)

        with open('/tmp/felix_X_compare_scaled.p', "wb") as pickle_model_file:
            pickle.dump(X_meta, pickle_model_file)

        with open('/tmp/felix_y_compare_scaled.p', "wb") as pickle_model_file:
            pickle.dump(y_meta, pickle_model_file)

        with open('/tmp/felix_group_compare_scaled.p', "wb") as pickle_model_file:
            pickle.dump(group_meta, pickle_model_file)

        with open('/tmp/felix_cv_compare_scaled.p', "wb") as pickle_model_file:
            pickle.dump(cv_over_time, pickle_model_file)
    else:
        model = RandomForestRegressor(n_estimators=1000)
        model.fit(X_meta, y_meta)
    mp_glob.ml_model = model

    #random sampling 10 iterations
    study_uncertainty = optuna.create_study(direction='maximize')
    study_uncertainty.optimize(optimize_uncertainty, n_trials=100, n_jobs=1)

    #get most uncertain for k datasets and run k runs in parallel
    data2most_uncertain = {}
    for u_trial in study_uncertainty.trials:
        u_dataset = u_trial.params['dataset_id']
        u_value = u_trial.value
        if u_value >= 0:
            data2most_uncertain[u_dataset] = (u_trial, u_value)

    k_keys_sorted_by_values = heapq.nlargest(topk, data2most_uncertain, key=lambda s: data2most_uncertain[s][1])

    mp_glob.my_trials = []
    for keyy in k_keys_sorted_by_values:
        mp_glob.my_trials.append(data2most_uncertain[keyy][0])

    with MyPool(processes=topk) as pool:
        results = pool.map(run_AutoML_global, range(topk))

    for result_p in results:
        for f_progress in range(len(result_p['feature_l'])):
            X_meta = np.vstack((X_meta, result_p['feature_l'][f_progress]))
            y_meta.append(result_p['target_l'][f_progress])
            group_meta.append(result_p['group_l'])


