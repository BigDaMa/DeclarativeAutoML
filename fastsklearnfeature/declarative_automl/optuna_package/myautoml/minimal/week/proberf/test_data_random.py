import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.feature_transformation.FeatureTransformations import FeatureTransformations
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import space2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import ifNull
from anytree import Node
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import optimize_accuracy_under_minimal_sample_ensemble
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import generate_features_minimum_sample_ensemble
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import utils_run_AutoML_ensemble
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.util_classes_new import ConstraintEvaluation, ConstraintRun, space2str
from anytree import RenderTree
import argparse
import openml
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.my_global_vars as mp_global
import optuna
import time
import copy
import getpass

test_holdout_dataset_id = 75097
X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data('data', randomstate=42, task_id=test_holdout_dataset_id)
metafeature_values_hold = data2features(X_train_hold, y_train_hold, categorical_indicator_hold)

gen = SpaceGenerator()
space = gen.generate_params()


my_list = list(space.name2node.keys())
my_list.sort()
print('len list:' + str(len(my_list)))

def generate_constraints(node):
    all_constraints = ''
    for child in node.children:
        if node.name in my_list:
            all_constraints += "{'type': 'ineq', 'fun': lambda x:  x[" + str(my_list.index(node.name)) + "]-x[" + str(my_list.index(child.name)) + "]},\n"
        all_constraints += generate_constraints(child)
    return all_constraints

all_constraints = ''
print(generate_constraints(space.parameter_tree))

X = pickle.load(open('/home/felix/phd2/dec_automl/oct16_1day_al/felix_X_compare_scaled.p', "rb"))
y = pickle.load(open('/home/felix/phd2/dec_automl/oct16_1day_al/felix_y_compare_scaled.p', "rb"))
groups = pickle.load(open('/home/felix/phd2/dec_automl/oct16_1day_al/felix_group_compare_scaled.p', "rb"))

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


static_names = ['global_search_time_constraint',
                           'global_evaluation_time_constraint',
                           'global_memory_constraint',
                           'global_cv',
                           'global_number_cv',
                           'privacy',
                           'sample_fraction',
                           'training_time_constraint',
                           'inference_time_constraint',
                           'pipeline_size_constraint',
                           'fairness_constraint',
                           'product_cvs',
                           'sampled_instances',
                           'number_of_evaluations',
                           'log_search_time_constraint',
                           'log_evaluation_time_constraint',
                           'log_search_memory_constraint',
                           'log_privacy_constraint',
                           'log_sampled_instances',
                           'has_privacy_constraint',
                           'has_evaluation_time_constraint']



_, feature_names = get_feature_names(my_list_constraints)

search_time = 60 * 5
evaluation_time = 0.1 * search_time
memory_limit = 10
cv = 1
number_of_cvs = 1
privacy_limit = None
sample_fraction = 1.0


training_time_limit = None
inference_time_limit = None
pipeline_size_limit = None
fairness_limit = None


hold_out_fraction = -1
use_ensemble = -1
use_incremental_data = -1
shuffle_validation = -1


my_list_constraints_values = [search_time,
                                      evaluation_time,
                                      memory_limit,
                                      cv,
                                      number_of_cvs,
                                      ifNull(privacy_limit, constant_value=1000),
                                      hold_out_fraction,
                                      sample_fraction,
                                      ifNull(training_time_limit, constant_value=search_time),
                                      ifNull(inference_time_limit, constant_value=60),
                                      ifNull(pipeline_size_limit, constant_value=350000000),
                                      ifNull(fairness_limit, constant_value=0.0),
                                      int(use_ensemble),
                                      int(use_incremental_data),
                                      int(shuffle_validation)
                                      ]

features = space2features(space, my_list_constraints_values, metafeature_values_hold)
_, feature_names = get_feature_names(my_list_constraints)
features = FeatureTransformations().fit(features).transform(features, feature_names=feature_names)

hold_out_test_instances = 1.0 * features[0, feature_names.index('NumberOfInstances')] * features[0, feature_names.index('sample_fraction')]
print(hold_out_test_instances)


assert len(feature_names) == X.shape[1]


print(X.shape)

#model_success = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=1, max_depth=None)

#model_success = pickle.load(open('/home/felix/phd2/dec_automl/okt25_al_opt_models/my_great_model_compare_scaled7.p', "rb"))
model_success = pickle.load(open('/home/' + getpass.getuser() + '/data/my_temp/my_great_model_compare_scaled.p', "rb"))
import xgboost as xgb
#model_success = xgb.XGBRegressor(tree_method="hist", n_estimators=100)
#model_success.fit(X, y)


new_array = [X[0,:]]
for i in range(30000):
    new_array.append(X[0,:])
start_time = time.time()
print(len(model_success.predict(new_array)))
print('period: ' + str(time.time() - start_time))


from optuna.samplers import RandomSampler

best_random_val = 0.0
best_opt = 0.0

def optimize_accuracy_under_minimal_sample_ensemble_features(trial, metafeature_values_hold, search_time, model_success,
                                        memory_limit=10,
                                        privacy_limit=None,
                                        evaluation_time=None,
                                        hold_out_fraction=None,
                                        training_time_limit=None,
                                        inference_time_limit=None,
                                        pipeline_size_limit=None,
                                        fairness_limit=None,
                                        tune_space=False,
                                        tune_eval_time=False,
                                        tune_val_fraction=False,
                                        tune_cv=False
                                        ):
    features, space = generate_features_minimum_sample_ensemble(trial, metafeature_values_hold, search_time,
                      memory_limit=memory_limit,
                      privacy_limit=privacy_limit,
                      evaluation_time=evaluation_time,
                      hold_out_fraction=hold_out_fraction,
                      training_time_limit=training_time_limit,
                      inference_time_limit=inference_time_limit,
                      pipeline_size_limit=pipeline_size_limit,
                      fairness_limit=fairness_limit,
                      tune_space=tune_space,
                      save_data=False,
                      tune_eval_time=tune_eval_time,
                      tune_val_fraction=tune_val_fraction,
                      tune_cv=tune_cv
                      )

    trial.set_user_attr('features', copy.deepcopy(features))
    #trial.set_user_attr('space', copy.deepcopy(space))

    return 1.0


'''
startts = time.time()
mp_global.study_prune = optuna.create_study(direction='maximize', sampler=RandomSampler()) #TODO
mp_global.study_prune.optimize(lambda trial: optimize_accuracy_under_minimal_sample_ensemble_features(trial=trial,
                                                                       metafeature_values_hold=metafeature_values_hold,
                                                                       search_time=search_time,
                                                                       model_success=model_success,
                                                                       memory_limit=memory_limit,
                                                                       privacy_limit=privacy_limit,
                                                                       #evaluation_time=int(0.1*search_time_frozen),
                                                                       #hold_out_fraction=0.33,
                                                                       tune_space=True,
                                                                       tune_val_fraction=True
                                                                       ), n_trials=100000, n_jobs=1)
print('sampling_time: ' + str(time.time() - startts))
'''


random_configs = pickle.load(open('/home/' + getpass.getuser() + '/data/my_temp/randomconfigs.p', "rb"))['random_configs']

'''
random_configs = []
spaces = []
for trial in mp_global.study_prune.trials:
    random_configs.append(trial.user_attrs['features'][0,:])
    #spaces.append(trial.user_attrs['space'])
'''


start_time = time.time()
print(len(model_success.predict(random_configs)))
print(np.max(model_success.predict(random_configs)))
print('period: ' + str(time.time() - start_time))


'''
new_dict = {}
new_dict['random_configs'] = random_configs
#new_dict['spaces'] = spaces
with open('/tmp/randomconfigs.p', "wb+") as pickle_model_file:
    pickle.dump(new_dict, pickle_model_file)
'''


