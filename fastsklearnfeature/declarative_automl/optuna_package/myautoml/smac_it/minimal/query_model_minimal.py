import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.smac_it.CustomRandomForest import CustomRandomForest
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
import optuna
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import data2features
import copy
from anytree import RenderTree
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import utils_run_AutoML
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.MyAutoMLProcessClassBalance import MyAutoML
import openml


openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'

model = pickle.load(open('/tmp/smac_model.p', "rb"))
conf = pickle.load(open('/tmp/smac_conf.p', "rb"))

new_model = CustomRandomForest(**conf)
new_model.rf = model

cs = conf['configspace']

print(list(cs._hyperparameters.keys()))

first = cs._hyperparameters[cs._idx_to_hyperparameter[1]]
print(first)

val = first._sample(rs=np.random.RandomState(seed=42))

print(val)
print(first._transform(val)) # vector => real value


new_val = False
print(first._inverse_transform(new_val)) # real value => vector


dataset_id = 448#55
X_train, X_test, y_train, y_test, categorical_indicator, attribute_names = get_data(dataset_id, randomstate=42)
metafeature_values = data2features(X_train, y_train, categorical_indicator)


def query_model(trial, search_time=60, global_memory_constraint=None, privacy_constraint=None, training_time_constraint=None, inference_time_constraint=None, pipeline_size_constraint=None):
    gen = SpaceGenerator()
    space = gen.generate_params()
    #space.sample_parameters(trial)

    scaled_vector = np.zeros((1, len(cs._hyperparameters)))
    scaled_vector[:] = np.nan

    for param_str, real_val in trial.params.items():
        scaled_val = cs._hyperparameters[param_str]._inverse_transform(real_val)
        vector_id = cs._hyperparameter_idx[param_str]

        scaled_vector[0, vector_id] = scaled_val

    param_dict = {}
    param_dict['global_search_time_constraint'] = search_time
    param_dict['use_evaluation_time_constraint'] = trial.suggest_categorical('use_evaluation_time_constraint', [True, False])

    evaluation_time = search_time
    if param_dict['use_evaluation_time_constraint']:
        evaluation_time = trial.suggest_int('global_evaluation_time_constraint', 10, search_time, log=False)
    param_dict['global_evaluation_time_constraint'] = evaluation_time

    # how many cvs should be used
    cv = 1
    number_of_cvs = 1
    hold_out_fraction = None

    param_dict['use_hold_out'] = trial.suggest_categorical('use_hold_out', [True])
    if param_dict['use_hold_out']:
        hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0, 1)
        param_dict['hold_out_fraction'] = hold_out_fraction
    else:
        cv = trial.suggest_int('global_cv', 2, 20, log=False)  # todo: calculate minimum number of splits based on y
        param_dict['global_cv'] = cv

        param_dict['use_multiple_cvs'] = trial.suggest_categorical('use_multiple_cvs', [True, False])

        number_of_cvs = 1
        if param_dict['use_multiple_cvs']:
            number_of_cvs = trial.suggest_int('global_number_cv', 2, 10, log=False)
        param_dict['global_number_cv'] = number_of_cvs

    sample_fraction = 1.0
    param_dict['use_sampling'] = trial.suggest_categorical('use_sampling', [True, False])
    if param_dict['use_sampling']:
        sample_fraction = trial.suggest_uniform('sample_fraction', 0, 1)
    param_dict['sample_fraction'] = sample_fraction

    #now the constraints
    if type(None) == type(global_memory_constraint):
        param_dict['use_search_memory_constraint'] = False
        param_dict['global_memory_constraint'] = 10
    else:
        param_dict['use_search_memory_constraint'] = True
        param_dict['global_memory_constraint'] = global_memory_constraint

    if type(None) == type(privacy_constraint):
        param_dict['use_privacy_constraint'] = False
        param_dict['privacy_constraint'] = 10
    else:
        param_dict['use_privacy_constraint'] = True
        param_dict['privacy_constraint'] = privacy_constraint

    if type(None) == type(training_time_constraint):
        param_dict['use_training_time_constraint'] = False
        param_dict['training_time_constraint'] = search_time
    else:
        param_dict['use_training_time_constraint'] = True
        param_dict['training_time_constraint'] = training_time_constraint

    if type(None) == type(inference_time_constraint):
        param_dict['use_inference_time_constraint'] = False
        param_dict['inference_time_constraint'] = 60
    else:
        param_dict['use_inference_time_constraint'] = True
        param_dict['inference_time_constraint'] = inference_time_constraint

    if type(None) == type(pipeline_size_constraint):
        param_dict['use_pipeline_size_constraint'] = False
        param_dict['pipeline_size_constraint'] = 350000000
    else:
        param_dict['use_pipeline_size_constraint'] = True
        param_dict['pipeline_size_constraint'] = pipeline_size_constraint

    for param_str, real_val in param_dict.items():
        if param_str in cs._hyperparameter_idx:
            print(param_str)
            scaled_val = cs._hyperparameters[param_str]._inverse_transform(real_val)
            vector_id = cs._hyperparameter_idx[param_str]

            scaled_vector[0, vector_id] = scaled_val

    success = new_model._predict(X=scaled_vector, metafeatures=metafeature_values)[0][0][0]


    try:
        print(study_prune.best_trial)
        if success < study_prune.best_trial.value:
            trial.set_user_attr('space', copy.deepcopy(space))
    except:
        pass

    return success




study_prune = optuna.create_study(direction='minimize')
study_prune.optimize(lambda trial: query_model(trial=trial, search_time=60), n_trials=1000, n_jobs=1)

space = study_prune.best_trial.user_attrs['space']

for pre, _, node in RenderTree(space.parameter_tree):
    if node.status == True:
        print("%s%s" % (pre, node.name))

my_scorer=make_scorer(f1_score)

search_dynamic = None
result = 0
try:
    result, search_dynamic = utils_run_AutoML(study_prune.best_trial,
                                                 X_train=X_train,
                                                 X_test=X_test,
                                                 y_train=y_train,
                                                 y_test=y_test,
                                                 categorical_indicator=categorical_indicator,
                                                 my_scorer=my_scorer,
                                                 search_time=60,
                                                 memory_limit=10
                                 )
except:
    result = 0

print(result)

gen_new = SpaceGenerator()
space_new = gen_new.generate_params()
for pre, _, node in RenderTree(space_new.parameter_tree):
    if node.status == True:
        print("%s%s" % (pre, node.name))

test_score = 0.0
search_default = None
try:
    search_default = MyAutoML(n_jobs=1,
                      time_search_budget=60,
                      space=space_new,
                      evaluation_budget=int(0.1 * 60),
                      main_memory_budget_gb=10,
                      hold_out_fraction=0.33
                      )

    best_result = search_default.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=my_scorer)

    test_score = my_scorer(search_default.get_best_pipeline(), X_test, y_test)

except:
    test_score = 0.0

print('my system: ' + str(result))
print('default system: ' + str(test_score))
