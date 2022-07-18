from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.MyAutoMLProcessClassBalanceDict import MyAutoML
import optuna
import time
from sklearn.metrics import make_scorer
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
import copy
from anytree import RenderTree
from sklearn.ensemble import RandomForestRegressor
from optuna.samplers import RandomSampler
import pickle
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.mp_global_vars as mp_glob
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.feature_transformation.FeatureTransformations import FeatureTransformations
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import space2features
#from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import MyPool
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import ifNull
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import generate_parameters_minimal_sample
from optuna.samplers import TPESampler
import multiprocessing as mp
from multiprocessing import Lock
import openml
from sklearn.metrics import balanced_accuracy_score
import random
from fastsklearnfeature.declarative_automl.fair_data.test import get_X_y_id
import sklearn
import multiprocessing

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)

openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'

def predict_range(model, X):
    y_pred = model.predict(X)
    return y_pred

map_dataset = {}
map_dataset['31'] = 'foreign_worker@{yes,no}'
map_dataset['802'] = 'sex@{female,male}'
map_dataset['1590'] = 'sex@{Female,Male}'
map_dataset['1461'] = 'AGE@{True,False}'
map_dataset['42193'] = 'race_Caucasian@{0,1}'
map_dataset['1480'] = 'V2@{Female,Male}'
# map_dataset['804'] = 'Gender@{0,1}'
map_dataset['42178'] = 'gender@STRING'
map_dataset['981'] = 'Gender@{Female,Male}'
map_dataset['40536'] = 'samerace@{0,1}'
map_dataset['40945'] = 'sex@{female,male}'
map_dataset['451'] = 'Sex@{female,male}'
# map_dataset['945'] = 'sex@{female,male}'
map_dataset['446'] = 'sex@{Female,Male}'
map_dataset['1017'] = 'sex@{0,1}'
map_dataset['957'] = 'Sex@{0,1}'
map_dataset['41430'] = 'SEX@{True,False}'
map_dataset['1240'] = 'sex@{Female,Male}'
map_dataset['1018'] = 'sex@{Female,Male}'
# map_dataset['55'] = 'SEX@{male,female}'
map_dataset['38'] = 'sex@{F,M}'
map_dataset['1003'] = 'sex@{male,female}'
map_dataset['934'] = 'race@{black,white}'

my_openml_tasks = [75126, 75125, 75121, 75120, 75116, 75115, 75114, 189859, 189878, 189786, 167204, 190156, 75156, 166996, 190157, 190158, 168791, 146597, 167203, 167085, 190154, 75098, 190159, 75169, 126030, 146594, 211723, 189864, 189863, 189858, 75236, 190155, 211720, 167202, 75108, 146679, 146592, 166866, 167205, 2356, 75225, 146576, 166970, 258, 75154, 146574, 275, 273, 75221, 75180, 166944, 166951, 189828, 3049, 75139, 167100, 75232, 126031, 189899, 75146, 288, 146600, 166953, 232, 75133, 75092, 75129, 211722, 75100, 2120, 189844, 271, 75217, 146601, 75212, 75153, 75109, 189870, 75179, 146596, 75215, 189840, 3044, 168785, 189779, 75136, 75199, 75235, 189841, 189845, 189869, 254, 166875, 75093, 75159, 146583, 75233, 75089, 167086, 167087, 166905, 167088, 167089, 167097, 167106, 189875, 167090, 211724, 75234, 75187, 2125, 75184, 166897, 2123, 75174, 75196, 189829, 262, 236, 75178, 75219, 75185, 126021, 211721, 3047, 75147, 189900, 75118, 146602, 166906, 189836, 189843, 75112, 75195, 167101, 167094, 75149, 340, 166950, 260, 146593, 75142, 75161, 166859, 166915, 279, 245, 167096, 253, 146578, 267, 2121, 75141, 336, 166913, 75176, 256, 75166, 2119, 75171, 75143, 75134, 166872, 166932, 146603, 126028, 3055, 75148, 75223, 3054, 167103, 75173, 166882, 3048, 3053, 2122, 75163, 167105, 75131, 126024, 75192, 75213, 146575, 166931, 166957, 166956, 75250, 146577, 146586, 166959, 75210, 241, 166958, 189902, 75237, 189846, 75157, 189893, 189890, 189887, 189884, 189883, 189882, 189881, 189880, 167099, 189894]

my_scorer = make_scorer(balanced_accuracy_score)


mp_glob.total_search_time = 60*60#60
topk = 28#28#26 # 20
continue_from_checkpoint = False

starting_time_tt = time.time()

my_lock = Lock()

mgr = mp.Manager()
dictionary = mgr.dict()


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

print(len(feature_names_new))

def run_AutoML(trial):
    space = trial.user_attrs['space']

    print(trial.params)

    #make this a hyperparameter
    search_time = trial.params['global_search_time_constraint']# * 60

    evaluation_time = int(0.1 * search_time)
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
        task_id = trial.params['dataset_id'] #get same random seed

    for pre, _, node in RenderTree(space.parameter_tree):
        if node.status == True:
            print("%s%s" % (pre, node.name))

    my_random_seed = int(time.time())
    if 'data_random_seed' in trial.user_attrs:
        my_random_seed = trial.user_attrs['data_random_seed']

    try:
        X, y, sensitive_attribute_id, categorical_indicator = get_X_y_id(key=trial.params['fair_dataset_id'])
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,
                                                                                    y,
                                                                                    random_state=my_random_seed,
                                                                                    stratify=y,
                                                                                    train_size=0.66)
    except:
        return {'objective': 0.0}


    # default params
    gen_new = SpaceGenerator()
    space_new = gen_new.generate_params()
    for pre, _, node in RenderTree(space_new.parameter_tree):
        if node.status == True:
            print("%s%s" % (pre, node.name))

    search_time = 60*60

    search_static = MyAutoML(n_jobs=1,
                             time_search_budget=search_time,
                             space=space_new,
                             evaluation_budget=int(0.1 * search_time),
                             hold_out_fraction=0.33,
                             fairness_group_id=sensitive_attribute_id,
                             fairness_limit=0.5
                             )

    try:
        search_static.fit(X_train, y_train, categorical_indicator=categorical_indicator,
                                        scorer=my_scorer)



        my_lock.acquire()

        fairness_list = []
        for t in search_static.study.trials:
            if 'fairness' in t.user_attrs:
                fairness_list.append(t.user_attrs['fairness'])

        fairness_stored = dictionary['fairness']
        fairness_stored = np.append(fairness_stored, np.array(fairness_list))
        dictionary['fairness'] = fairness_stored


        '''
        training_time_stored = dictionary['training_time']       
        training_time_stored = np.append(training_time_stored, np.array(search_static.store_training_times))
        dictionary['training_time'] = training_time_stored

        inference_time_stored = dictionary['inference_time']
        inference_time_stored = np.append(inference_time_stored, np.array(search_static.store_inference_times))
        dictionary['inference_time'] = inference_time_stored

        pipeline_size_stored = copy.deepcopy(dictionary['pipeline_size'])
        pipeline_size_stored = np.append(pipeline_size_stored, np.array(search_static.store_pipeline_sizes))
        dictionary['pipeline_size'] = pipeline_size_stored
        '''

        my_lock.release()



    except:
        test_score_default = 0.0

    return {'objective': 0.0}



def sample_configuration(trial):
    try:
        gen = SpaceGenerator()
        space = gen.generate_params()
        space.sample_parameters(trial) # no tuning of the space

        trial.set_user_attr('space', copy.deepcopy(space))

        search_time, evaluation_time, memory_limit, privacy_limit, training_time_limit, inference_time_limit, pipeline_size_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, task_id = generate_parameters_minimal_sample(
            trial, mp_glob.total_search_time, my_openml_tasks)

        fair_dataset_id = trial.suggest_categorical('fair_dataset_id', list(map_dataset.keys()))

        my_random_seed = int(time.time())

        X, y, sensitive_attribute_id, categorical_indicator = get_X_y_id(key=fair_dataset_id)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,
                                                                                    y,
                                                                                    random_state=my_random_seed,
                                                                                    stratify=y,
                                                                                    train_size=0.66)

        trial.set_user_attr('data_random_seed', my_random_seed)

        # add metafeatures of data
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
    except:
        return None
    return features

def random_config(trial):
    features = sample_configuration(trial)
    if type(features) == type(None):
        return -1 * np.inf
    return 0.0


X_meta = np.empty((0, len(feature_names_new)), dtype=float)
y_meta = []
group_meta = []
aquisition_function_value = []


#path2files = '/home/neutatz/phd2/decAutoML2weeks_compare2default/single_cpu_machine1_4D_start_and_class_imbalance'
path2files = '/tmp'


def get_best_trial():
    sampler = RandomSampler()
    study_uncertainty = optuna.create_study(sampler=sampler)
    study_uncertainty.optimize(random_config, n_trials=1, n_jobs=1)
    return study_uncertainty.best_trial

def sample_and_evaluate(my_id1):
    if time.time() - starting_time_tt > 60*60:#60*60*1:
        print('done')
        return -1

    best_trial = get_best_trial()
    run_AutoML(best_trial)
    return 0

assert len(X_meta) == len(y_meta)

dictionary['training_time'] = np.array([])
dictionary['inference_time'] = np.array([])
dictionary['pipeline_size'] = np.array([])
dictionary['fairness'] = np.array([])


with NestablePool(processes=topk) as pool:
    results = pool.map(sample_and_evaluate, range(100000))

print('storing stuff')


with open('/tmp/constraints_fairness.p', "wb") as pickle_model_file:
    pickle.dump(dictionary['fairness'], pickle_model_file)

'''
with open('/tmp/constraints_training_time.p', "wb") as pickle_model_file:
    pickle.dump(dictionary['training_time'], pickle_model_file)

with open('/tmp/constraints_inference_time.p', "wb") as pickle_model_file:
    pickle.dump(dictionary['inference_time'], pickle_model_file)

with open('/tmp/constraints_pipeline_size.p', "wb") as pickle_model_file:
    pickle.dump(dictionary['pipeline_size'], pickle_model_file)
'''