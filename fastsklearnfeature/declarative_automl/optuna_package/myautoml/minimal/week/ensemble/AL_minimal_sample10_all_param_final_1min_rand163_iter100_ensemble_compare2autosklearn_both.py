import os
import shutil
os.environ['OPENBLAS_NUM_THREADS'] = '1'
#from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.ensemble.AutoEnsemble import MyAutoML
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.ensemble.AutoEnsembleSuccessive import MyAutoML as AutoSuccess
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
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import ifNull
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import generate_parameters_minimal_sample_ensemble
from optuna.samplers import TPESampler
import multiprocessing
from multiprocessing import Lock
import openml
from sklearn.metrics import balanced_accuracy_score
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
import autosklearn
import getpass
import libtmux


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
openml.config.cache_directory = '/home/' + getpass.getuser() + '/phd2/cache_openml'

def predict_range(model, X):
    y_pred = model.predict(X)
    return y_pred

my_openml_tasks = [75126, 75125, 75121, 75120, 75116, 75115, 75114, 189859, 189878, 189786, 167204, 190156, 75156, 166996, 190157, 190158, 168791, 146597, 167203, 167085, 190154, 75098, 190159, 75169, 126030, 146594, 211723, 189864, 189863, 189858, 75236, 190155, 211720, 167202, 75108, 146679, 146592, 166866, 167205, 2356, 75225, 146576, 166970, 258, 75154, 146574, 275, 273, 75221, 75180, 166944, 166951, 189828, 3049, 75139, 167100, 75232, 126031, 189899, 75146, 288, 146600, 166953, 232, 75133, 75092, 75129, 211722, 75100, 2120, 189844, 271, 75217, 146601, 75212, 75153, 75109, 189870, 75179, 146596, 75215, 189840, 3044, 168785, 189779, 75136, 75199, 75235, 189841, 189845, 189869, 254, 166875, 75093, 75159, 146583, 75233, 75089, 167086, 167087, 166905, 167088, 167089, 167097, 167106, 189875, 167090, 211724, 75234, 75187, 2125, 75184, 166897, 2123, 75174, 75196, 189829, 262, 236, 75178, 75219, 75185, 126021, 211721, 3047, 75147, 189900, 75118, 146602, 166906, 189836, 189843, 75112, 75195, 167101, 167094, 75149, 340, 166950, 260, 146593, 75142, 75161, 166859, 166915, 279, 245, 167096, 253, 146578, 267, 2121, 75141, 336, 166913, 75176, 256, 75166, 2119, 75171, 75143, 75134, 166872, 166932, 146603, 126028, 3055, 75148, 75223, 3054, 167103, 75173, 166882, 3048, 3053, 2122, 75163, 167105, 75131, 126024, 75192, 75213, 146575, 166931, 166957, 166956, 75250, 146577, 146586, 166959, 75210, 241, 166958, 189902, 75237, 189846, 75157, 189893, 189890, 189887, 189884, 189883, 189882, 189881, 189880, 167099, 189894]

my_scorer = make_scorer(balanced_accuracy_score)


mp_glob.total_search_time = 5*60#60
topk = 20 # 20
continue_from_checkpoint = False

starting_time_tt = time.time()

my_lock = Lock()

mgr = multiprocessing.Manager()
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
                           'pipeline_size_constraint',
                       'use_ensemble',
                       'use_incremental_data'
                       ]

feature_names, feature_names_new = get_feature_names(my_list_constraints)

print(len(feature_names_new))

random_runs = (163)


def run_AutoML(trial):
    repetitions_count = 10

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

    ensemble_size = 50
    if not trial.params['use_ensemble']:
        ensemble_size = 1

    use_incremental_data = trial.params['use_incremental_data']

    for pre, _, node in RenderTree(space.parameter_tree):
        if node.status == True:
            print("%s%s" % (pre, node.name))

    my_random_seed = int(time.time())
    if 'data_random_seed' in trial.user_attrs:
        my_random_seed = trial.user_attrs['data_random_seed']

    try:
        X_train, X_test, y_train, y_test, categorical_indicator, attribute_names = get_data('data', randomstate=my_random_seed, task_id=task_id)
    except:
        return {'objective': 0.0}

    dynamic_params = []
    for random_i in range(repetitions_count):
        search = AutoSuccess(cv=cv,
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
                          pipeline_size_limit=pipeline_size_limit,
                          max_ensemble_models=ensemble_size,
                          use_incremental_data=use_incremental_data)

        test_score = 0.0
        try:
            search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=my_scorer)

            search.ensemble(X_train, y_train)
            y_hat_test = search.ensemble_predict(X_test)
            test_score = balanced_accuracy_score(y_test, y_hat_test)
        except:
            pass
        dynamic_params.append(test_score)
    dynamic_values = np.array(dynamic_params)

    if np.sum(dynamic_values) == 0:
        return {'objective': 0.0}

    static_params = []
    for random_i in range(repetitions_count):
        # default params
        gen_new = SpaceGenerator()
        space_new = gen_new.generate_params()
        for pre, _, node in RenderTree(space_new.parameter_tree):
            if node.status == True:
                print("%s%s" % (pre, node.name))

        search_static = AutoSuccess(n_jobs=1,
                                    time_search_budget=search_time,
                                    space=space_new,
                                    evaluation_budget=int(0.1 * search_time),
                                    main_memory_budget_gb=memory_limit,
                                    differential_privacy_epsilon=privacy_limit,
                                    hold_out_fraction=0.33,
                                    training_time_limit=training_time_limit,
                                    inference_time_limit=inference_time_limit,
                                    pipeline_size_limit=pipeline_size_limit,
                                    max_ensemble_models=50
                                    )

        test_score_default = 0.0
        try:
            best_result = search_static.fit(X_train, y_train, categorical_indicator=categorical_indicator,
                                            scorer=my_scorer)

            search_static.ensemble(X_train, y_train)
            y_hat_test = search_static.ensemble_predict(X_test)
            test_score_default = balanced_accuracy_score(y_test, y_hat_test)
        except:
            test_score_default = 0.0
        static_params.append(test_score_default)

    static_values = np.array(static_params)


    #autosklearn
    current_id_name = str(str(time.time()) + '_' + str(np.random.randint(100))).replace('.', '_')
    tmp_path_pickle = "/home/" + getpass.getuser() + "/data/auto_tmp/autosklearn" + current_id_name

    send_data = {}
    send_data['categorical_indicator'] = categorical_indicator
    send_data['search_time'] = search_time
    send_data['X_train'] = X_train
    send_data['y_train'] = y_train
    send_data['X_test'] = X_test
    send_data['y_test'] = y_test

    static_values_autosklearn = copy.deepcopy(static_values)
    try:
        with open(tmp_path_pickle + ".pickle", "wb+") as output_file:
            pickle.dump(send_data, output_file)

        server = libtmux.Server()
        conda_name = 'AutoMLD'
        session = server.new_session(session_name="data" + current_id_name, kill_session=True, attach=False)
        session.attached_pane.send_keys('exec bash')
        session.attached_pane.send_keys('conda activate ' + conda_name)
        session.attached_pane.send_keys('cd /home/' + getpass.getuser() + '/Software/DeclarativeAutoML')
        session.attached_pane.send_keys("python /home/" + getpass.getuser() + "/Software/DeclarativeAutoML/fastsklearnfeature/declarative_automl/optuna_package/myautoml/minimal/week/ensemble/run_autosklearn.py " + str(tmp_path_pickle))
        session.attached_pane.send_keys('exit')

        while True:
            if os.path.exists(tmp_path_pickle + "_result.pickle"):
                time.sleep(60)
                with open(tmp_path_pickle + "_result.pickle", 'rb') as result_file:
                    static_values_autosklearn = pickle.load(result_file)['static_values']
                    print('########################################################')
                    print('autosklearn result: ' + str(static_values_autosklearn))
                    print('********************************************************')
                    break
            else:
                time.sleep(10)

    except:
        pass

    if os.path.exists(tmp_path_pickle + "_result.pickle"):
        os.remove(tmp_path_pickle + "_result.pickle")

    if os.path.exists(tmp_path_pickle + ".pickle"):
        os.remove(tmp_path_pickle + "_result.pickle")

    if np.mean(static_values) < np.mean(static_values_autosklearn):
        static_values = static_values_autosklearn

    dynamic_values.sort()
    static_values.sort()

    frequency = np.sum(dynamic_values > static_values) / float(repetitions_count)
    return {'objective': frequency}


def run_AutoML_global(trial_id):
    comparison = run_AutoML(mp_glob.my_trials[trial_id])['objective']

    # get all best validation scores and evaluate them
    feature_list = []
    target_list = []

    feature_list.append(copy.deepcopy(mp_glob.my_trials[trial_id].user_attrs['features']))
    target_list.append(comparison)

    return {'feature_l': feature_list,
            'target_l': target_list,
            'group_l': copy.deepcopy(mp_glob.my_trials[trial_id].params['dataset_id']),
            'trial_id_l': trial_id}


def calculate_max_std(N, min_value=0, max_value=1):
    max_elements = np.ones(int(N / 2)) * max_value
    min_elements = np.ones(int(N / 2)) * min_value
    return np.std(np.append(min_elements, max_elements))


def sample_configuration(trial):
    try:
        gen = SpaceGenerator()
        space = gen.generate_params()
        space.sample_parameters(trial) # no tuning of the space

        trial.set_user_attr('space', copy.deepcopy(space))

        search_time, evaluation_time, memory_limit, privacy_limit, training_time_limit, inference_time_limit, pipeline_size_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, task_id, use_ensemble, use_incremental_data = generate_parameters_minimal_sample_ensemble(
            trial, mp_glob.total_search_time, my_openml_tasks)

        my_random_seed = int(time.time())

        X_train, X_test, y_train, y_test, categorical_indicator, attribute_names = get_data('data', randomstate=my_random_seed, task_id=task_id)

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
                                      pipeline_size_limit,
                                      int(use_ensemble),
                                      int(use_incremental_data)]

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


path2files = '/home/' + getpass.getuser() + '/data/my_temp'


if continue_from_checkpoint:
    with open(path2files + '/felix_X_compare_scaled.p', 'rb') as handle:
        X_meta = pickle.load(handle)

    with open(path2files + '/felix_y_compare_scaled.p', 'rb') as handle:
        y_meta = pickle.load(handle)

    with open(path2files + '/felix_group_compare_scaled.p', 'rb') as handle:
        group_meta = pickle.load(handle)

    with open(path2files + '/felix_acquisition function value_scaled.p', 'rb') as handle:
        aquisition_function_value = pickle.load(handle)
else:
    #cold start - random sampling
    study_random = optuna.create_study(direction='maximize', sampler=RandomSampler(seed=42))
    study_random.optimize(random_config, n_trials=random_runs, n_jobs=1)

    mp_glob.my_trials = []
    trial_id2aqval = {}
    counter_trial_id = 0
    for u_trial in study_random.trials:
        if u_trial.value == 0.0: #prune failed configurations
            mp_glob.my_trials.append(u_trial)
            trial_id2aqval[counter_trial_id] = 0.0
            counter_trial_id += 1

    with NestablePool(processes=topk) as pool:
        results = pool.map(run_AutoML_global, range(len(mp_glob.my_trials)))

    for result_p in results:
        for f_progress in range(len(result_p['feature_l'])):
            X_meta = np.vstack((X_meta, result_p['feature_l'][f_progress]))
            y_meta.append(result_p['target_l'][f_progress])
            group_meta.append(result_p['group_l'])
            aquisition_function_value.append(trial_id2aqval[result_p['trial_id_l']])

    print('done')


class Objective(object):
    def __init__(self, model_uncertainty):
        self.model_uncertainty = model_uncertainty

    def __call__(self, trial):
        features = sample_configuration(trial)
        if type(features) == type(None):
            return -1 * np.inf

        predictions = []
        for tree in range(self.model_uncertainty.n_estimators):
            predictions.append(predict_range(self.model_uncertainty.estimators_[tree], features))

        stddev_pred = np.std(np.matrix(predictions).transpose(), axis=1)
        uncertainty = stddev_pred[0]

        objective = uncertainty
        return objective

def get_best_trial(model_uncertainty):
    sampler = TPESampler()
    study_uncertainty = optuna.create_study(direction='maximize', sampler=sampler)
    my_objective = Objective(model_uncertainty)
    study_uncertainty.optimize(my_objective, n_trials=100, n_jobs=1)
    return study_uncertainty.best_trial

def sample_and_evaluate(my_id1):
    if time.time() - starting_time_tt > 60*60*24*3:
        return -1

    X_meta = copy.deepcopy(dictionary['X_meta'])
    y_meta = copy.deepcopy(dictionary['y_meta'])

    # how many are there
    my_len = min(len(X_meta), len(y_meta))
    X_meta = X_meta[0:my_len, :]
    y_meta = y_meta[0:my_len]

    #assert len(X_meta) == len(y_meta), 'len(X) != len(y)'
    try:
        model_uncertainty = RandomForestRegressor(n_estimators=1000, random_state=my_id1, n_jobs=1)
        model_uncertainty.fit(X_meta, y_meta)

        best_trial = get_best_trial(model_uncertainty)
        features_of_sampled_point = best_trial.user_attrs['features']

        result = run_AutoML(best_trial)
        actual_y = result['objective']
    except Exception as e:
        print('exception: ' + str(e))
        return 0

    my_lock.acquire()
    try:
        X_meta = dictionary['X_meta']
        dictionary['X_meta'] = np.vstack((X_meta, features_of_sampled_point))

        y_meta = dictionary['y_meta']
        y_meta.append(actual_y)
        dictionary['y_meta'] = y_meta

        #assert len(X_meta) == len(y_meta), 'len(X) != len(y)'

        group_meta = dictionary['group_meta']
        group_meta.append(best_trial.params['dataset_id'])
        dictionary['group_meta'] = group_meta

        #assert len(X_meta) == len(group_meta), 'len(X) != len(group)'

        aquisition_function_value = dictionary['aquisition_function_value']
        aquisition_function_value.append(best_trial.value)
        dictionary['aquisition_function_value'] = aquisition_function_value

        #assert len(X_meta) == len(aquisition_function_value), 'len(X) != len(acquisition)'
    except Exception as e:
        print('exception: ' + str(e))
    finally:
        my_lock.release()

    return 0

assert len(X_meta) == len(y_meta)

dictionary['X_meta'] = X_meta
dictionary['y_meta'] = y_meta
dictionary['group_meta'] = group_meta
dictionary['aquisition_function_value'] = aquisition_function_value

with NestablePool(processes=topk) as pool:
    results = pool.map(sample_and_evaluate, range(100000))

print('storing stuff')

model_uncertainty = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=1)
model_uncertainty.fit(dictionary['X_meta'], dictionary['y_meta'])

with open('/home/' + getpass.getuser() + '/data/my_temp/my_great_model_compare_scaled.p', "wb") as pickle_model_file:
    pickle.dump(model_uncertainty, pickle_model_file)

with open('/home/' + getpass.getuser() + '/data/my_temp/felix_X_compare_scaled.p', "wb") as pickle_model_file:
    pickle.dump(dictionary['X_meta'], pickle_model_file)

with open('/home/' + getpass.getuser() + '/data/my_temp/felix_y_compare_scaled.p', "wb") as pickle_model_file:
    pickle.dump(dictionary['y_meta'], pickle_model_file)

with open('/home/' + getpass.getuser() + '/data/my_temp/felix_group_compare_scaled.p', "wb") as pickle_model_file:
    pickle.dump(dictionary['group_meta'], pickle_model_file)

with open('/home/' + getpass.getuser() + '/data/my_temp/felix_acquisition function value_scaled.p', "wb") as pickle_model_file:
    pickle.dump(dictionary['aquisition_function_value'], pickle_model_file)
