import optuna
from sklearn.metrics import make_scorer

import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import optimize_accuracy_under_minimal_sample
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import utils_run_AutoML
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.util_classes_new import ConstraintEvaluation, ConstraintRun, space2str
from anytree import RenderTree
import argparse
import openml
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.my_global_vars as mp_global
from sklearn.metrics import balanced_accuracy_score
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime
import numpy as np
import sklearn
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.cbays.new_emukit.Space_GenerationTreeBalanceConstrained2 import SpaceGenerator as EmuSpaceGenerator
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.cbays.new_emukit.Emukit import MyAutoML as AutoEmu


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

folktable_dict = [ACSEmployment,ACSIncome,ACSPublicCoverage,ACSMobility,ACSTravelTime]

print(args)

memory_budget = 500.0
privacy = None

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
ca_data = data_source.get_data(states=['CA'], download=True)

def get_race_att(data):
    for att_race in range(len(data.features)):
        if data.features[att_race] == 'RAC1P':
            return att_race

for test_holdout_dataset_id in [args.dataset]:

    task = folktable_dict[test_holdout_dataset_id]
    X, y, group = task.df_to_numpy(ca_data)
    X[:, get_race_att(task)] = X[:, get_race_att(task)] == 1.0
    sensitive_attribute_id = get_race_att(task)
    categorical_indicator_hold = np.zeros(X.shape[1], dtype=bool)

    print('label: ' + str(y))

    X_train_hold, X_test_hold, y_train_hold, y_test_hold = sklearn.model_selection.train_test_split(X, y, random_state=42, stratify=y,
                                                                                train_size=0.6)


    metafeature_values_hold = data2features(X_train_hold, y_train_hold, categorical_indicator_hold)

    #model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/training_sampling_min_2Drandom_machine2/my_great_model_compare_scaled.p', "rb"))
    #model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/july30_machine1/my_great_model_compare_scaled.p', "rb"))
    #model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/july30_machine4/my_great_model_compare_scaled.p', "rb"))
    #model_success = pickle.load(open('/home/neutatz/data/my_temp/my_great_model_compare_scaled.p', "rb"))
    #model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/sep14_sampling/my_great_model_compare_scaled.p', "rb"))

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
                           'fairness_constraint']

    _, feature_names = get_feature_names(my_list_constraints)

    #plot_most_important_features(model, feature_names, k=len(feature_names))

    dynamic_approach = []
    static_approach = []

    new_constraint_evaluation_dynamic_all = []
    new_constraint_evaluation_default_all = []

    #for minutes_to_search in [1, 5, 10, 60]:#range(1, 6):
    for fairness in [1.0, 0.999379364633232, 0.9940665258902305, 0.9807694321364909, 0.9494830824433045]:

        minutes_to_search = 5
        search_time_frozen = minutes_to_search * 60
        pipeline_size = None
        training_time = None

        '''
        current_dynamic = []
        new_constraint_evaluation_dynamic = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                 constraint={'fairness': fairness},
                                                                 system_def='dynamic')
        for repeat in range(10):

            mp_global.study_prune = optuna.create_study(direction='maximize')
            mp_global.study_prune.optimize(lambda trial: optimize_accuracy_under_minimal_sample(trial=trial,
                                                                                   metafeature_values_hold=metafeature_values_hold,
                                                                                   search_time=search_time_frozen,
                                                                                   model_success=model_success,
                                                                                   memory_limit=memory_budget,
                                                                                   privacy_limit=privacy,
                                                                                   pipeline_size_limit=pipeline_size,
                                                                                   training_time_limit=training_time,
                                                                                   #evaluation_time=int(0.1*search_time_frozen),
                                                                                   #hold_out_fraction=0.33,
                                                                                   tune_space=True
                                                                                   ), n_trials=1000, n_jobs=1)

            space = mp_global.study_prune.best_trial.user_attrs['space']

            for pre, _, node in RenderTree(space.parameter_tree):
                if node.status == True:
                    print("%s%s" % (pre, node.name))

            try:
                result = None
                search_dynamic = None

                result, search_dynamic = utils_run_AutoML(mp_global.study_prune.best_trial,
                                                             X_train=X_train_hold,
                                                             X_test=X_test_hold,
                                                             y_train=y_train_hold,
                                                             y_test=y_test_hold,
                                                             categorical_indicator=categorical_indicator_hold,
                                                             my_scorer=my_scorer,
                                                             search_time=search_time_frozen,
                                                             memory_limit=memory_budget,
                                                             privacy_limit=privacy,
                                                             pipeline_size_limit=pipeline_size,
                                                             training_time_limit=training_time
                                             )

                new_constraint_evaluation_dynamic.append(ConstraintRun(space_str=space2str(space.parameter_tree), params=mp_global.study_prune.best_trial.params, test_score=result, estimated_score=mp_global.study_prune.best_trial.value))
            except:
                result = 0
                new_constraint_evaluation_dynamic.append(ConstraintRun(space_str=space2str(space.parameter_tree), params=mp_global.study_prune.best_trial.params, test_score=result, estimated_score=mp_global.study_prune.best_trial.value))

            print("test result: " + str(result))
            current_dynamic.append(result)

            print('dynamic: ' + str(current_dynamic))

        dynamic_approach.append(current_dynamic)
        new_constraint_evaluation_dynamic_all.append(new_constraint_evaluation_dynamic)

        print('dynamic: ' + str(dynamic_approach))
        '''

        current_static = []
        new_constraint_evaluation_default = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                 constraint={'fairness': fairness},
                                                                 system_def='default')


        for repeat in range(10):
            gen_new = EmuSpaceGenerator()
            space = gen_new.generate_params(y_train_hold)

            try:
                result = None
                search_default = AutoEmu(n_jobs=1,
                                          time_search_budget=search_time_frozen,
                                          space=space,
                                          evaluation_budget=int(0.1 * search_time_frozen),
                                          main_memory_budget_gb=memory_budget,
                                          differential_privacy_epsilon=privacy,
                                          hold_out_fraction=0.33,
                                          pipeline_size_limit=pipeline_size,
                                          training_time_limit=training_time,
                                          fairness_limit=fairness,
                                          fairness_group_id=sensitive_attribute_id
                                          )

                best_result = search_default.fit(X_train_hold, y_train_hold,
                                                 categorical_indicator=categorical_indicator_hold, scorer=my_scorer)
                result = my_scorer(search_default.get_best_pipeline(), X_test_hold, y_test_hold)

                new_constraint_evaluation_default.append(
                    ConstraintRun(space_str='', params='default',
                                  test_score=result, estimated_score=0.0))
            except Exception as e:
                print('exception' + str(e))
                result = 0
                new_constraint_evaluation_default.append(
                    ConstraintRun(space_str='', params='default',
                                  test_score=result, estimated_score=0.0))

            print("test result: " + str(result))
            current_static.append(result)

        static_approach.append(current_static)
        new_constraint_evaluation_default_all.append(new_constraint_evaluation_default)


    results_dict_log = {}
    #results_dict_log['dynamic'] = new_constraint_evaluation_dynamic_all
    results_dict_log['static'] = new_constraint_evaluation_default_all
    pickle.dump(results_dict_log, open('/home/neutatz/data/automl_runs/log_' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))

    results_dict = {}
    #results_dict['dynamic'] = dynamic_approach
    results_dict['static'] = static_approach
    pickle.dump(results_dict, open('/home/neutatz/data/automl_runs/' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))