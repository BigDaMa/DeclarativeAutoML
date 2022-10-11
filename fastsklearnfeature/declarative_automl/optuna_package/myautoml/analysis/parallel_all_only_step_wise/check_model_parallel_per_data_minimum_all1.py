import optuna
from sklearn.metrics import make_scorer
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import optimize_accuracy_under_minimal_sample_ensemble
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import utils_run_AutoML_ensemble
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.util_classes_new import ConstraintEvaluation, ConstraintRun, space2str
from anytree import RenderTree
import argparse
import openml
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.my_global_vars as mp_global
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestRegressor

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



print(args)

memory_budget = 500.0
privacy = None

for test_holdout_dataset_id in [args.dataset]:

    X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data('data', randomstate=42, task_id=test_holdout_dataset_id)
    metafeature_values_hold = data2features(X_train_hold, y_train_hold, categorical_indicator_hold)

    #X_surrogate = pickle.load(open('/home/neutatz/data/my_temp/felix_X_compare_scaled.p', "rb"))
    #y_surrogate = pickle.load(open('/home/neutatz/data/my_temp/felix_y_compare_scaled.p', "rb"))

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

    _, feature_names = get_feature_names(my_list_constraints)

    #plot_most_important_features(model, feature_names, k=len(feature_names))

    dynamic_approach = []

    new_constraint_evaluation_dynamic_all = []

    for days_surrogate in [1, 2, 3]:#[1, 2, 3, 4, 5, 6, 7]:#[1, 5, 10, 60]:#range(1, 6):

        #model_success = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=1)
        #model_success.fit(X_surrogate[0: int((days_surrogate / 7.0) * len(X_surrogate)), :], y_surrogate[0: int((days_surrogate / 7.0) * len(X_surrogate))])

        model_success = pickle.load(open('/home/neutatz/data/my_temp/my_great_model_compare_scaled' + str(days_surrogate) + '.p', "rb"))

        '''
        X_day = pickle.load(
            open('/home/neutatz/data/28_sep_incremental_days_al_m1/day' + str(days_surrogate) + '/felix_X_compare_scaled.p', "rb"))
        y_day = pickle.load(
            open('/home/neutatz/data/28_sep_incremental_days_al_m1/day' + str(days_surrogate) + '/felix_y_compare_scaled.p', "rb"))

        model_success.fit(X_day, y_day)
        '''

        current_dynamic = []

        minutes_to_search = 5
        search_time_frozen = minutes_to_search * 60

        new_constraint_evaluation_dynamic = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                 constraint={'days_surrogate': days_surrogate},
                                                                 system_def='dynamic')
        for repeat in range(10):

            mp_global.study_prune = optuna.create_study(direction='maximize')
            mp_global.study_prune.optimize(lambda trial: optimize_accuracy_under_minimal_sample_ensemble(trial=trial,
                                                                                   metafeature_values_hold=metafeature_values_hold,
                                                                                   search_time=search_time_frozen,
                                                                                   model_success=model_success,
                                                                                   memory_limit=memory_budget,
                                                                                   privacy_limit=privacy,
                                                                                   #evaluation_time=int(0.1*search_time_frozen),
                                                                                   #hold_out_fraction=0.33,
                                                                                   tune_space=True,
                                                                                   tune_val_fraction=True
                                                                                   ), n_trials=1000, n_jobs=1)

            space = mp_global.study_prune.best_trial.user_attrs['space']

            for pre, _, node in RenderTree(space.parameter_tree):
                if node.status == True:
                    print("%s%s" % (pre, node.name))

            try:
                result = None
                search_dynamic = None

                result, search_dynamic = utils_run_AutoML_ensemble(mp_global.study_prune.best_trial,
                                                             X_train=X_train_hold,
                                                             X_test=X_test_hold,
                                                             y_train=y_train_hold,
                                                             y_test=y_test_hold,
                                                             categorical_indicator=categorical_indicator_hold,
                                                             my_scorer=my_scorer,
                                                             search_time=search_time_frozen,
                                                             memory_limit=memory_budget,
                                                             privacy_limit=privacy
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


    results_dict_log = {}
    results_dict_log['dynamic'] = new_constraint_evaluation_dynamic_all

    pickle.dump(results_dict_log, open('/home/neutatz/data/automl_runs/log_' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))

    results_dict = {}
    results_dict['dynamic'] = dynamic_approach

    pickle.dump(results_dict, open('/home/neutatz/data/automl_runs/' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))