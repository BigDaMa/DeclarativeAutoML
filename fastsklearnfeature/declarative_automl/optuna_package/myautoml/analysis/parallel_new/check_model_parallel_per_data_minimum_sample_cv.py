from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.MyAutoMLProcessClassBalance import MyAutoML
import optuna
from sklearn.metrics import make_scorer
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import optimize_accuracy_under_minimal_sample
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import utils_run_AutoML
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.util_classes import ConstraintEvaluation, ConstraintRun
from anytree import RenderTree
import argparse
import openml
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.my_global_vars as mp_global
from sklearn.metrics import balanced_accuracy_score

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

    #model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/training_sampling_min_2Drandom_machine2/my_great_model_compare_scaled.p', "rb"))
    #model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/july30_machine1/my_great_model_compare_scaled.p', "rb"))
    #model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/july30_machine4/my_great_model_compare_scaled.p', "rb"))
    model_success = pickle.load(open('/tmp/my_great_model_compare_scaled.p', "rb"))
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
                           'pipeline_size_constraint']

    _, feature_names = get_feature_names(my_list_constraints)

    #plot_most_important_features(model, feature_names, k=len(feature_names))

    dynamic_approach = []
    static_approach = []

    new_constraint_evaluation_dynamic_all = []
    new_constraint_evaluation_default_all = []

    for minutes_to_search in [1]:#range(1, 6):

        current_dynamic = []
        current_static = []

        search_time_frozen = minutes_to_search * 60

        new_constraint_evaluation_dynamic = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                 constraint={'search_time': minutes_to_search},
                                                                 system_def='dynamic')
        new_constraint_evaluation_default = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                 constraint={'search_time': minutes_to_search},
                                                                 system_def='default')

        for repeat in range(10):

            mp_global.study_prune = optuna.create_study(direction='maximize')
            mp_global.study_prune.optimize(lambda trial: optimize_accuracy_under_minimal_sample(trial=trial,
                                                                                   metafeature_values_hold=metafeature_values_hold,
                                                                                   search_time=search_time_frozen,
                                                                                   model_success=model_success,
                                                                                   memory_limit=memory_budget,
                                                                                   privacy_limit=privacy,
                                                                                   #evaluation_time=int(0.1*search_time_frozen),
                                                                                   #hold_out_fraction=0.33,
                                                                                   tune_space=True,
                                                                                   tune_cv=True
                                                                                   ), n_trials=1000, n_jobs=1)

            space = mp_global.study_prune.best_trial.user_attrs['space']



            for pre, _, node in RenderTree(space.parameter_tree):
                if node.status == True:
                    print("%s%s" % (pre, node.name))

            try:
                result, search_dynamic = utils_run_AutoML(mp_global.study_prune.best_trial,
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
                new_constraint_evaluation_dynamic.append(ConstraintRun(space, search_dynamic.study.best_trial, result, more=mp_global.study_prune.best_trial))
            except:
                result = 0

                new_constraint_evaluation_dynamic.append(ConstraintRun(space, 'shit happened', result, more=mp_global.study_prune.best_trial))

            print("test result: " + str(result))
            current_dynamic.append(result)

            print('dynamic: ' + str(current_dynamic))
            print('static: ' + str(current_static))


            gen_new = SpaceGenerator()
            space_new = gen_new.generate_params()
            for pre, _, node in RenderTree(space_new.parameter_tree):
                if node.status == True:
                    print("%s%s" % (pre, node.name))

            try:
                search_default = MyAutoML(n_jobs=1,
                                  time_search_budget=search_time_frozen,
                                  space=space_new,
                                  evaluation_budget=int(0.1 * search_time_frozen),
                                  main_memory_budget_gb=memory_budget,
                                  differential_privacy_epsilon=privacy,
                                  hold_out_fraction=0.33
                                  )

                best_result = search_default.fit(X_train_hold, y_train_hold, categorical_indicator=categorical_indicator_hold, scorer=my_scorer)

                test_score = my_scorer(search_default.get_best_pipeline(), X_test_hold, y_test_hold)

                new_constraint_evaluation_default.append(ConstraintRun('default_space', search_default.study.best_trial, test_score))
            except:
                test_score = 0.0

                new_constraint_evaluation_default.append(ConstraintRun('default_space', 'shit happened', test_score))
            current_static.append(test_score)

            print("result: " + str(best_result) + " test: " + str(test_score))

            print('dynamic: ' + str(current_dynamic))
            print('static: ' + str(current_static))

        dynamic_approach.append(current_dynamic)
        static_approach.append(current_static)

        new_constraint_evaluation_dynamic_all.append(new_constraint_evaluation_dynamic)
        new_constraint_evaluation_default_all.append(new_constraint_evaluation_default)

        print('dynamic: ' + str(dynamic_approach))
        print('static: ' + str(static_approach))


    results_dict_log = {}
    results_dict_log['dynamic'] = new_constraint_evaluation_dynamic_all
    results_dict_log['static'] = new_constraint_evaluation_default_all

    pickle.dump(results_dict_log, open('/home/neutatz/data/automl_runs/log_' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))

    results_dict = {}
    results_dict['dynamic'] = dynamic_approach
    results_dict['static'] = static_approach

    pickle.dump(results_dict, open('/home/neutatz/data/automl_runs/' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))