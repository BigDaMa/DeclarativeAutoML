import optuna
from sklearn.metrics import make_scorer
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import optimize_accuracy_under_minimal_sample
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.util_classes_new import ConstraintEvaluation, ConstraintRun, space2str
from anytree import RenderTree
import argparse
import openml
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.my_global_vars as mp_global
from sklearn.metrics import balanced_accuracy_score
import sklearn

from autosklearn.metrics import balanced_accuracy

openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'

my_scorer = make_scorer(balanced_accuracy_score)

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="OpenML datatset ID", type=int)
parser.add_argument("--outputname", "-o", help="Name of the output file")
args = parser.parse_args()
print(args.dataset)

#args.dataset = 75097
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

    for minutes_to_search in [1]:#[1, 5, 10, 60]:#range(1, 6):

        current_dynamic = []
        current_default = []

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
                                                                                   tune_space=True
                                                                                   ), n_trials=1000, n_jobs=1)

            space = mp_global.study_prune.best_trial.user_attrs['space']

            for pre, _, node in RenderTree(space.parameter_tree):
                if node.status == True:
                    print("%s%s" % (pre, node.name))

            feat_type = []
            for c_i in range(len(categorical_indicator_hold)):
                if categorical_indicator_hold[c_i]:
                    feat_type.append('Categorical')
                else:
                    feat_type.append('Numerical')

            try:
                result = None

                from autosklearn.experimental.askl2 import AutoSklearn2Classifier
                from autosklearn.flexible.Config import Config

                if mp_global.study_prune.best_trial.value > 0.5:
                    Config.config = space.name2node
                    Config.setup()
                    automl = AutoSklearn2Classifier(
                        time_left_for_this_task=search_time_frozen,
                        per_run_time_limit=int(0.1 * search_time_frozen),
                        delete_tmp_folder_after_terminate=True,
                        metric=balanced_accuracy,
                        seed=repeat,
                        memory_limit=1024 * 250
                    )

                    sample_fraction = mp_global.study_prune.best_trial.params['sample_fraction']

                    X_train_sample = X_train_hold
                    y_train_sample = y_train_hold
                    if sample_fraction < 1.0:
                        X_train_sample, _, y_train_sample, _ = sklearn.model_selection.train_test_split(X_train_hold,
                                                                                                        y_train_hold,
                                                                                                        random_state=42,
                                                                                                        stratify=y_train_hold,
                                                                                                        train_size=sample_fraction)

                    automl.fit(X_train_sample.copy(), y_train_sample.copy(), feat_type=feat_type,
                               metric=balanced_accuracy)
                    automl.refit(X_train_sample.copy(), y_train_sample.copy())
                    result = my_scorer(automl, X_test_hold, y_test_hold)
                else:
                    Config.config = dict()
                    Config.setup()
                    automl = AutoSklearn2Classifier(
                        time_left_for_this_task=search_time_frozen,
                        per_run_time_limit=int(0.1 * search_time_frozen),
                        delete_tmp_folder_after_terminate=True,
                        metric=balanced_accuracy,
                        seed=repeat,
                        memory_limit=1024 * 250
                    )

                    X_train_sample = X_train_hold
                    y_train_sample = y_train_hold

                    automl.fit(X_train_sample.copy(), y_train_sample.copy(), feat_type=feat_type, metric=balanced_accuracy)
                    automl.refit(X_train_sample.copy(), y_train_sample.copy())
                    result = my_scorer(automl, X_test_hold, y_test_hold)

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


        for repeat in range(10):

            feat_type = []
            for c_i in range(len(categorical_indicator_hold)):
                if categorical_indicator_hold[c_i]:
                    feat_type.append('Categorical')
                else:
                    feat_type.append('Numerical')

            try:
                result = None

                from autosklearn.experimental.askl2 import AutoSklearn2Classifier
                from autosklearn.flexible.Config import Config

                Config.config = dict()
                Config.setup()
                automl = AutoSklearn2Classifier(
                    time_left_for_this_task=search_time_frozen,
                    per_run_time_limit=int(0.1 * search_time_frozen),
                    delete_tmp_folder_after_terminate=True,
                    metric=balanced_accuracy,
                    seed=repeat,
                    memory_limit=1024 * 250
                )

                X_train_sample = X_train_hold
                y_train_sample = y_train_hold

                automl.fit(X_train_sample.copy(), y_train_sample.copy(), feat_type=feat_type, metric=balanced_accuracy)
                automl.refit(X_train_sample.copy(), y_train_sample.copy())
                result = my_scorer(automl, X_test_hold, y_test_hold)

                new_constraint_evaluation_default.append(ConstraintRun(space_str='full', params='None', test_score=result, estimated_score=0.0))
            except:
                result = 0
                new_constraint_evaluation_default.append(ConstraintRun(space_str='full', params='None', test_score=result, estimated_score=0.0))

            print("test result: " + str(result))
            current_default.append(result)

            print('default: ' + str(current_default))

        static_approach.append(current_default)
        new_constraint_evaluation_default_all.append(new_constraint_evaluation_default)

        print('default ' + str(static_approach))




    results_dict_log = {}
    results_dict_log['dynamic'] = new_constraint_evaluation_dynamic_all
    results_dict_log['static'] = new_constraint_evaluation_default_all

    pickle.dump(results_dict_log, open('/home/neutatz/data/automl_runs/log_' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))

    results_dict = {}
    results_dict['dynamic'] = dynamic_approach
    results_dict['static'] = static_approach

    pickle.dump(results_dict, open('/home/neutatz/data/automl_runs/' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))

