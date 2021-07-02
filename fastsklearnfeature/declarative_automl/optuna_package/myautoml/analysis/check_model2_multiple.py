import pickle
from autosklearn.metalearning.metafeatures.metafeatures import calculate_all_metafeatures_with_labels
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.MyAutoMLProcessClassBalance import MyAutoML
import optuna
import time
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
import copy
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import plot_most_important_features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import optimize_accuracy_under_constraints2
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import utils_run_AutoML
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_feature_names
from anytree import RenderTree
import numpy as np

my_scorer=make_scorer(f1_score)

test_holdout_dataset_ids = [1134, 1495, 41147, 316, 1085, 1046, 1111, 55, 1116, 448, 1458, 162, 1101, 1561, 1061, 1506, 1235, 4135, 151, 51, 41138, 40645, 1510, 1158, 312, 38, 52, 1216, 41007, 1130]
#test_holdout_dataset_ids = [312, 316, 1216]

memory_budget = 20.0
privacy = None

results_dict = {}


class ConstraintRun(object):
    def __init__(self, space, best_trial, test_score, more=None):
        self.space = space
        self.best_trial = best_trial
        self.test_score = test_score
        self.more = more

    def print_space(self):
        for pre, _, node in RenderTree(self.space.parameter_tree):
            if node.status == True:
                print("%s%s" % (pre, node.name))

    def get_best_config(self):
        return self.best_trial.params

class ConstraintEvaluation(object):
    def __init__(self, dataset=None, constraint=None, system_def=None):
        self.runs = []
        self.dataset = dataset
        self.constraint = constraint
        self.system_def = system_def

    def append(self, constraint_run: ConstraintRun):
        self.runs.append(constraint_run)

    def get_best_run(self):
        max_score = -np.inf
        max_run = None
        for i in range(len(self.runs)):
            if max_score < self.runs[i].test_score:
                max_score = self.runs[i].test_score
                max_run = self.runs[i]
        return max_run

    def get_best_config(self):
        best_run = self.get_best_run()
        return best_run.get_best_config()

    def get_worst_run(self):
        min_score = np.inf
        min_run = None
        for i in range(len(self.runs)):
            if min_score > self.runs[i].test_score:
                min_score = self.runs[i].test_score
                min_run = self.runs[i]
        return min_run


dynamic_approach_log_eval = []
static_approach_log_eval = []


for test_holdout_dataset_id in test_holdout_dataset_ids:

    X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data(test_holdout_dataset_id, randomstate=42)
    metafeature_values_hold = data2features(X_train_hold, y_train_hold, categorical_indicator_hold)

    #try:
    #AL dataset sampling
    #model_compare = pickle.load(open('/home/felix/phd2/picture_progress/new_compare/my_great_model_compare.p', "rb"))
    #model_success = pickle.load(open('/home/felix/phd2/picture_progress/new_success/my_great_model_success_rate.p', "rb"))
    #except:
        #model = pickle.load(open('/tmp/my_great_model.p', "rb"))
    #model = pickle.load(open('/home/felix/phd2/my_meta_model/my_great_model.p', "rb")

    #uniform dataset sampling
    #model_compare = pickle.load(open('/home/felix/phd2/picture_progress/uniform_sampling/test_model/my_great_model_compare.p', "rb"))
    #model_success = pickle.load(open('/home/felix/phd2/picture_progress/uniform_sampling/test_model/my_great_model_success.p', "rb"))

    model_compare = pickle.load(open('/home/neutatz/phd2/picture_progress/al_only/my_great_model_compare.p', "rb"))
    #model_success = pickle.load(open('/home/neutatz/phd2/picture_progress/al_only/my_great_model_success.p', "rb"))

    #model_success = pickle.load(open('/tmp/my_great_model_compare_scaled.p', "rb"))
    #model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/machine 4_squared/my_great_model_compare_scaled.p', "rb"))
    #model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/machine 2/my_great_model_compare_scaled.p', "rb"))
    #model_success = pickle.load(open("/home/neutatz/phd2/decAutoML2weeks_compare2default/1 week/machine4 cost_sensitive/my_great_model_compare_scaled.p", "rb"))

    #model_success = pickle.load(open("/home/neutatz/phd2/decAutoML2weeks_compare2default/1 week/more weeks cost sensitive - machine4/my_great_model_compare_scaled.p", "rb"))
    #model_success = pickle.load(open("/home/neutatz/phd2/decAutoML2weeks_compare2default/3 weeks/more_weeks_scaled_compare_5min_machine2/my_great_model_compare_scaled.p", "rb"))

    model_success = pickle.load(open(
        "/home/neutatz/phd2/decAutoML2weeks_compare2default/more_weeks/machine2_5min/my_great_model_compare_scaled.p",
        "rb"))

    #model_success = pickle.load(open("/home/neutatz/phd2/decAutoML2weeks_compare2default/3 weeks/machine2_normal_scaled_compare/my_great_model_compare_scaled.p", "rb"))


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

    for minutes_to_search in range(1, 6):

        current_dynamic = []
        current_static = []

        search_time_frozen = minutes_to_search * 60

        new_constraint_evaluation_dynamic = ConstraintEvaluation(dataset=test_holdout_dataset_id, constraint={'search_time': minutes_to_search}, system_def='dynamic')
        new_constraint_evaluation_default = ConstraintEvaluation(dataset=test_holdout_dataset_id, constraint={'search_time': minutes_to_search}, system_def='default')

        for repeat in range(5):

            space = None
            best_estimate_value = -1
            for m_run in range(5):
                study_prune = optuna.create_study(direction='maximize')
                study_prune.optimize(lambda trial: optimize_accuracy_under_constraints2(trial=trial,
                                                                                       metafeature_values_hold=metafeature_values_hold,
                                                                                       search_time=search_time_frozen,
                                                                                       model_compare=model_compare,
                                                                                       model_success=model_success,
                                                                                       memory_limit=memory_budget,
                                                                                       privacy_limit=privacy,
                                                                                       #evaluation_time=int(0.1*search_time_frozen),
                                                                                       #hold_out_fraction=0.33,
                                                                                       tune_space=True,
                                                                                       ), n_trials=500, n_jobs=1)

                if best_estimate_value < study_prune.best_trial.value:
                    space = study_prune.best_trial.user_attrs['space']
                    best_estimate_value = study_prune.best_trial.value



            for pre, _, node in RenderTree(space.parameter_tree):
                if node.status == True:
                    print("%s%s" % (pre, node.name))

            search_dynamic = None
            result = 0
            try:
                result, search_dynamic = utils_run_AutoML(study_prune.best_trial,
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
                new_constraint_evaluation_dynamic.append(ConstraintRun(space, search_dynamic.study.best_trial, result, more=study_prune.best_trial))
            except:
                result = 0

                new_constraint_evaluation_dynamic.append(ConstraintRun(space, 'shit happend', result, more=study_prune.best_trial))
            from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import show_progress
            #show_progress(search, X_test_hold, y_test_hold, my_scorer)

            print("test result: " + str(result))
            current_dynamic.append(result)

            print('dynamic: ' + str(current_dynamic))
            print('static: ' + str(current_static))


            gen_new = SpaceGenerator()
            space_new = gen_new.generate_params()
            for pre, _, node in RenderTree(space_new.parameter_tree):
                if node.status == True:
                    print("%s%s" % (pre, node.name))

            test_score = 0.0
            search_default = None
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

        print('dynamic: ' + str(dynamic_approach))
        print('static: ' + str(static_approach))

        results_dict[test_holdout_dataset_id] = {}
        results_dict[test_holdout_dataset_id]['dynamic'] = dynamic_approach
        results_dict[test_holdout_dataset_id]['static'] = static_approach

        pickle.dump(results_dict, open('/home/neutatz/phd2/picture_progress/all_test_datasets/all_results_cost_sens_mul.p', 'wb+'))

        dynamic_approach_log_eval.append(copy.deepcopy(new_constraint_evaluation_dynamic))
        static_approach_log_eval.append(copy.deepcopy(new_constraint_evaluation_default))

        results_dict_log = {}
        results_dict_log['dynamic'] = dynamic_approach_log_eval
        results_dict_log['static'] = static_approach_log_eval

        pickle.dump(results_dict_log, open('/home/neutatz/phd2/picture_progress/all_test_datasets/eval_dict_log_mul.p', 'wb+'))
