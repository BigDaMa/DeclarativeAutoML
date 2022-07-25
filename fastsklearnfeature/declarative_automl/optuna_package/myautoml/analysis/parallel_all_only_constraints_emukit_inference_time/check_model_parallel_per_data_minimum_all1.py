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



print(args)

memory_budget = 500.0
privacy = None

for test_holdout_dataset_id in [args.dataset]:

    X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data('data', randomstate=42, task_id=test_holdout_dataset_id)
    metafeature_values_hold = data2features(X_train_hold, y_train_hold, categorical_indicator_hold)

    #model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/training_sampling_min_2Drandom_machine2/my_great_model_compare_scaled.p', "rb"))
    #model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/july30_machine1/my_great_model_compare_scaled.p', "rb"))
    #model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/july30_machine4/my_great_model_compare_scaled.p', "rb"))
    model_success = pickle.load(open('/home/neutatz/data/my_temp/my_great_model_compare_scaled.p', "rb"))
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

    #for minutes_to_search in [1, 5, 10, 60]:#range(1, 6):
    for inference_time in [0.0007504010200500488, 0.0007910609245300294,0.0008180351257324219,0.0010206394195556641,0.0014590854644775393, 0.00301915740966]:

        minutes_to_search = 5
        search_time_frozen = minutes_to_search * 60
        pipeline_size = None
        training_time = None


        current_static = []
        new_constraint_evaluation_default = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                 constraint={'inference_time': inference_time},
                                                                 system_def='default')


        for repeat in range(10):
            gen_new = EmuSpaceGenerator()
            space = gen_new.generate_params()

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
                                          inference_time_limit=inference_time
                                          )

                best_result = search_default.fit(X_train_hold, y_train_hold,
                                                 categorical_indicator=categorical_indicator_hold, scorer=my_scorer)
                result = my_scorer(search_default.get_best_pipeline(), X_test_hold, y_test_hold)

                new_constraint_evaluation_default.append(
                    ConstraintRun(space_str=space2str(space.parameter_tree), params='default',
                                  test_score=result, estimated_score=0.0))
            except:
                result = 0
                new_constraint_evaluation_default.append(
                    ConstraintRun(space_str=space2str(space.parameter_tree), params='default',
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