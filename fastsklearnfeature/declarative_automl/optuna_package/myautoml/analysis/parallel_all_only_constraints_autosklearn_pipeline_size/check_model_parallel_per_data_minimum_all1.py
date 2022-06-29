from autosklearn.metrics import make_scorer
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.util_classes import ConstraintEvaluation, ConstraintRun
import argparse
import openml
from sklearn.metrics import balanced_accuracy_score
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
import time
import numpy as np
import sklearn
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel_all_only_constraints_autosklearn_pipeline_size import utils
from functools import partial

openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="OpenML datatset ID", type=int)
parser.add_argument("--outputname", "-o", help="Name of the output file")
args = parser.parse_args()
print(args.dataset)

#args.dataset = 168794

print(args)

memory_budget = 500.0
privacy = None

for test_holdout_dataset_id in [args.dataset]:

    X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data('data', randomstate=42, task_id=test_holdout_dataset_id)

    dynamic_approach = []

    new_constraint_evaluation_dynamic_all = []

    for pipeline_size in [3175.0, 4026.22, 6651.52, 8359.48, 16797.0, 32266.96]:
        minutes_to_search = 5

        current_dynamic = []
        search_time_frozen = minutes_to_search * 60
        new_constraint_evaluation_dynamic = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                 constraint={'pipeline_size': pipeline_size},
                                                                 system_def='dynamic')

        for repeat in range(10):

            try:
                scorerr = make_scorer('balanced_accuracy_constraints', partial(utils.balanced_accuracy_score_constraints, pipeline_size_constraint=pipeline_size), worst_possible_result=-np.inf)


                feat_type = []
                for c_i in range(len(categorical_indicator_hold)):
                    if categorical_indicator_hold[c_i]:
                        feat_type.append('Categorical')
                    else:
                        feat_type.append('Numerical')

                tmp_path = "/home/neutatz/data/auto_tmp/autosklearn" + str(time.time()) + '_' + str(np.random.randint(100))
                automl = AutoSklearn2Classifier(time_left_for_this_task=search_time_frozen, metric=scorerr, seed=repeat, memory_limit=1024*250, tmp_folder=tmp_path, delete_tmp_folder_after_terminate=True)
                automl.fit(X_train_hold, y_train_hold, feat_type=feat_type)

                joined_pipeline_size, joined_inference_time, selected_models, selected_weights = utils.return_model_size(
                    automl, X_train_hold, y_train_hold, pipeline_size_constraint=pipeline_size)

                result = 0
                try:
                    y_test_pred = utils.predict_with_models(automl, selected_models, selected_weights, X_test_hold)

                    y_class = np.argmax(y_test_pred, axis=1)
                    result = sklearn.metrics.balanced_accuracy_score(y_test_hold, y_class)
                except:
                    pass

                new_constraint_evaluation_dynamic.append(ConstraintRun('test', 'test', result, more='test'))
            except Exception as e:
                print(e)
                result = 0

                new_constraint_evaluation_dynamic.append(ConstraintRun('test', 'shit happened', result, more='test'))

            current_dynamic.append(result)
        dynamic_approach.append(current_dynamic)
        new_constraint_evaluation_dynamic_all.append(new_constraint_evaluation_dynamic)
        print('dynamic: ' + str(dynamic_approach))

    results_dict_log = {}
    results_dict_log['dynamic'] = new_constraint_evaluation_dynamic_all
    pickle.dump(results_dict_log, open('/home/neutatz/data/automl_runs/log_' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))

    results_dict = {}
    results_dict['dynamic'] = dynamic_approach
    pickle.dump(results_dict, open('/home/neutatz/data/automl_runs/' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))