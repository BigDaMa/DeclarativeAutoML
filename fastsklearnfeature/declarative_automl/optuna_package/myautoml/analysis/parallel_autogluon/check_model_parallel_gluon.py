from sklearn.metrics import make_scorer
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.util_classes import ConstraintEvaluation, ConstraintRun
import argparse
import openml
from sklearn.metrics import balanced_accuracy_score
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import time
import numpy as np
import getpass
import os
import shutil
from codecarbon import EmissionsTracker




openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'

my_scorer = make_scorer(balanced_accuracy_score)

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="OpenML datatset ID", type=int)
parser.add_argument("--outputname", "-o", help="Name of the output file")
args = parser.parse_args()
print(args.dataset)


print(args)

#args.dataset = 168794

memory_budget = 500.0
privacy = None

for test_holdout_dataset_id in [args.dataset]:

    X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data('data', randomstate=42, task_id=test_holdout_dataset_id)

    dynamic_approach = []

    new_constraint_evaluation_dynamic_all = []

    #for minutes_to_search in [10, 30, 60, 5*60]:
    for minutes_to_search in [5 * 60]:

        current_dynamic = []
        search_time_frozen = minutes_to_search #* 60
        new_constraint_evaluation_dynamic = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                 constraint={'search_time': minutes_to_search},
                                                                 system_def='dynamic')

        for repeat in range(10):

            tmp_path = "/home/" + getpass.getuser() + "/data/auto_tmp/autosklearn" + str(time.time()) + '_' + str(np.random.randint(1000)) + 'folder'

            try:
                tracker = EmissionsTracker()
                tracker.start()
                df = pd.DataFrame(data=X_train_hold)
                label = 'my_target_label1234'
                df[label] = y_train_hold
                my_data_train = TabularDataset(data=df)

                df_test = pd.DataFrame(data=X_test_hold)
                #df_test[label] = y_test_hold
                my_data_test = TabularDataset(data=df_test)


                predictor = TabularPredictor(label=label, eval_metric='balanced_accuracy', path=tmp_path).fit(train_data=my_data_train, time_limit=search_time_frozen, presets='best_quality')
                tracker.stop()
                y_hat = predictor.predict(my_data_test)
                print("Predictions:  \n", y_hat)

                result = balanced_accuracy_score(y_test_hold, y_hat)

                new_constraint_evaluation_dynamic.append(ConstraintRun('test', 'test', result, more='test', tracker=tracker))
            except:
                result = 0
                new_constraint_evaluation_dynamic.append(ConstraintRun('test', 'shit happened', result, more='test'))
            finally:
                if os.path.exists(tmp_path) and os.path.isdir(tmp_path):
                    shutil.rmtree(tmp_path)

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