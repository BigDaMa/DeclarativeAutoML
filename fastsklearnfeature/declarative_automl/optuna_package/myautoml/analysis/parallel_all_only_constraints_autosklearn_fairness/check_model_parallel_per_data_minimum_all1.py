from autosklearn.metrics import make_scorer
import pickle
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
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime
from autosklearn.util.metric import Fair

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

folktable_dict = [ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime]

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

    Fair.fairness_column = sensitive_attribute_id

    print('label: ' + str(y))

    X_train_hold, X_test_hold, y_train_hold, y_test_hold = sklearn.model_selection.train_test_split(X, y,
                                                                                                    random_state=42,
                                                                                                    stratify=y,
                                                                                                    train_size=0.6)

    dynamic_approach = []

    new_constraint_evaluation_dynamic_all = []

    for fairness in [1.0, 0.999379364633232, 0.9940665258902305, 0.9807694321364909, 0.9494830824433045]:
        minutes_to_search = 5
        training_time = None

        current_dynamic = []
        search_time_frozen = minutes_to_search * 60
        new_constraint_evaluation_dynamic = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                 constraint={'fairness': fairness},
                                                                 system_def='dynamic')

        for repeat in range(10):

            try:
                scorerr = make_scorer('balanced_accuracy_constraints', partial(utils.balanced_accuracy_score_constraints, fairness_constraint=fairness), worst_possible_result=-np.inf)


                feat_type = []
                for c_i in range(len(categorical_indicator_hold)):
                    if categorical_indicator_hold[c_i]:
                        feat_type.append('Categorical')
                    else:
                        feat_type.append('Numerical')

                tmp_path = "/home/neutatz/data/auto_tmp/autosklearn" + str(time.time()) + '_' + str(np.random.randint(100))
                automl = AutoSklearn2Classifier(time_left_for_this_task=search_time_frozen, metric=scorerr, seed=repeat, memory_limit=1024*250, tmp_folder=tmp_path, delete_tmp_folder_after_terminate=True)
                automl.fit(X_train_hold, y_train_hold, feat_type=feat_type)

                selected_models, selected_weights, joined_pipeline_size, joined_inference_time, joined_training_time, joined_fairness = utils.return_model_size(
                    automl, X_train_hold, y_train_hold, fairness_constraint=fairness)

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