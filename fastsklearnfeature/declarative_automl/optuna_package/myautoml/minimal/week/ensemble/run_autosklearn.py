import argparse
import getpass
import numpy as np
import time
from sklearn.metrics import balanced_accuracy_score
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
import autosklearn
import os
import copy
import shutil
import pickle
import traceback

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("tmp_path_pickle", type=str)
    args = parser.parse_args()

    tmp_path_pickle = args.tmp_path_pickle

    repetitions_count = 10

    categorical_indicator = None
    search_time = None
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    with open(tmp_path_pickle + ".pickle", 'rb') as send_data_file:
        send_data = pickle.load(send_data_file)
        categorical_indicator = send_data['categorical_indicator']
        search_time = send_data['search_time']
        X_train = send_data['X_train']
        y_train = send_data['y_train']
        X_test = send_data['X_test']
        y_test = send_data['y_test']

    static_params = []
    for random_i in range(repetitions_count):
        tmp_path = "/home/" + getpass.getuser() + "/data/auto_tmp/autosklearn" + str(time.time()) + '_' + str(
            np.random.randint(1000) + 'folder')
        test_score_default = 0.0
        try:
            scorerr = autosklearn.metrics.make_scorer(
                'balanced_accuracy_score',
                balanced_accuracy_score
            )

            feat_type = []
            for c_i in range(len(categorical_indicator)):
                if categorical_indicator[c_i]:
                    feat_type.append('Categorical')
                else:
                    feat_type.append('Numerical')

            automl = AutoSklearn2Classifier(time_left_for_this_task=search_time, metric=scorerr, seed=random_i,
                                            memory_limit=1024 * 250, tmp_folder=tmp_path,
                                            delete_tmp_folder_after_terminate=True)
            automl.fit(X_train, y_train, feat_type=feat_type)
            y_hat = automl.predict(X_test)
            test_score_default = balanced_accuracy_score(y_test, y_hat)
        except Exception as e:
            print(str(e) + '\n\n')
            traceback.print_exc()
            test_score_default = 0.0
        finally:
            if os.path.exists(tmp_path) and os.path.isdir(tmp_path):
                shutil.rmtree(tmp_path)
        static_params.append(test_score_default)

    static_values = copy.deepcopy(np.array(static_params))

    send_data = {}
    send_data['static_values'] = static_values
    with open(tmp_path_pickle + "_result.pickle", "wb+") as output_file:
        pickle.dump(send_data, output_file)
    print('done')