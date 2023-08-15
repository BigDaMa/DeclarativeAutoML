from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import plot_most_important_features
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
import openml
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import numpy as np
import optuna
from sklearn.model_selection import LeaveOneGroupOut
import getpass
import heapq
from functools import partial
import copy
from sklearn.model_selection import TimeSeriesSplit

openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/' + getpass.getuser() + '/phd2/cache_openml'


#model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/sep20_constraint_model/my_great_model_compare_scaled.p', "rb"))

#model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/okt1_2week_constraint_model/my_great_model_compare_scaled.p', "rb"))


#X = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/okt1_2week_constraint_model/felix_X_compare_scaled.p', "rb"))
#y = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/okt1_2week_constraint_model/felix_y_compare_scaled.p', "rb"))
#groups = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/okt1_2week_constraint_model/felix_group_compare_scaled.p', "rb"))

days_f1score_all = []

for rrepeat in range(5):

    days_f1score = []

    #for discrete in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for discrete in [0.1]:

        for day in [1,2,3,4,5,6,7]:


            X = pickle.load(open('/home/' + getpass.getuser() + '/data/my_temp/felix_X_compare_scaled.p', "rb"))
            old_len = len(X)

            y = pickle.load(open('/home/' + getpass.getuser() + '/data/my_temp/felix_y_compare_scaled.p', "rb"))
            groups = pickle.load(open('/home/' + getpass.getuser() + '/data/my_temp/felix_group_compare_scaled.p', "rb"))

            X = X[0: int((day / 7.0) * old_len), :]
            y = y[0: int((day / 7.0) * old_len)]
            groups = groups[0: int((day / 7.0) * old_len)]

            X_test_all = pickle.load(open('/home/' + getpass.getuser() + '/data/constraints/felix_X_compare_scaled.p', "rb"))
            y_test_all = pickle.load(open('/home/' + getpass.getuser() + '/data/constraints/felix_y_compare_scaled.p', "rb"))

            print(np.unique(groups))


            new_groups = []
            for g in groups:
                if '_' in g:
                    new_groups.append(g.split('_')[-1])
                else:
                    new_groups.append(g)
            groups = new_groups

            print(np.unique(groups))
            print(len(np.unique(groups)))

            #print(np.unique(groups))


            print(X.shape)

            #model_success = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1)
            #model_success.fit(X, y)

            #model_success = xgb.XGBRegressor(tree_method="hist", n_estimators=1000, max_depth=12)
            # Fit the model using predictor X and response y.
            #model_success.fit(X, y)

            best_trials = []
            study = None

            for folds in [10]:

                def objective(trial, folds):

                    group_kfold_group = GroupKFold(n_splits=5)
                    scores_grouped = []
                    for i_group, (train_index_group, test_index_group) in enumerate(group_kfold_group.split(X, y, groups)):

                        try:
                            print('best: ' + str(study.best_params))
                        except:
                            pass

                        max_depth = None
                        if trial.suggest_categorical("use_max_depth", [True, False]):
                            max_depth = trial.suggest_int("max_depth", 1, 150, log=True)

                        model_success = RandomForestClassifier(n_estimators=trial.suggest_int('n_estimators', 1, 1000, log=True),
                                                              bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
                                                              min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                                                              min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                                                              max_features=trial.suggest_float("max_features", 0.0, 1.0),
                                                              max_depth=max_depth,
                                                              random_state=42, n_jobs=-1)

                        group_kfold = TimeSeriesSplit(n_splits=folds)#GroupKFold(n_splits=folds)
                        if folds == 224:
                            group_kfold = LeaveOneGroupOut()

                        scores = []
                        errors = []
                        for train_index, test_index in group_kfold.split(X, y, groups):

                            test_index_filtered = []
                            for test_i in range(len(test_index)):
                                if test_index[test_i] in test_index_group:
                                    test_index_filtered.append(test_index[test_i])

                            train_index_filtered = []
                            for train_i in range(len(train_index)):
                                if train_index[train_i] in train_index_group:
                                    train_index_filtered.append(train_index[train_i])

                            print('train index: ' + str(train_index_filtered))
                            print('test index: ' + str(test_index_filtered))

                            X_train, X_test = np.array(X)[train_index_filtered], np.array(X)[test_index_filtered]
                            y_train, y_test = np.array(y)[train_index_filtered], np.array(y)[test_index_filtered]

                            print(y_train >= discrete)
                            print(np.sum(y_train >= discrete))

                            model_success.fit(X_train, y_train >= discrete)
                            y_test_predict = model_success.predict(X_test)
                            print('predict: ' + str(y_test_predict))
                            errors.append(f1_score(y_test >= discrete, y_test_predict))
                            print('fold')

                        print('r2: ' + str(np.mean(scores)))
                        #return np.mean(scores)
                        #return np.mean(errors)
                        scores_grouped.append(np.mean(errors))
                    return np.mean(scores_grouped)

                #study = optuna.create_study(direction='maximize')


                study = optuna.create_study(direction='maximize')


                for t in best_trials:
                    study.enqueue_trial(t.params)

                study.optimize(partial(objective, folds=folds), n_trials=100)

                def get_val(trial):
                    return trial.value

                best_trials = heapq.nlargest(10, study.trials, key=get_val)


            print(study.best_params)

            max_depth = None
            if study.best_params["use_max_depth"]:
                max_depth = study.best_params["max_depth"]

            model_success = RandomForestClassifier(n_estimators=study.best_params['n_estimators'],
                                                      bootstrap=study.best_params['bootstrap'],
                                                      min_samples_split=study.best_params['min_samples_split'],
                                                      min_samples_leaf=study.best_params['min_samples_leaf'],
                                                      max_features=study.best_params['max_features'],
                                                      max_depth=max_depth,
                                                      random_state=42, n_jobs=-1)
            model_success.fit(X, np.array(y) >= discrete)

            y_test_predict_all = model_success.predict(X_test_all)

            days_f1score.append(f1_score(np.array(y_test_all) >= discrete, y_test_predict_all))

            print(days_f1score)

    print(days_f1score)
    days_f1score_all.append(copy.deepcopy(days_f1score))
    print(days_f1score_all)

    my_dict = {}
    my_dict['res'] = days_f1score_all

    with open('/tmp/avg_series_ts.p', "wb+") as pickle_model_file:
        pickle.dump(my_dict, pickle_model_file)

print(days_f1score_all)
