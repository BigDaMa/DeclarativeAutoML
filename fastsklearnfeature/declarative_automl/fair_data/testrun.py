from sklearn.metrics import balanced_accuracy_score
import sklearn.metrics
from sklearn.metrics import make_scorer
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
import time
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.MyAutoMLProcessClassBalanceDict import MyAutoML
from fastsklearnfeature.declarative_automl.fair_data.test import get_X_y_id

map_dataset = {}
map_dataset['31'] = 'foreign_worker@{yes,no}'
map_dataset['802'] = 'sex@{female,male}'
map_dataset['1590'] = 'sex@{Female,Male}'
map_dataset['1461'] = 'AGE@{True,False}'
map_dataset['42193'] = 'race_Caucasian@{0,1}'
map_dataset['1480'] = 'V2@{Female,Male}'
# map_dataset['804'] = 'Gender@{0,1}'
map_dataset['42178'] = 'gender@STRING'
map_dataset['981'] = 'Gender@{Female,Male}'
map_dataset['40536'] = 'samerace@{0,1}'
map_dataset['40945'] = 'sex@{female,male}'
map_dataset['451'] = 'Sex@{female,male}'
# map_dataset['945'] = 'sex@{female,male}'
map_dataset['446'] = 'sex@{Female,Male}'
map_dataset['1017'] = 'sex@{0,1}'
map_dataset['957'] = 'Sex@{0,1}'
map_dataset['41430'] = 'SEX@{True,False}'
map_dataset['1240'] = 'sex@{Female,Male}'
map_dataset['1018'] = 'sex@{Female,Male}'
# map_dataset['55'] = 'SEX@{male,female}'
map_dataset['38'] = 'sex@{F,M}'
map_dataset['1003'] = 'sex@{male,female}'
map_dataset['934'] = 'race@{black,white}'

auc = make_scorer(balanced_accuracy_score)

for key in list(map_dataset.keys()):
    print('running key: ' + str(key))
    try:

        X, y, sensitive_attribute_id, categorical_indicator = get_X_y_id(key=key)
        print(y)

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, stratify=y,
                                                                                    train_size=0.6)

        gen = SpaceGenerator()
        space = gen.generate_params()

        # print(space.generate_additional_features_v1(start=True))
        print(len(space.name2node))
        # print(space)
        new_list = list()
        space.generate_additional_features_v2(start=True, sum_list=new_list)
        print(new_list)
        print(len(new_list))

        new_list = list()
        space.generate_additional_features_v2_name(start=True, sum_list=new_list)
        print(new_list)
        print(len(new_list))

        from anytree import RenderTree

        for pre, _, node in RenderTree(space.parameter_tree):
            print("%s%s: %s" % (pre, node.name, node.status))

        search = MyAutoML(n_jobs=1,
                          time_search_budget=2 * 60,
                          space=space,
                          main_memory_budget_gb=40,
                          hold_out_fraction=0.3,
                          fairness_limit=0.90,
                          fairness_group_id=sensitive_attribute_id)

        begin = time.time()

        best_result = search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=auc)

        from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import show_progress

        #show_progress(search, X_test, y_test, auc)

        # importances = optuna.importance.get_param_importances(search.study)
        # print(importances)

        test_score = auc(search.get_best_pipeline(), X_test, y_test)

        print("result: " + str(best_result) + " test: " + str(test_score))
    except Exception as e:
        print('exception: ' + str(e))
