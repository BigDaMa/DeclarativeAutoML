from sklearn.metrics import balanced_accuracy_score
import sklearn.metrics
from sklearn.metrics import make_scorer
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
import time
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.MyAutoMLProcessClassBalanceDict import MyAutoML
from fastsklearnfeature.declarative_automl.fair_data.test import get_X_y_id
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime
import numpy as np

auc = make_scorer(balanced_accuracy_score)

#data_source = ACSDataSource(survey_year='2018', horizon='5-Year', survey='person')
#acs_data = data_source.get_data(states=None, download=True)
#features, label, group = ACSEmployment.df_to_numpy(acs_data)

def get_race_att(data):
    for att_race in range(len(data.features)):
        if data.features[att_race] == 'RAC1P':
            return att_race



data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
#ca_data = data_source.get_data(states=None, download=True)
ca_data = data_source.get_data(states=['CA'], download=True)

task = ACSEmployment
ca_features, ca_labels, group = task.df_to_numpy(ca_data)
ca_features[:, get_race_att(task)] = ca_features[:, get_race_att(task)] == 1.0
print(np.sum(ca_features[:, get_race_att(task)]))
print(len(ca_features))
print(ca_labels)
#print(get_race_att(ACSIncome))

'''
ca_features, ca_labels, group = ACSIncome.df_to_numpy(ca_data)
print(ACSIncome.features)

ca_features, ca_labels, group = ACSEmployment.df_to_numpy(ca_data)
print(ACSEmployment.features)

ca_features, ca_labels, group = ACSPublicCoverage.df_to_numpy(ca_data)
print(ACSPublicCoverage.features)

ca_features, ca_labels, group = ACSMobility.df_to_numpy(ca_data)
print(ACSMobility.features)

ca_features, ca_labels, group = ACSTravelTime.df_to_numpy(ca_data)
print(ACSTravelTime.features)

print(len(ca_features))

print(ACSEmployment.features)

print(ca_features)
print(group)
'''

for task in [ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime]:
    print('running key: ' + str(task))
    try:

        X, y, group = task.df_to_numpy(ca_data)
        X[:, get_race_att(task)] = X[:, get_race_att(task)] == 1.0
        sensitive_attribute_id = get_race_att(task)
        categorical_indicator = np.zeros(X.shape[1], dtype=bool)

        print('label: ' + str(y))

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
