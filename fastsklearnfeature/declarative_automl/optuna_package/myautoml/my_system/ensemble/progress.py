from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.ensemble.AutoEnsembleSuccessiveStore import MyAutoML as AutoEnsembleML
import sklearn
import openml
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer
import numpy as np
from matplotlib import pyplot as plt
import getpass
import pickle
from sklearn.model_selection import StratifiedShuffleSplit

openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'

auc = make_scorer(balanced_accuracy_score)

# dataset = openml.datasets.get_dataset(1114)

#dataset = openml.datasets.get_dataset(1116)
dataset = openml.datasets.get_dataset(31)  # 51
#dataset = openml.datasets.get_dataset(40685)
#dataset = openml.datasets.get_dataset(1596)
#dataset = openml.datasets.get_dataset(41167)
#dataset = openml.datasets.get_dataset(41147)
#dataset = openml.datasets.get_dataset(1596)

#41161
dataset = openml.datasets.get_dataset(41161)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute
)

#print(X[:, 12])

#X[:,12] = X[:,12] > np.mean(X[:,12])

#print(X[:,12])

print(X.shape)

'''
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, stratify=y,
                                                                            train_size=0.6)
                                                                            
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
for i, (train_index, test_index) in enumerate(sss.split(X, y)):
	break

print(train_index)
print(test_index)
split_dict = {}
split_dict['train_index'] = train_index
split_dict['test_index'] = test_index

with open('split.p', "wb+") as pickle_model_file:
    pickle.dump(split_dict, pickle_model_file)
'''
split_dict = pickle.load(open('split.p', "rb"))
X_train = X[split_dict['train_index']]
X_test = X[split_dict['test_index']]
y_train = y[split_dict['train_index']]
y_test = y[split_dict['test_index']]

gen = SpaceGenerator()
space = gen.generate_params()

searchtime = 10 * 60

times_all = []
best_val_scores_all = []


for repeat in range(10):

    search = AutoEnsembleML(n_jobs=1,
                      evaluation_budget=searchtime * 0.1,
                      time_search_budget=searchtime,
                      max_ensemble_models=1,
                      use_incremental_data=True,
                      shuffle_validation=True,
                      hold_out_fraction=0.33,
                      space=space
                      )
    search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=auc)


    times = [0.0]
    validation_scores = [0.0]
    best_val_scores = [0.0]

    for t in search.study.trials:
        try:
            current_time = t.user_attrs['time_since_start']
            current_val = t.value
            if current_val < 0:
                current_val = 0
            times.append(current_time)
            validation_scores.append(current_val)

            if best_val_scores[-1] < current_val:
                best_val_scores.append(current_val)
            else:
                best_val_scores.append(best_val_scores[-1])

        except:
            pass

    print(times)
    print(validation_scores)
    print(best_val_scores)

    times_all.append(times)
    best_val_scores_all.append(best_val_scores)


t_all = []
for tt in times_all:
    t_all.extend(tt)
t_all = list(set(t_all))
t_all.sort()

def extend_v(t_all, t1, v1):
    v1_new = [0.0]
    current_i = 0
    for i in range(1, len(t_all)):
        while current_i + 1 != len(t1) and t_all[i] >= t1[current_i+1]:
            current_i += 1
        v1_new.append(v1[current_i])
    return v1_new

v_all = []
for i in range(len(times_all)):
    v_all.append(np.array(extend_v(t_all, times_all[i], best_val_scores_all[i])))

v_new = np.vstack(v_all)
print(v_new.shape)

avg_v = np.mean(v_new, axis=0)


print(t_all)
print(avg_v)


plt.plot(t_all, avg_v)
plt.show()

my_dict = {}
my_dict['time'] = t_all
my_dict['avg'] = avg_v

with open('/tmp/avg_series.p', "wb+") as pickle_model_file:
    pickle.dump(my_dict, pickle_model_file)
