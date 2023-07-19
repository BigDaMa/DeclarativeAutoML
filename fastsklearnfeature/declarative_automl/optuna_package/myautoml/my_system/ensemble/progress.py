from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.ensemble.AutoEnsembleSuccessiveStore import MyAutoML as AutoEnsembleML
import sklearn
import openml
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer

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

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, stratify=y,
                                                                            train_size=0.6)

gen = SpaceGenerator()
space = gen.generate_params()

searchtime = 2*60

search = AutoEnsembleML(n_jobs=1,
                  evaluation_budget=searchtime*0.1,
                  time_search_budget=searchtime,
                  max_ensemble_models=1,
                  use_incremental_data=True,
                  shuffle_validation=True,
                  hold_out_fraction=0.33,
                  space=space
                  )
search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=auc)

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import show_progress
show_progress(search, X_test, y_test, auc)