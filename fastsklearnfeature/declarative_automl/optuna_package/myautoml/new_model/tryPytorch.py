from autoPyTorch import (AutoNetClassification,
                         AutoNetMultilabel,
                         AutoNetRegression,
                         AutoNetImageClassification,
                         AutoNetImageClassificationMultipleDatasets)

# Other imports for later usage
import pandas as pd
import numpy as np
import os as os
import openml
import json
from sklearn.metrics import r2_score
import pickle
from sklearn.model_selection import GroupKFold

autonet = AutoNetRegression(config_preset="tiny_cs", result_logger_dir="logs/")

# Get the current configuration as dict
current_configuration = autonet.get_current_autonet_config()

# Get the ConfigSpace object with all hyperparameters, conditions, default values and default ranges
hyperparameter_search_space = autonet.get_hyperparameter_search_space()

# Print all possible configuration options
#autonet.print_help()

# Get data from the openml task "Supervised Classification on credit-g (https://www.openml.org/t/31)"
'''
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
'''
X = np.array(pickle.load(open('/home/neutatz/phd2/picture_progress/al_only/felix_X_success.p', "rb")))
y = np.array(pickle.load(open('/home/neutatz/phd2/picture_progress/al_only/felix_y_success.p', "rb")))
group = np.array(pickle.load(open('/home/neutatz/phd2/picture_progress/al_only/felix_group_success.p', "rb")))


#import matplotlib.pyplot as plt
#plt.hist(y_meta, bins=100)
#plt.show()

gkf = GroupKFold(n_splits=2)
temp_ids, test_ids = list(gkf.split(X, y, groups=group))[0]
X_temp = X[temp_ids]
y_temp = y[temp_ids]
group_temp = group[temp_ids]
X_test = X[test_ids]
y_test = y[test_ids]

gkf = GroupKFold(n_splits=2)
train_ids, val_ids = list(gkf.split(X_temp, y_temp, groups=group_temp))[0]
X_train = X[train_ids]
y_train = y[train_ids]
X_val = X[val_ids]
y_val = y[val_ids]


autonet = AutoNetRegression(config_preset="tiny_cs", result_logger_dir="logs/")
# Fit (note that the settings are for demonstration, you might need larger budgets)
results_fit = autonet.fit(X_train=X_train,
                          Y_train=y_train,
                          X_valid=X_val,
                          Y_valid=y_val,
                          max_runtime=300,
                          min_budget=60,
                          max_budget=100,
                          refit=True)

# Save fit results as json
with open("logs/results_fit.json", "w") as file:
    json.dump(results_fit, file)

score = autonet.score(X_test=X_test, Y_test=y_test)
print("Accuracy score", score)

y_test_pred = autonet.predict(X=X_test)
print('r2: ' + str(r2_score(y_test, y_test_pred)))

pytorch_model = autonet.get_pytorch_model()
print(pytorch_model)