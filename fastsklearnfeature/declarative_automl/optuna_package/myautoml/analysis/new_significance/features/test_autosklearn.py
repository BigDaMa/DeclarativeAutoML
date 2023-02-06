from pprint import pprint

import sklearn.datasets
import sklearn.metrics

import autosklearn.classification
import numpy as np

from sklearn.model_selection import GroupKFold

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)

groups = np.random.randint(0,5, size=len(X_train))

'''
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    resampling_strategy=sklearn.model_selection.KFold(n_splits=len(X_train))
)
'''

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    resampling_strategy=sklearn.model_selection.GroupKFold(n_splits=len(np.unique(groups))),
    resampling_strategy_arguments={'groups': groups}
)

automl.fit(X_train, y_train)
automl.refit(X_train, y_train)

y_hat = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))