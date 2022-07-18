from sklearn.metrics import make_scorer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd

def true_positive_rate_score(y_true, y_pred, sensitive_data, labels =[False, True]):
	ids = [[],[]]

	flat_y_true = y_true.values.flatten()

	true_positive_rate = np.zeros(2)

	sensitive_values = list(np.unique(sensitive_data))

	#print(sensitive_values)

	if len(sensitive_values) != 2:
		return 0.0

	for i in range(2):
		ids[i] = np.where(sensitive_data[y_true.index.values] == sensitive_values[i])[0]
		#print(ids[i])

		tp = -1
		if len(ids[i]) == 0:
			true_positive_rate[i] = 0.0
			continue
		elif len(np.unique(flat_y_true[ids[i]])) == 1:
			if np.unique(flat_y_true[ids[i]])[0] == labels[0]:
				tp = 0
			else:
				tp = len(ids[i])
		else:
			_, _, _, tp = confusion_matrix(flat_y_true[ids[i]], y_pred[ids[i]], labels=labels).ravel()

		true_positive_rate[i] = tp / float(len(ids[i]))

	return np.abs(true_positive_rate[0] - true_positive_rate[1])