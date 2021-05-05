from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

print(np.unique(y, return_counts=True))

p=0.5
class_weight = {}
unique_counts = np.unique(y, return_counts=True)
sorted_ids = np.argsort(unique_counts[1])
class_weight[unique_counts[0][sorted_ids[0]]] = p / unique_counts[1][sorted_ids[0]]
class_weight[unique_counts[0][sorted_ids[1]]] = (1.0 - p) / unique_counts[1][sorted_ids[1]]

print(class_weight)

print(212 / len(y))



model = RandomForestClassifier(random_state=42).fit(X, y, sample_weight=compute_sample_weight(class_weight='balanced', y=y))

print(model.feature_importances_)