from sklearn.base import ClassifierMixin
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

def calculate_class_weight(y, custom_weight):
    class_weight = {}
    unique_counts = np.unique(y, return_counts=True)
    sorted_ids = np.argsort(unique_counts[1])
    class_weight[unique_counts[0][sorted_ids[0]]] = custom_weight / unique_counts[1][sorted_ids[0]]
    class_weight[unique_counts[0][sorted_ids[1]]] = (1.0 - custom_weight) / unique_counts[1][sorted_ids[1]]
    return class_weight

