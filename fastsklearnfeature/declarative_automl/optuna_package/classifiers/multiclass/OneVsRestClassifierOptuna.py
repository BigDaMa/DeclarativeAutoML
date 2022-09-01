from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import warnings
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import is_regressor
from sklearn.multiclass import _ConstantPredictor
import copy

def _fit_binary(estimator, X, y, classes=None):
    """Fit a single binary estimator."""
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn("Label %s is present in all training examples." %
                          str(classes[c]))
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:
        estimator.fit(X, y)
    return estimator

def _fit_binary_iterative(estimator, X, y, classes=None, sample_weight=None, number_steps=2):
    """Fit a single binary estimator."""
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn("Label %s is present in all training examples." %
                          str(classes[c]))
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:
        estimator.custom_iterative_fit(X, y, sample_weight=sample_weight, number_steps=number_steps)
    return estimator

def _predict_binary(estimator, X):
    """Make predictions using a single binary estimator."""
    if is_regressor(estimator):
        return estimator.predict(X)

    invert_op = getattr(estimator, "decision_function", None)
    score = None
    if type(invert_op) != type(None):
        score = np.ravel(estimator.decision_function(X))
    else:
        score = estimator.predict_proba(X)[:, 1]
    return score

class OneVsRestClassifierOptuna(OneVsRestClassifier):

    def __init__(self, estimator, *, n_jobs=None):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.estimators_ = []



    def has_iterative_fit(self):
        invert_op = getattr(self.estimator, "custom_iterative_fit", None)
        return type(invert_op) != type(None)

    def get_max_steps(self):
        return self.estimator.get_max_steps()

    #todo: think about sample weight
    def fit(self, X, y):
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = list(col.toarray().ravel() for col in Y.T)
        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.

        if len(self.estimators_) == 0:
            self.estimators_ = [None] * len(columns)
        for i, column in enumerate(columns):
            if type(self.estimators_[i]) == type(None):
                self.estimators_[i] = copy.deepcopy(self.estimator)
            self.estimators_[i] = _fit_binary(self.estimators_[i], X, column,
                                                        classes=["not %s" % self.label_binarizer_.classes_[i],
                                                                 self.label_binarizer_.classes_[i]])

        return self


    def custom_iterative_fit(self, X, y=None, sample_weight=None, number_steps=2):
        """Fit underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like of shape (n_samples, n_features)
            Data.

        y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        Returns
        -------
        self
        """
        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = list(col.toarray().ravel() for col in Y.T)
        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.


        if len(self.estimators_) == 0:
            self.estimators_ = [None] * len(columns)
        for i, column in enumerate(columns):
            if type(self.estimators_[i]) == type(None):
                self.estimators_[i] = copy.deepcopy(self.estimator)
            self.estimators_[i] = _fit_binary_iterative(self.estimators_[i], X, column,
                classes=["not %s" % self.label_binarizer_.classes_[i], self.label_binarizer_.classes_[i]],
                sample_weight=None, number_steps=2)

        return self

