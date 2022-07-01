import sklearn.datasets
import sklearn.metrics
from sklearn.utils.validation import check_is_fitted
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from autosklearn.constants import REGRESSION_TASKS
import pickle
import sys
from autosklearn.automl import _model_predict
import numpy as np
import logging

import time

def balanced_accuracy_score_constraints(y_true, y_pred, sample_weight=None, model=None, inference_time=None, training_time=None,pipeline_size_constraint=None,inference_time_constraint=None, training_time_constraint=None):
    ba = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)

    sum_of_constraint_distance = 0
    if type(model) != type(None) and type(pipeline_size_constraint) != type(None):
        dumped_obj = pickle.dumps(model)
        pipeline_size = sys.getsizeof(dumped_obj)

        if pipeline_size > pipeline_size_constraint:
            sum_of_constraint_distance += pipeline_size - pipeline_size_constraint

    if type(inference_time_constraint) != type(None) and type(inference_time) != type(None):
        if inference_time > inference_time_constraint:
            sum_of_constraint_distance += inference_time - inference_time_constraint

    if type(training_time_constraint) != type(None) and type(training_time) != type(None):
        if training_time > training_time_constraint:
            sum_of_constraint_distance += training_time - training_time_constraint

    if sum_of_constraint_distance > 0:
        return -1 * sum_of_constraint_distance
    else:
        return ba


def get_selected_model_identifiers_and_weight(automl):
    output = []

    for i, weight in enumerate(automl.automl_.ensemble_.weights_):
        identifier = automl.automl_.ensemble_.identifiers_[i]
        if weight > 0.0:
            output.append((identifier, weight))

    return output

def return_model_size(automl, X_train, y_train, pipeline_size_constraint=None, inference_time_constraint=None, training_time_constraint=None):
    try:
        for i, tmp_model in enumerate(automl.automl_.models_.values()):
            if isinstance(tmp_model, (DummyRegressor, DummyClassifier)):
                check_is_fitted(tmp_model)
            else:
                check_is_fitted(tmp_model.steps[-1][-1])
        models = automl.automl_.models_
    except sklearn.exceptions.NotFittedError:
        # When training a cross validation model, self.cv_models_
        # will contain the Voting classifier/regressor product of cv
        # self.models_ in the case of cv, contains unfitted models
        # Raising above exception is a mechanism to detect which
        # attribute contains the relevant models for prediction
        try:
            check_is_fitted(list(automl.automl_.cv_models_.values())[0])
            models = automl.automl_.cv_models_
        except sklearn.exceptions.NotFittedError:
            raise ValueError('Found no fitted models!')

    id2weight = get_selected_model_identifiers_and_weight(automl)

    id2weight.sort(key=lambda x: x[1]*-1)

    print(id2weight)

    joined_pipeline_size = 0
    joined_inference_time = 0
    joined_training_time = 0
    selected_models = []
    selected_weights = []
    for m_identifier, current_weight in id2weight:
        new_model = models[m_identifier]

        pipeline_size = 0
        inference_time = 0
        training_time = 0

        too_big = False
        if type(pipeline_size_constraint) != type(None):
            dumped_obj = pickle.dumps(new_model)
            pipeline_size = sys.getsizeof(dumped_obj)
            print('pipeline size: ' + str(pipeline_size))
            if joined_pipeline_size + pipeline_size > pipeline_size_constraint:
                too_big = True

        if type(inference_time_constraint) != type(None):
            inference_times = []
            for _ in range(10):
                random_id = np.random.randint(low=0, high=X_train.shape[0])
                if isinstance(new_model, VotingClassifier):
                    new_model.le_ = preprocessing.LabelEncoder().fit(y_train)
                start_inference = time.time()
                new_model.predict_proba(X_train[[random_id]])
                inference_times.append(time.time() - start_inference)
            inference_time = np.mean(inference_times)
            logging.warning('post model_type: ' + str(type(new_model)) + ' inference time: ' + str(inference_time))

            if joined_inference_time + inference_time > inference_time_constraint:
                too_big = True

        if type(training_time_constraint) != type(None):
            for key, value in automl.automl_.runhistory_.data.items():
                if key.config_id == m_identifier[1]:
                    if 'training_time_cv' in value.additional_info:
                        training_time = np.sum(value.additional_info['training_time_cv'])
                    else:
                        training_time = value.additional_info['training_time']

                    if joined_training_time + training_time > training_time_constraint:
                        too_big = True
        if too_big:
            cv_models = []
            fold_id = 0
            for est in new_model.estimators_:

                logging.warning('post model_type: ' + str(type(est)))

                pipeline_size = 0
                inference_time = 0
                training_time = 0

                too_big = False
                if type(pipeline_size_constraint) != type(None):
                    dumped_obj = pickle.dumps(est)
                    pipeline_size = sys.getsizeof(dumped_obj)
                    print('pipeline size: ' + str(pipeline_size))
                    if joined_pipeline_size + pipeline_size > pipeline_size_constraint:
                        too_big = True

                if type(inference_time_constraint) != type(None):
                    inference_times = []
                    for _ in range(10):
                        random_id = np.random.randint(low=0, high=X_train.shape[0])
                        start_inference = time.time()
                        est.predict_proba(X_train[[random_id]])
                        inference_times.append(time.time() - start_inference)
                    inference_time = np.mean(inference_times)
                    logging.warning('post model_type: ' + str(type(est)) + ' inference time: ' + str(inference_time))

                    if joined_inference_time + inference_time > inference_time_constraint:
                        too_big = True

                if type(training_time_constraint) != type(None):
                    for key, value in automl.automl_.runhistory_.data.items():
                        if key.config_id == m_identifier[1]:
                            training_time = value.additional_info['training_time_cv'][fold_id]
                            if joined_training_time + training_time > training_time_constraint:
                                too_big = True

                if not too_big:
                    joined_pipeline_size += pipeline_size
                    joined_inference_time += inference_time
                    joined_training_time += training_time
                    cv_models.append(est)

                fold_id += 1
            if len(cv_models) > 1:
                eclf1 = VotingClassifier(estimators=None, voting='soft')
                eclf1.estimators_ = cv_models
                eclf1.le_ = preprocessing.LabelEncoder().fit(y_train)
                selected_models.append(eclf1)
                selected_weights.append(current_weight)
            elif len(cv_models) == 1:
                selected_models.append(cv_models[0])
                selected_weights.append(current_weight)
            else:
                pass
        else:
            joined_pipeline_size += pipeline_size
            joined_inference_time += inference_time
            joined_training_time += training_time
            selected_models.append(new_model)
            selected_weights.append(current_weight)

    return joined_pipeline_size, joined_inference_time, joined_training_time, selected_models, selected_weights


def predict_ensemble(predictions, weights) -> np.ndarray:

    average = np.zeros_like(predictions[0], dtype=np.float64)
    tmp_predictions = np.empty_like(predictions[0], dtype=np.float64)

    # if predictions.shape[0] == len(self.weights_),
    # predictions include those of zero-weight models.
    if len(predictions) == len(weights):
        for pred, weight in zip(predictions, weights):
            np.multiply(pred, weight, out=tmp_predictions)
            np.add(average, tmp_predictions, out=average)

    # if prediction model.shape[0] == len(non_null_weights),
    # predictions do not include those of zero-weight models.
    elif len(predictions) == np.count_nonzero(weights):
        non_null_weights = [w for w in weights if w > 0]
        for pred, weight in zip(predictions, non_null_weights):
            np.multiply(pred, weight, out=tmp_predictions)
            np.add(average, tmp_predictions, out=average)

    # If none of the above applies, then something must have gone wrong.
    else:
        raise ValueError("The dimensions of ensemble predictions"
                         " and ensemble weights do not match!")
    del tmp_predictions
    return average


def predict_with_models(automl, selected_models, weights, X):

    X = automl.automl_.InputValidator.feature_validator.transform(X)

    all_predictions = []
    print('predictions')
    for m in selected_models:
        all_predictions.append(_model_predict(model=m, X=X, task=automl.automl_._task, batch_size=None))
        print(all_predictions)

    print(all_predictions)

    predictions = predict_ensemble(all_predictions, weights)

    if automl.automl_._task not in REGRESSION_TASKS:
        # Make sure average prediction probabilities
        # are within a valid range
        # Individual models are checked in _model_predict
        predictions = np.clip(predictions, 0.0, 1.0)

    return predictions