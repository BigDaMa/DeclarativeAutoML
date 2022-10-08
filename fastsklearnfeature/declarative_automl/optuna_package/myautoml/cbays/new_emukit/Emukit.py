import optuna
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import openml
import numpy as np
from sklearn.compose import ColumnTransformer
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.SimpleImputerOptuna import SimpleImputerOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.QuadraticDiscriminantAnalysisOptuna import QuadraticDiscriminantAnalysisOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.KNeighborsClassifierOptuna import KNeighborsClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.HistGradientBoostingClassifierOptuna import HistGradientBoostingClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.private.PrivateLogisticRegressionOptuna import PrivateLogisticRegressionOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.private.PrivateGaussianNBOptuna import PrivateGaussianNBOptuna
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.cbays.new_emukit.Space_GenerationTreeBalanceConstrained2 import SpaceGenerator
from sklearn.model_selection import StratifiedKFold
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.PassiveAggressiveOptuna import PassiveAggressiveOptuna
import pandas as pd
import time
import resource
import copy
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.automl_parameters as mp_global
import multiprocessing
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.define_space as myspace
import pickle
import os
import glob
from dataclasses import dataclass
import sys
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.fairness.metric import true_positive_rate_score
from emukit.core.loop import UserFunctionResult
from emukit.examples.gp_bayesian_optimization.unknown_constraint_bayesian_optimization import (
    UnknownConstraintGPBayesianOptimization,)
from emukit.core.initial_designs.random_design import RandomDesign
import traceback
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.categorical_encoding.LabelEncoderOptuna import LabelEncoderOptuna
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.categorical_encoding.OneHotEncoderOptuna import OneHotEncoderOptuna

@dataclass
class StopWhenOptimumReachedCallback:
    optimum: float

    def __call__(self, study, trial):
        if study.best_value == self.optimum:
            study.stop()

class TimeException(Exception):
    def __init__(self, message="Time is over!"):
        self.message = message
        super().__init__(self.message)


def constraints_satisfied(p, return_dict, key, training_time, training_time_limit, pipeline_size_limit, inference_time_limit, X, fairness=None, fairness_limit=None):

    return_dict[key + 'result' + '_fairness'] = fairness
    if type(fairness_limit) != type(None) and type(fairness) != type(None):
        if fairness < fairness_limit:
            return_dict[key + 'result'] = -1 * (fairness_limit - fairness)  # return the difference to satisfying the constraint
            return False

    return_dict[key + 'result' + '_training_time'] = training_time
    if type(training_time_limit) != type(None) and training_time > training_time_limit:
        return False

    dumped_obj = pickle.dumps(p)
    pipeline_size = sys.getsizeof(dumped_obj)
    return_dict[key + 'result' + '_pipeline_size'] = pipeline_size
    if type(pipeline_size_limit) != type(None) and pipeline_size > pipeline_size_limit:
        return False

    inference_times = []
    for i in range(10):
        random_id = np.random.randint(low=0, high=X.shape[0])
        start_inference = time.time()
        p.predict(X[[random_id]])
        inference_times.append(time.time() - start_inference)
    return_dict[key + 'result' + '_inference_time'] = np.mean(inference_times)
    if type(inference_time_limit) != type(None) and np.mean(inference_times) > inference_time_limit:
        return False
    return True



def evaluatePipeline(key, return_dict):
    try:
        p = return_dict['p']
        number_of_cvs = return_dict['number_of_cvs']
        cv = return_dict['cv']
        training_sampling_factor = return_dict['training_sampling_factor']
        scorer = return_dict['scorer']
        X = return_dict['X']
        y = return_dict['y']
        main_memory_budget_gb = return_dict['main_memory_budget_gb']
        hold_out_fraction = return_dict['hold_out_fraction']

        training_time_limit = return_dict['training_time_limit']
        inference_time_limit = return_dict['inference_time_limit']
        pipeline_size_limit = return_dict['pipeline_size_limit']
        adversarial_robustness_constraint = return_dict['adversarial_robustness_constraint']
        fairness_limit = return_dict['fairness_limit']
        group_id = return_dict['fairness_group_id']

        size = int(main_memory_budget_gb * 1024.0 * 1024.0 * 1024.0)
        resource.setrlimit(resource.RLIMIT_AS, (size, resource.RLIM_INFINITY))

        training_time = 0
        trained_pipeline = None
        if training_sampling_factor == 1.0:
            start_training = time.time()
            p.fit(X, y)
            training_time = time.time() - start_training

            constraints_satisfied(p, return_dict, key, training_time, training_time_limit, pipeline_size_limit,
                                  inference_time_limit, X, None, fairness_limit)
            trained_pipeline = copy.deepcopy(p)



        scores = []
        fairness_scores = []

        if type(hold_out_fraction) == type(None):
            for cv_num in range(number_of_cvs):
                my_splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(time.time())).split(X, y)
                #my_splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(42)).split(X, y)
                for train_ids, test_ids in my_splits:

                    if training_sampling_factor < 1.0:
                        X_train, _, y_train, _ = sklearn.model_selection.train_test_split(X[train_ids, :], y[train_ids],
                                                                                                    random_state=42,
                                                                                                    stratify=y[train_ids],
                                                                                                    train_size=training_sampling_factor)
                    else:
                        X_train = X[train_ids, :]
                        y_train = y[train_ids]

                    start_training = time.time()
                    p.fit(X_train, y_train)
                    training_time = time.time() - start_training
                    scores.append(scorer(p, X[test_ids, :], pd.DataFrame(y[test_ids])))
                    if type(fairness_limit) != type(None):
                        fairness_scores.append(1 - true_positive_rate_score(pd.DataFrame(y), p.predict(X), sensitive_data=X[:, group_id]))

        else:
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, stratify=y,
                                                                  test_size=hold_out_fraction)

            if training_sampling_factor < 1.0:
                X_train, _, y_train, _ = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                  random_state=42,
                                                                                  stratify=y_train,
                                                                                  train_size=training_sampling_factor)

            start_training = time.time()
            p.fit(X_train, y_train)
            training_time = time.time() - start_training
            scores.append(scorer(p, X_test, pd.DataFrame(y_test)))
            if type(fairness_limit) != type(None):
                fairness_scores.append(1 - true_positive_rate_score(pd.DataFrame(y), p.predict(X), sensitive_data=X[:, group_id]))

        #print(fairness_scores)
        if training_sampling_factor < 1.0:
            trained_pipeline = copy.deepcopy(p)

        constraints_satisfied(p, return_dict, key, training_time, training_time_limit, pipeline_size_limit,
                                  inference_time_limit, X, np.mean(fairness_scores), fairness_limit)


        return_dict[key + 'result'] = np.mean(scores)

        if return_dict['study_best_value'] < return_dict[key + 'result']:
            with open('/tmp/my_pipeline' + str(key) + '.p', "wb") as pickle_pipeline_file:
                pickle.dump(trained_pipeline, pickle_pipeline_file)

    except Exception as e:
        return_dict[key + 'result'] = -1 * np.inf
        print(p)
        print(str(e) + '\n\n')





class MyAutoML:
    def __init__(self, cv=5,
                 number_of_cvs=1,
                 hold_out_fraction=None,
                 evaluation_budget=np.inf,
                 time_search_budget=10*60,
                 n_jobs=1,
                 space=None,
                 study=None,
                 main_memory_budget_gb=4,
                 sample_fraction=1.0,
                 differential_privacy_epsilon=None,
                 training_time_limit=None,
                 inference_time_limit=None,
                 pipeline_size_limit=None,
                 adversarial_robustness_constraint=None,
                 fairness_limit=None,
                 fairness_group_id=None
                 ):
        self.cv = cv
        self.time_search_budget = time_search_budget
        self.n_jobs = n_jobs
        self.evaluation_budget = evaluation_budget
        self.number_of_cvs = number_of_cvs
        self.hold_out_fraction = hold_out_fraction

        self.classifier_list = myspace.classifier_list
        self.private_classifier_list = myspace.private_classifier_list
        self.preprocessor_list = myspace.preprocessor_list
        self.scaling_list = myspace.scaling_list
        self.categorical_encoding_list = myspace.categorical_encoding_list
        self.augmentation_list = myspace.augmentation_list

        self.fairness_limit = fairness_limit
        self.fairness_group_id = fairness_group_id

        #generate binary or mapping for each hyperparameter


        self.space = space
        self.study = study
        self.main_memory_budget_gb = main_memory_budget_gb
        self.sample_fraction = sample_fraction
        self.differential_privacy_epsilon = differential_privacy_epsilon

        self.training_time_limit = training_time_limit
        self.inference_time_limit = inference_time_limit
        self.pipeline_size_limit = pipeline_size_limit
        self.adversarial_robustness_constraint = adversarial_robustness_constraint

        self.random_key = str(time.time()) + '-' + str(np.random.randint(0, 1000))

        self.best_value = -np.inf
        self.best_pipeline = None


    def get_best_pipeline(self):
        try:
            return self.best_pipeline
        except:
            return None


    def predict(self, X):
        best_pipeline = self.get_best_pipeline()
        return best_pipeline.predict(X)


    def fit(self, X_new, y_new, sample_weight=None, categorical_indicator=None, scorer=None):
        self.start_fitting = time.time()

        my_factor = -1

        if self.sample_fraction < 1.0:
            X, _, y, _ = sklearn.model_selection.train_test_split(X_new, y_new, random_state=42, stratify=y_new, train_size=self.sample_fraction)
        else:
            X = X_new
            y = y_new

        def objective1(parameterization):
            start_total = time.time()

            self.space.parameterization = parameterization

            try:

                imputer = SimpleImputerOptuna()
                imputer.init_hyperparameters(self.space, X, y)

                scaler = self.space.suggest_categorical('scaler', self.scaling_list)
                scaler.init_hyperparameters(self.space, X, y)

                onehot_transformer = self.space.suggest_categorical('categorical_encoding', self.categorical_encoding_list)
                onehot_transformer.init_hyperparameters(self.space, X, y)

                preprocessor = self.space.suggest_categorical('preprocessor', self.preprocessor_list)
                preprocessor.init_hyperparameters(self.space, X, y)


                if type(self.differential_privacy_epsilon) == type(None):
                    classifier = self.space.suggest_categorical('classifier', self.classifier_list)
                else:
                    classifier = self.space.suggest_categorical('private_classifier', self.private_classifier_list)

                classifier.init_hyperparameters(self.space, X, y)

                class_weighting = False
                custom_weighting = False
                custom_weight = 'balanced'

                if isinstance(classifier, KNeighborsClassifierOptuna) or \
                        isinstance(classifier, QuadraticDiscriminantAnalysisOptuna) or \
                        isinstance(classifier, HistGradientBoostingClassifierOptuna) or \
                        isinstance(classifier, PrivateLogisticRegressionOptuna) or \
                        isinstance(classifier, PrivateGaussianNBOptuna):
                    pass
                else:
                    class_weighting = self.space.suggest_categorical('class_weighting', [True, False])
                    if class_weighting:
                        custom_weighting = self.space.suggest_categorical('custom_weighting', [True, False])
                        if custom_weighting:
                            unique_counts = np.unique(y)
                            custom_weight = {}
                            for unique_i in range(len(unique_counts)):
                                custom_weight[unique_counts[unique_i]] = self.space.suggest_uniform(
                                    'custom_class_weight' + str(unique_i), 0.0, 1.0, check=False)

                if class_weighting:
                    classifier.set_weight(custom_weight)


                use_training_sampling = self.space.suggest_categorical('use_training_sampling', [True, False])
                training_sampling_factor = 1.0
                if use_training_sampling:
                    training_sampling_factor = self.space.suggest_uniform('training_sampling_factor', 0.0, 1.0)


                augmentation = self.space.suggest_categorical('augmentation', self.augmentation_list)
                augmentation.init_hyperparameters(self.space, X, y)

                numeric_transformer = Pipeline([('imputation', imputer), ('scaler', scaler)])

                if isinstance(onehot_transformer, OneHotEncoderOptuna):
                    categorical_transformer = Pipeline([('removeNAN', LabelEncoderOptuna()), ('onehot_transform', onehot_transformer)])
                else:
                    categorical_transformer = Pipeline([('onehot_transform', onehot_transformer)])


                my_transformers = []
                if np.sum(np.invert(categorical_indicator)) > 0:
                    my_transformers.append(('num', numeric_transformer, np.invert(categorical_indicator)))

                if np.sum(categorical_indicator) > 0:
                    my_transformers.append(('cat', categorical_transformer, categorical_indicator))


                data_preprocessor = ColumnTransformer(transformers=my_transformers)

                my_pipeline = Pipeline([('data_preprocessing', data_preprocessor), ('preprocessing', preprocessor), ('augmentation', augmentation),
                              ('classifier', classifier)])

                key = 'My_automl' + self.random_key + 'My_process' + str(time.time()) + "##" + str(np.random.randint(0,1000))

                manager = multiprocessing.Manager()
                return_dict = manager.dict()

                return_dict['p'] = copy.deepcopy(my_pipeline)
                return_dict['number_of_cvs'] = self.number_of_cvs
                return_dict['cv'] = self.cv
                return_dict['training_sampling_factor'] = training_sampling_factor
                return_dict['scorer'] = scorer
                return_dict['X'] = X
                return_dict['y'] = y
                return_dict['main_memory_budget_gb'] = self.main_memory_budget_gb
                return_dict['hold_out_fraction'] = self.hold_out_fraction
                return_dict['training_time_limit'] = self.training_time_limit
                return_dict['inference_time_limit'] = self.inference_time_limit
                return_dict['pipeline_size_limit'] = self.pipeline_size_limit
                return_dict['adversarial_robustness_constraint'] = self.adversarial_robustness_constraint
                return_dict['fairness_limit'] = self.fairness_limit
                return_dict['fairness_group_id'] = self.fairness_group_id

                try:
                    return_dict['study_best_value'] = self.best_value
                except ValueError:
                    return_dict['study_best_value'] = -np.inf

                already_used_time = time.time() - self.start_fitting

                if already_used_time + 2 >= self.time_search_budget:  # already over budget
                    time.sleep(2)
                    return 0.0, np.inf * my_factor

                remaining_time = np.min([self.evaluation_budget, self.time_search_budget - already_used_time])


                my_process = multiprocessing.Process(target=evaluatePipeline, name='start'+key, args=(key, return_dict,))
                my_process.start()
                my_process.join(int(remaining_time))

                # If thread is active
                while my_process.is_alive():
                    # Terminate foo
                    my_process.terminate()
                    my_process.join()

                #del mp_global.mp_store[key]

                result = -1 * np.inf
                if key + 'result' in return_dict:
                    result = return_dict[key + 'result']

                pipeline_size = np.inf
                if key + 'result' + '_pipeline_size' in return_dict:
                    pipeline_size = return_dict[key + 'result' + '_pipeline_size']

                inference_time = np.inf
                if key + 'result' + '_inference_time' in return_dict:
                    inference_time = return_dict[key + 'result' + '_inference_time']

                training_time = np.inf
                if key + 'result' + '_training_time' in return_dict:
                    training_time = return_dict[key + 'result' + '_training_time']

                fairness = -np.inf
                if key + 'result' + '_fairness' in return_dict:
                    fairness = return_dict[key + 'result' + '_fairness']

                if result > 0:
                    pickle_file_name = '/tmp/my_pipeline' + str(key) + '.p'
                    if os.path.exists(pickle_file_name):
                        if self.best_value < result:

                            check = True
                            if type(self.pipeline_size_limit) != type(None):
                                check = pipeline_size <= self.pipeline_size_limit

                            if type(self.inference_time_limit) != type(None):
                                check = inference_time <= self.inference_time_limit

                            if type(self.training_time_limit) != type(None):
                                check = training_time <= self.training_time_limit

                            if type(self.fairness_limit) != type(None):
                                check = fairness >= self.fairness_limit

                            if check:
                                self.best_value = result
                                with open(pickle_file_name, "rb") as pickle_pipeline_file:
                                    self.best_pipeline = pickle.load(pickle_pipeline_file)
                        os.remove(pickle_file_name)

                if type(self.pipeline_size_limit) != type(None):
                    return result, (pipeline_size - self.pipeline_size_limit) * my_factor

                if type(self.inference_time_limit) != type(None):
                    return result, (inference_time - self.inference_time_limit) * my_factor

                if type(self.training_time_limit) != type(None):
                    return result, (training_time - self.training_time_limit) * my_factor

                if type(self.fairness_limit) != type(None):
                    return result, (self.fairness_limit - fairness) * my_factor

                return result, (- 1) * my_factor


            except Exception as e:
                traceback.print_exc()
                print('Exception: ' + str(e) + '\n' + traceback.format_exc() + '\n\n')
                return 0.0, np.inf * my_factor

        ####
        def objective_multi(parameterization):
            results = np.zeros((len(parameterization), 1))
            constraints = np.zeros((len(parameterization), 1))
            for p in range(parameterization.shape[0]):
                results[p], constraints[p] = objective1(parameterization[p])
            return results * -1, constraints

        design = RandomDesign(self.space.get_space())
        num_data_points = 5
        X_init = design.get_samples(num_data_points)
        y_init, y_constraint_init = objective_multi(X_init)

        bo = UnknownConstraintGPBayesianOptimization(variables_list=self.space.get_space().parameters, X=X_init, Y=y_init, Yc=y_constraint_init, batch_size=1)
        results = []
        while time.time() - self.start_fitting < self.time_search_budget:
            X_new = bo.get_next_points(results)
            Y_new, Yc_new = objective1(X_new[0])
            Y_new *= -1
            results = [UserFunctionResult(X_new[0], np.array([Y_new]), Y_constraint=np.array([Yc_new]))]

        #clean up all remaining pipelines
        for my_path in glob.glob('/tmp/my_pipeline' + 'My_automl' + self.random_key + 'My_process' + '*.p'):
            os.remove(my_path)

        return self.best_value




if __name__ == "__main__":
    # auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
    from sklearn.metrics import balanced_accuracy_score

    auc = make_scorer(balanced_accuracy_score)

    # dataset = openml.datasets.get_dataset(1114)

    # dataset = openml.datasets.get_dataset(1116)
    dataset = openml.datasets.get_dataset(31)  # 51

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )

    print(X[:, 12])

    X[:,12] = X[:,12] > np.mean(X[:,12])

    print(X[:,12])

    print(X.shape)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, stratify=y,
                                                                                train_size=0.6)


    numberss = []
    for ii in range(10):
        gen = SpaceGenerator()
        space = gen.generate_params(y_train)

        '''
        search = MyAutoML(n_jobs=1,
                          time_search_budget=1 * 60,
                          space=space,
                          main_memory_budget_gb=40,
                          hold_out_fraction=0.33,
                          fairness_limit=0.95,
                          fairness_group_id=12)
        '''
        search = MyAutoML(n_jobs=1,
                          time_search_budget=2 * 60,
                          space=space,
                          main_memory_budget_gb=40,
                          hold_out_fraction=0.33,
                          pipeline_size_limit=8359)

        begin = time.time()

        best_result = search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=auc)

        try:
            test_score = auc(search.get_best_pipeline(), X_test, y_test)
            print("result: " + str(best_result) + " test: " + str(test_score))
        except:
            pass


        print(time.time() - begin)

        import matplotlib.pyplot as plt
        #plt.plot(list(range(len(search.log))), search.log)
        #plt.show()

        print(np.sum(np.array(search.log) <= 8359))
        numberss.append(np.sum(np.array(search.log) <= 8359))

    print('final: ' + str(numberss))