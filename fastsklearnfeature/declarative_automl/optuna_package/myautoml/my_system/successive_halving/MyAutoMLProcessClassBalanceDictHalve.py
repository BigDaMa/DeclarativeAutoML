import optuna
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import openml
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.CategoricalMissingTransformer import CategoricalMissingTransformer
from sklearn.compose import ColumnTransformer
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.SimpleImputerOptuna import SimpleImputerOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.QuadraticDiscriminantAnalysisOptuna import QuadraticDiscriminantAnalysisOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.KNeighborsClassifierOptuna import KNeighborsClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.HistGradientBoostingClassifierOptuna import HistGradientBoostingClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.private.PrivateLogisticRegressionOptuna import PrivateLogisticRegressionOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.private.PrivateGaussianNBOptuna import PrivateGaussianNBOptuna
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
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


@dataclass
class StopWhenOptimumReachedCallback:
    optimum: float

    def __call__(self, study, trial):
        try:
            if study.best_value == self.optimum:
                study.stop()
        except:
            pass

class TimeException(Exception):
    def __init__(self, message="Time is over!"):
        self.message = message
        super().__init__(self.message)


def constraints_satisfied(p, return_dict, key, training_time, training_time_limit, pipeline_size_limit, inference_time_limit, X):
    return_dict[key + 'result' + '_training_time'] = training_time
    if type(training_time_limit) != type(None) and training_time > training_time_limit:
        return_dict[key + 'result'] = -1 * (
                    training_time - training_time_limit)  # return the difference to satisfying the constraint
        return False

    dumped_obj = pickle.dumps(p)
    pipeline_size = sys.getsizeof(dumped_obj)
    return_dict[key + 'result' + '_pipeline_size'] = pipeline_size
    if type(pipeline_size_limit) != type(None) and pipeline_size > pipeline_size_limit:
        return_dict[key + 'result'] = -1 * (
                    pipeline_size - pipeline_size_limit)  # return the difference to satisfying the constraint
        return False

    inference_times = []
    for i in range(10):
        random_id = np.random.randint(low=0, high=X.shape[0])
        start_inference = time.time()
        p.predict(X[[random_id]])
        inference_times.append(time.time() - start_inference)
    return_dict[key + 'result' + '_inference_time'] = np.mean(inference_times)
    if type(inference_time_limit) != type(None) and np.mean(inference_times) > inference_time_limit:
        return_dict[key + 'result'] = -1 * (np.mean(
            inference_times) - inference_time_limit)  # return the difference to satisfying the constraint
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

        trial = return_dict['trial']


        size = int(main_memory_budget_gb * 1024.0 * 1024.0 * 1024.0)
        resource.setrlimit(resource.RLIMIT_AS, (size, resource.RLIM_INFINITY))

        training_time = 0
        trained_pipeline = None
        '''
        if training_sampling_factor == 1.0:
            start_training = time.time()
            p.fit(X, y)
            training_time = time.time() - start_training

            if not constraints_satisfied(p, return_dict, key, training_time, training_time_limit, pipeline_size_limit, inference_time_limit, X):
                return

            trained_pipeline = copy.deepcopy(p)
        '''


        #sample based on budget:
        budget = len(np.unique(y))*10
        it_is_over = False
        max_score = 0

        intermediate_steps = []

        while True:
            scores = []

            if type(hold_out_fraction) == type(None):
                for cv_num in range(number_of_cvs):
                    my_splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(time.time())).split(X, y)
                    #my_splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(42)).split(X, y)
                    for train_ids, test_ids in my_splits:
                        X_train = X[train_ids, :]
                        y_train = y[train_ids]

                        if training_sampling_factor < 1.0:
                            X_train, _, y_train, _ = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                                        random_state=42,
                                                                                                        stratify=y_train,
                                                                                                        train_size=training_sampling_factor)

                        if budget < len(train_ids):
                            X_train, _, y_train, _ = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                     random_state=42,
                                                                     stratify=y_train,
                                                                     train_size=budget)
                        else:
                            it_is_over = True

                        start_training = time.time()
                        p.fit(X_train, y_train)
                        training_time = time.time() - start_training
                        scores.append(scorer(p, X[test_ids, :], pd.DataFrame(y[test_ids])))

            else:
                X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, stratify=y,
                                                                      test_size=hold_out_fraction)

                if training_sampling_factor < 1.0:
                    X_train, _, y_train, _ = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                      random_state=42,
                                                                                      stratify=y_train,
                                                                                      train_size=training_sampling_factor)

                if budget < len(X_train):
                    X_train, _, y_train, _ = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                      random_state=42,
                                                                                      stratify=y_train,
                                                                                      train_size=budget)
                else:
                    it_is_over = True

                start_training = time.time()
                p.fit(X_train, y_train)
                training_time = time.time() - start_training
                scores.append(scorer(p, X_test, pd.DataFrame(y_test)))

                if not constraints_satisfied(p, return_dict, key, training_time, training_time_limit, pipeline_size_limit, inference_time_limit, X):
                    return None
                else:

                    if return_dict['study_best_value'] < np.mean(scores):
                        return_dict[key + 'trained_pipeline'] = copy.deepcopy(p)

                    return_dict[key + 'intermediate_steps'] = intermediate_steps
                    return_dict[key + 'result'] = np.mean(scores)


            if training_sampling_factor < 1.0:
                if not constraints_satisfied(p, return_dict, key, training_time, training_time_limit, pipeline_size_limit, inference_time_limit, X):
                    return None

                trained_pipeline = copy.deepcopy(p)

            if max_score < np.average(scores):
                max_score = np.average(scores)
            trial.report(max_score, budget)
            intermediate_steps.append((max_score, budget))

            #if len(intermediate_steps) >=2 and intermediate_steps[-1][0] <= intermediate_steps[-2][0]:
            #    return None

            if trial.should_prune():
                print('trial should be pruned')
                return_dict[key + 'pruned'] = True
                return None
            print('budget: ' + str(budget) + ' val score: ' + str(np.average(scores)))
            budget *= 2
            if it_is_over:
                break
            #TODO: return all results and report


        return_dict[key + 'result'] = np.mean(scores)

        return_dict[key + 'intermediate_steps'] = intermediate_steps

        if return_dict['study_best_value'] < return_dict[key + 'result']:
            return_dict[key + 'trained_pipeline'] = trained_pipeline

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
                 adversarial_robustness_constraint=None
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


    def get_best_pipeline(self):
        try:
            return self.study.best_trial.user_attrs['pipeline']
        except:
            return None


    def predict(self, X):
        best_pipeline = self.get_best_pipeline()
        return best_pipeline.predict(X)


    def fit(self, X_new, y_new, sample_weight=None, categorical_indicator=None, scorer=None):
        self.start_fitting = time.time()

        if self.sample_fraction < 1.0:
            X, _, y, _ = sklearn.model_selection.train_test_split(X_new, y_new, random_state=42, stratify=y_new, train_size=self.sample_fraction)
        else:
            X = X_new
            y = y_new

        def objective1(trial):
            start_total = time.time()

            key = 'My_automl' + self.random_key + 'My_process' + str(time.time()) + "##" + str(
                np.random.randint(0, 1000))

            manager = multiprocessing.Manager()
            return_dict = manager.dict()

            result = 0

            try:

                self.space.trial = trial

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
                categorical_transformer = Pipeline([('removeNAN', CategoricalMissingTransformer()), ('onehot_transform', onehot_transformer)])


                my_transformers = []
                if np.sum(np.invert(categorical_indicator)) > 0:
                    my_transformers.append(('num', numeric_transformer, np.invert(categorical_indicator)))

                if np.sum(categorical_indicator) > 0:
                    my_transformers.append(('cat', categorical_transformer, categorical_indicator))


                data_preprocessor = ColumnTransformer(transformers=my_transformers)

                my_pipeline = Pipeline([('data_preprocessing', data_preprocessor), ('preprocessing', preprocessor), ('augmentation', augmentation),
                              ('classifier', classifier)])



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
                return_dict['trial']  = trial

                try:
                    return_dict['study_best_value'] = self.study.best_value
                except ValueError:
                    return_dict['study_best_value'] = -np.inf

                already_used_time = time.time() - self.start_fitting

                if already_used_time + 2 >= self.time_search_budget:  # already over budget
                    time.sleep(2)
                    return -1 * (time.time() - start_total)

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
                if key + 'intermediate_steps' in return_dict:
                    intermediate_steps = return_dict[key + 'intermediate_steps']
                    for inter_i in range(len(intermediate_steps)):
                        max_score = intermediate_steps[inter_i][0]
                        budget = intermediate_steps[inter_i][1]
                        trial.report(max_score, budget)

                result = -1 * (time.time() - start_total)
                if key + 'result' in return_dict:
                    result = return_dict[key + 'result']

                if key + 'result' + '_training_time' in return_dict:
                    trial.set_user_attr('training_time', return_dict[key + 'result' + '_training_time'])
                if key + 'result' + '_pipeline_size' in return_dict:
                    trial.set_user_attr('pipeline_size', return_dict[key + 'result' + '_pipeline_size'])
                if key + 'result' + '_inference_time' in return_dict:
                    trial.set_user_attr('inference_time', return_dict[key + 'result' + '_inference_time'])

                trial.set_user_attr('evaluation_time', time.time() - start_total)
                trial.set_user_attr('time_since_start', time.time() - self.start_fitting)

                if result > 0:
                    if key + 'trained_pipeline' in return_dict:
                        if return_dict['study_best_value'] < result:
                            trial.set_user_attr('pipeline', return_dict[key + 'trained_pipeline'])


            except Exception as e:
                print(str(e) + '\n\n')
                return -1 * np.inf

            if key + 'pruned' in return_dict and return_dict[key + 'pruned']:
                raise optuna.TrialPruned()
            elif np.isneginf(result):
                raise optuna.TrialPruned()
            else:
                return result

        if type(self.study) == type(None):
            self.study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        self.study.optimize(objective1, timeout=self.time_search_budget,
                            n_jobs=self.n_jobs,
                            catch=(TimeException,),
                            callbacks=[StopWhenOptimumReachedCallback(1.0)]) # todo: check for scorer to know what is the optimum

        #clean up all remaining pipelines
        for my_path in glob.glob('/tmp/my_pipeline' + 'My_automl' + self.random_key + 'My_process' + '*.p'):
            os.remove(my_path)

        return self.study.best_value




if __name__ == "__main__":
    # auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
    from sklearn.metrics import balanced_accuracy_score

    auc = make_scorer(balanced_accuracy_score)

    # dataset = openml.datasets.get_dataset(1114)

    dataset = openml.datasets.get_dataset(1116)
    #dataset = openml.datasets.get_dataset(187)  # 51

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )

    print(X.shape)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, stratify=y,
                                                                                train_size=0.6)

    gen = SpaceGenerator()
    space = gen.generate_params()

    # print(space.generate_additional_features_v1(start=True))
    print(len(space.name2node))
    # print(space)
    new_list = list()
    space.generate_additional_features_v2(start=True, sum_list=new_list)
    print(new_list)
    print(len(new_list))

    new_list = list()
    space.generate_additional_features_v2_name(start=True, sum_list=new_list)
    print(new_list)
    print(len(new_list))

    from anytree import RenderTree

    for pre, _, node in RenderTree(space.parameter_tree):
        print("%s%s: %s" % (pre, node.name, node.status))

    search = MyAutoML(n_jobs=1,
                      time_search_budget=2*60,
                      space=space,
                      main_memory_budget_gb=40,
                      hold_out_fraction=0.3)

    begin = time.time()

    best_result = search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=auc)

    from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import show_progress

    show_progress(search, X_test, y_test, auc)

    # importances = optuna.importance.get_param_importances(search.study)
    # print(importances)

    test_score = auc(search.get_best_pipeline(), X_test, y_test)

    print("result: " + str(best_result) + " test: " + str(test_score))

    print(time.time() - begin)