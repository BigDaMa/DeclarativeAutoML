from imblearn.pipeline import Pipeline
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import openml
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.CategoricalMissingTransformer import CategoricalMissingTransformer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.compose import ColumnTransformer
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.cbays.ax_version.Space_GenerationTreeBalanceConstrained import SpaceGenerator
from sklearn.model_selection import StratifiedKFold
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
import sys
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.MyIdentity import IdentityTransformation
import numpy as np
from ax import *
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.SimpleImputerOptuna import SimpleImputerOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.QuadraticDiscriminantAnalysisOptuna import QuadraticDiscriminantAnalysisOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.PassiveAggressiveOptuna import PassiveAggressiveOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.KNeighborsClassifierOptuna import KNeighborsClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.HistGradientBoostingClassifierOptuna import HistGradientBoostingClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.private.PrivateLogisticRegressionOptuna import PrivateLogisticRegressionOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.private.PrivateGaussianNBOptuna import PrivateGaussianNBOptuna
from ax.models.numpy.randomforest import RandomForest

class TimeException(Exception):
    def __init__(self, message="Time is over!"):
        self.message = message
        super().__init__(self.message)


def calculate_class_weight(y, custom_weight):
    class_weight = {}
    unique_counts = np.unique(y, return_counts=True)
    sorted_ids = np.argsort(unique_counts[1])
    class_weight[unique_counts[0][sorted_ids[0]]] = custom_weight / unique_counts[1][sorted_ids[0]]
    class_weight[unique_counts[0][sorted_ids[1]]] = (1.0 - custom_weight) / unique_counts[1][sorted_ids[1]]
    return class_weight

def evaluatePipeline(key, return_dict):
    try:
        class_weighting = mp_global.mp_store[key]['class_weighting']
        custom_weighting = mp_global.mp_store[key]['custom_weighting']
        custom_weight = mp_global.mp_store[key]['custom_weight']

        p = mp_global.mp_store[key]['p']
        number_of_cvs = mp_global.mp_store[key]['number_of_cvs']
        cv = mp_global.mp_store[key]['cv']
        scorer = mp_global.mp_store[key]['scorer']
        X = mp_global.mp_store[key]['X']
        y = mp_global.mp_store[key]['y']
        main_memory_budget_gb = mp_global.mp_store[key]['main_memory_budget_gb']
        hold_out_fraction = mp_global.mp_store[key]['hold_out_fraction']

        training_time_limit = mp_global.mp_store[key]['training_time_limit']
        inference_time_limit = mp_global.mp_store[key]['inference_time_limit']
        pipeline_size_limit = mp_global.mp_store[key]['pipeline_size_limit']
        adversarial_robustness_constraint = mp_global.mp_store[key]['adversarial_robustness_constraint']

        size = int(main_memory_budget_gb * 1024.0 * 1024.0 * 1024.0)
        resource.setrlimit(resource.RLIMIT_AS, (size, resource.RLIM_INFINITY))

        #todo: cancel fitting if over time
        start_training = time.time()
        p.fit(X, y)

        training_time = time.time() - start_training
        return_dict[key + 'result' + '_training_time'] = training_time
        if type(training_time_limit) != type(None) and training_time > training_time_limit:
            return_dict[key + 'result'] = 0.0
            return

        dumped_obj = pickle.dumps(p)
        pipeline_size = sys.getsizeof(dumped_obj)
        return_dict[key + 'result' + '_pipeline_size'] = pipeline_size
        if type(pipeline_size_limit) != type(None) and pipeline_size > pipeline_size_limit:
            return_dict[key + 'result'] = 0.0
            return

        inference_times = []
        for i in range(10):
            random_id = np.random.randint(low=0, high=X.shape[0])
            start_inference = time.time()
            p.predict(X[[random_id]])
            inference_times.append(time.time() - start_inference)
        return_dict[key + 'result' + '_inference_time'] = np.mean(inference_times)
        if type(inference_time_limit) != type(None) and np.mean(inference_times) > inference_time_limit:
            return_dict[key + 'result'] = 0.0
            return


        trained_pipeline = copy.deepcopy(p)
        scores = []

        if type(hold_out_fraction) == type(None):
            for cv_num in range(number_of_cvs):
                my_splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(time.time())).split(X, y)
                #my_splits = StratifiedKFold(n_splits=cv, shuffle=True, random_state=int(42)).split(X, y)
                for train_ids, test_ids in my_splits:
                    p.fit(X[train_ids, :], y[train_ids])

                    scores.append(scorer(p, X[test_ids, :], pd.DataFrame(y[test_ids])))
        else:
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, stratify=y,
                                                                  test_size=hold_out_fraction)
            p.fit(X_train, y_train)
            scores.append(scorer(p, X_test, pd.DataFrame(y_test)))

        return_dict[key + 'result'] = np.mean(scores)

        if mp_global.mp_store[key]['study_best_value'] < return_dict[key + 'result']:
            with open('/tmp/my_pipeline' + str(key) + '.p', "wb") as pickle_pipeline_file:
                pickle.dump(trained_pipeline, pickle_pipeline_file)

    except Exception as e:
        return_dict[key + 'result'] = 0.0
        print(p)
        print(str(e) + '\n\n')





class ConstrainedBayesianOpt:
    def __init__(self, cv=5,
                 number_of_cvs=1,
                 hold_out_fraction=None,
                 evaluation_budget=np.inf,
                 time_search_budget=10*60,
                 n_jobs=1,
                 space=None,
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
                custom_weight = 0.5

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
                            custom_weight = self.space.suggest_uniform('custom_weight', 0.00000001, 1.0)

                if class_weighting:
                    classifier.set_weight(custom_weight)

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

                print(my_pipeline)


                key = 'My_automl' + self.random_key + 'My_process' + str(time.time()) + "##" + str(np.random.randint(0,1000))

                mp_global.mp_store[key] = {}

                mp_global.mp_store[key]['class_weighting'] = class_weighting
                mp_global.mp_store[key]['custom_weighting'] = parameterization.get("custom_weighting")
                mp_global.mp_store[key]['custom_weight'] = parameterization.get("custom_weight")
                mp_global.mp_store[key]['p'] = copy.deepcopy(my_pipeline)
                mp_global.mp_store[key]['number_of_cvs'] = self.number_of_cvs
                mp_global.mp_store[key]['cv'] = self.cv
                mp_global.mp_store[key]['scorer'] = scorer
                mp_global.mp_store[key]['X'] = X
                mp_global.mp_store[key]['y'] = y
                mp_global.mp_store[key]['main_memory_budget_gb'] = self.main_memory_budget_gb
                mp_global.mp_store[key]['hold_out_fraction'] = self.hold_out_fraction
                mp_global.mp_store[key]['training_time_limit'] = self.training_time_limit
                mp_global.mp_store[key]['inference_time_limit'] = self.inference_time_limit
                mp_global.mp_store[key]['pipeline_size_limit'] = self.pipeline_size_limit
                mp_global.mp_store[key]['adversarial_robustness_constraint'] = self.adversarial_robustness_constraint

                print('class_weighting: ' + str(mp_global.mp_store[key]['class_weighting']) + ' custom_weighting: ' + str(mp_global.mp_store[key]['custom_weighting'])+ ' custom_weight: ' + str(mp_global.mp_store[key]['custom_weight'])+ ' augmentation: ' + str(parameterization.get('augmentation')))

                result_dictionary = {}

                mp_global.mp_store[key]['study_best_value'] = -np.inf

                already_used_time = time.time() - self.start_fitting

                if already_used_time + 2 >= self.time_search_budget:  # already over budget
                    time.sleep(2)
                    #result_dictionary['accuracy'] = (-1 * np.inf, 0.0)
                    result_dictionary['accuracy'] = (0.0, 0.0)
                    return result_dictionary

                remaining_time = np.min([self.evaluation_budget, self.time_search_budget - already_used_time])

                manager = multiprocessing.Manager()
                return_dict = manager.dict()
                my_process = multiprocessing.Process(target=evaluatePipeline, name='start'+key, args=(key, return_dict,))
                my_process.start()
                my_process.join(int(remaining_time))

                # If thread is active
                while my_process.is_alive():
                    # Terminate foo
                    my_process.terminate()
                    my_process.join()

                del mp_global.mp_store[key]

                result = -1 * (time.time() - start_total)
                print('result: ' + str(return_dict[key + 'result']))
                if key + 'result' in return_dict:
                    result_dictionary['accuracy'] = (return_dict[key + 'result'], 0.0)



                if key + 'result' + '_training_time' in return_dict:
                    result_dictionary['training_time'] = (return_dict[key + 'result' + '_training_time'], 0.0)
                else:
                    result_dictionary['training_time'] = (0.0, 0.0)

                if key + 'result' + '_pipeline_size' in return_dict:
                    result_dictionary['pipeline_size'] = (return_dict[key + 'result' + '_pipeline_size'], 0.0)
                else:
                    result_dictionary['pipeline_size'] = (0.0, 0.0)

                if key + 'result' + '_inference_time' in return_dict:
                    result_dictionary['inference_time'] = (return_dict[key + 'result' + '_inference_time'], 0.0)
                else:
                    result_dictionary['inference_time'] = (0.0, 0.0)

                result_dictionary['evaluation_time'] = (time.time() - start_total, 0.0)
                #result_dictionary['time_since_start'] = (time.time() - self.start_fitting, 0.0)


                if result > 0:
                    pickle_file_name = '/tmp/my_pipeline' + str(key) + '.p'
                    try:
                        if os.path.exists(pickle_file_name):
                            if self.study.best_value < result:
                                with open(pickle_file_name, "rb") as pickle_pipeline_file:
                                    #result_dictionary['pipeline'] = pickle.load(pickle_pipeline_file)
                                    pass
                            os.remove(pickle_file_name)
                    except:
                        if os.path.exists(pickle_file_name):
                            with open(pickle_file_name, "rb") as pickle_pipeline_file:
                                #result_dictionary['pipeline'] = pickle.load(pickle_pipeline_file)
                                pass
                            os.remove(pickle_file_name)

                return result_dictionary
            except Exception as e:
                print(str(e) + '\n\n')
                # result_dictionary['accuracy'] = (-1 * np.inf, 0.0)
                result_dictionary['accuracy'] = (0.0, 0.0)
                return result_dictionary

        exp = SimpleExperiment(
            name="constrained_Ax",
            search_space=self.space.get_space(),
            evaluation_function=objective1,
            objective_name='accuracy',
            minimize=False
        )
        #exp.add_tracking_metric(Metric('evaluation_time', lower_is_better=True))
        #exp.add_tracking_metric(Metric('time_since_start'))

        random_sampling_start = time.time()

        sobol = Models.SOBOL(exp.search_space)
        for i in range(5):
            exp.new_trial(generator_run=sobol.gen(1))

        print('random sampling: ' + str(time.time() - random_sampling_start))

        already_used_time = time.time() - self.start_fitting
        while already_used_time + 2 < self.time_search_budget:
            #try:
            eval_start = time.time()
            eval_data = exp.eval()
            print('eval time: ' + str(time.time() - eval_start))
            model_start = time.time()

            #from ax.modelbridge.factory import get_empirical_bayes_thompson
            #gpei = get_empirical_bayes_thompson(exp, eval_data)

            '''
            from ax.modelbridge.numpy import NumpyModelBridge
            from ax.modelbridge.transforms.unit_x import UnitX
            from ax.modelbridge.transforms.standardize_y import StandardizeY
            gpei = NumpyModelBridge(
                experiment=exp,
                search_space=self.space.get_space(),
                data=eval_data,
                model=RandomForest(),
                transforms=[UnitX, StandardizeY],  # Include the desired set of transforms
            )
            '''

            gpei = Models.GPEI(experiment=exp, data=eval_data)
            print('model time: ' + str(time.time() - model_start))
            bayesian_sampling_start = time.time()
            batch = exp.new_trial(generator_run=gpei.gen(1))
            print('bayesian_sampling time: ' + str(time.time() - bayesian_sampling_start))
            #except:
            #    break
            already_used_time = time.time() - self.start_fitting

        #clean up all remaining pipelines
        for my_path in glob.glob('/tmp/my_pipeline' + 'My_automl' + self.random_key + 'My_process' + '*.p'):
            os.remove(my_path)

        data = exp.fetch_data().df.values
        cur_max = 0.0
        arm_max = None
        for i in range(len(data)):
            if data[i, 1] == 'accuracy' and data[i, 2] > cur_max:
                    cur_max = data[i, 2]
                    arm_max = data[i, 0]

        print('cur_max: ' + str(cur_max))

        return cur_max




if __name__ == "__main__":
    auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

    #dataset = openml.datasets.get_dataset(1114)

    #dataset = openml.datasets.get_dataset(1116)
    dataset = openml.datasets.get_dataset(1464)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )

    print(X.shape)


    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, stratify=y, train_size=0.6)

    gen = SpaceGenerator()
    space = gen.generate_params()

    #for pre, _, node in RenderTree(space.parameter_tree):
    #    print("%s%s: %s" % (pre, node.name, node.status))

    search = ConstrainedBayesianOpt(n_jobs=1,
                      time_search_budget=60*60*3,
                      space=space,
                      main_memory_budget_gb=40,
                      hold_out_fraction=0.5)

    begin = time.time()

    best_result = search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=auc)

    print(best_result)

    #from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import show_progress
    #show_progress(search, X_test, y_test, auc)

    #importances = optuna.importance.get_param_importances(search.study)
    #print(importances)




    test_score = auc(search.get_best_pipeline(), X_test, y_test)

    print("result: " + str(best_result) + " test: " + str(test_score))

    print(time.time() - begin)