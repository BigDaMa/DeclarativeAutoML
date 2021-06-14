from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.MyAutoMLProcessClassBalance import MyAutoML
import time
from sklearn.metrics import make_scorer
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
import copy
from anytree import RenderTree
from sklearn.metrics import f1_score
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_feature_names

from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.epm.base_epm import AbstractEPM
from smac.epm.rf_with_instances import RandomForestWithInstances
import numpy as np
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.runhistory.runhistory2epm import RunHistory2EPM4InvScaledCost
from smac.scenario.scenario import Scenario
from smac.initial_design.latin_hypercube_design import LHDesign

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition

from smac.configspace import ConfigurationSpace

def predict_range(model, X):
    y_pred = model.predict(X)
    return y_pred

#test data

test_holdout_dataset_id = [1134, 1495, 41147, 316, 1085, 1046, 1111, 55, 1116, 448, 1458, 162, 1101, 1561, 1061, 1506, 1235, 4135, 151, 51, 41138, 40645, 1510, 1158, 312, 38, 52, 1216, 41007, 1130]

my_scorer = make_scorer(f1_score)


total_search_time = 5*60#60*60
topk = 20 # 20

my_openml_datasets = [3, 4, 13, 15, 24, 25, 29, 31, 37, 38, 40, 43, 44, 49, 50, 51, 52, 53, 55, 56, 59, 151, 152, 153, 161, 162, 164, 172, 179, 310, 311, 312, 316, 333, 334, 335, 336, 337, 346, 444, 446, 448, 450, 451, 459, 461, 463, 464, 465, 466, 467, 470, 472, 476, 479, 481, 682, 683, 747, 803, 981, 993, 1037, 1038, 1039, 1040, 1042, 1045, 1046, 1048, 1049, 1050, 1053, 1054, 1055, 1056, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1071, 1073, 1075, 1085, 1101, 1104, 1107, 1111, 1112, 1114, 1116, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1169, 1216, 1235, 1236, 1237, 1238, 1240, 1412, 1441, 1442, 1443, 1444, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1455, 1458, 1460, 1461, 1462, 1463, 1464, 1467, 1471, 1473, 1479, 1480, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1494, 1495, 1496, 1498, 1502, 1504, 1506, 1507, 1510, 1511, 1547, 1561, 1562, 1563, 1564, 1597, 4134, 4135, 4154, 4329, 4534, 23499, 40536, 40645, 40646, 40647, 40648, 40649, 40650, 40660, 40665, 40666, 40669, 40680, 40681, 40690, 40693, 40701, 40705, 40706, 40710, 40713, 40714, 40900, 40910, 40922, 40999, 41005, 41007, 41138, 41142, 41144, 41145, 41146, 41147, 41150, 41156, 41158, 41159, 41160, 41161, 41162, 41228, 41430, 41521, 41538, 41976, 42172, 42477]
for t_v in test_holdout_dataset_id:
    my_openml_datasets.remove(t_v)

my_list_constraints = ['global_search_time_constraint',
                           'global_evaluation_time_constraint',
                           'global_memory_constraint',
                           'global_cv',
                           'global_number_cv',
                           'privacy',
                           'hold_out_fraction',
                           'sample_fraction',
                           'training_time_constraint',
                           'inference_time_constraint',
                           'pipeline_size_constraint']

feature_names, feature_names_new = get_feature_names(my_list_constraints)


def run_AutoML(x):
    # which hyperparameters to use
    gen = SpaceGenerator()
    space = gen.generate_params()
    space.sample_parameters_SMAC(x)

    #make this a hyperparameter
    search_time = x['global_search_time_constraint']

    evaluation_time = search_time
    if x['use_evaluation_time_constraint']:
        evaluation_time = x['global_evaluation_time_constraint']

    memory_limit = 10
    if x['use_search_memory_constraint']:
        memory_limit = x['global_memory_constraint']

    privacy_limit = None
    if x['use_privacy_constraint']:
        privacy_limit = x['privacy_constraint']

    training_time_limit = search_time
    if x['use_training_time_constraint']:
        training_time_limit = x['training_time_constraint']

    inference_time_limit = 60
    if x['use_inference_time_constraint']:
        inference_time_limit = x['inference_time_constraint']

    pipeline_size_limit = 350000000
    if x['use_pipeline_size_constraint']:
        pipeline_size_limit = x['pipeline_size_constraint']

    cv = 1
    number_of_cvs = 1
    hold_out_fraction = None
    if x['use_hold_out'] == False:
        cv = x['global_cv']
        if x['use_multiple_cvs']:
            number_of_cvs = x['global_number_cv']
    else:
        hold_out_fraction = x['hold_out_fraction']

    sample_fraction = 1.0
    if x['use_sampling']:
        sample_fraction = x['sample_fraction']

    dataset_id = x['dataset_id'] #get same random seed

    for pre, _, node in RenderTree(space.parameter_tree):
        if node.status == True:
            print("%s%s" % (pre, node.name))



    my_random_seed = int(time.time())

    X_train, X_test, y_train, y_test, categorical_indicator, attribute_names = get_data(dataset_id, randomstate=my_random_seed)

    dynamic_params = []
    static_params = []
    for random_i in range(5):
        search = MyAutoML(cv=cv,
                          number_of_cvs=number_of_cvs,
                          n_jobs=1,
                          evaluation_budget=evaluation_time,
                          time_search_budget=search_time,
                          space=space,
                          main_memory_budget_gb=memory_limit,
                          differential_privacy_epsilon=privacy_limit,
                          hold_out_fraction=hold_out_fraction,
                          sample_fraction=sample_fraction,
                          training_time_limit=training_time_limit,
                          inference_time_limit=inference_time_limit,
                          pipeline_size_limit=pipeline_size_limit)

        test_score = 0.0
        try:
            search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=my_scorer)

            best_pipeline = search.get_best_pipeline()
            if type(best_pipeline) != type(None):
                test_score = my_scorer(search.get_best_pipeline(), X_test, y_test)
        except:
            pass
        dynamic_params.append(test_score)

        # default params
        gen_new = SpaceGenerator()
        space_new = gen_new.generate_params()
        for pre, _, node in RenderTree(space_new.parameter_tree):
            if node.status == True:
                print("%s%s" % (pre, node.name))

        search_static = MyAutoML(n_jobs=1,
                          time_search_budget=search_time,
                          space=space_new,
                          evaluation_budget=int(0.1 * search_time),
                          main_memory_budget_gb=memory_limit,
                          differential_privacy_epsilon=privacy_limit,
                          hold_out_fraction=0.33,
                          training_time_limit=training_time_limit,
                          inference_time_limit=inference_time_limit,
                          pipeline_size_limit=pipeline_size_limit
                          )

        try:
            best_result = search_static.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=my_scorer)
            test_score_default = my_scorer(search_static.get_best_pipeline(), X_test, y_test)
        except:
            test_score_default = 0.0
        static_params.append(test_score_default)

    #this was a bad idea because it is not comparable across scenarios
    #comparison = np.mean(dynamic_params) - np.mean(static_params)

    #conservativ cost function
    all_results = copy.deepcopy(dynamic_params)
    all_results.extend(static_params)

    if np.max(all_results) <= 0.0:
        comparison = 0.0
    else:
        #comparison = np.square(float(np.quantile(dynamic_params, 0.1)) / float(np.max(all_results)))
        comparison = float(np.quantile(dynamic_params, 0.1)) / float(np.max(all_results))

    return comparison

#smac config space

cs = ConfigurationSpace()
cs.add_hyperparameter(UniformIntegerHyperparameter('global_search_time_constraint', 10, total_search_time, default_value=10))
cs.add_hyperparameter(CategoricalHyperparameter('dataset_id', my_openml_datasets, default_value=my_openml_datasets[0]))


use_evaluation_time_constraint_p = CategoricalHyperparameter('use_evaluation_time_constraint', [True, False], default_value=False)
global_evaluation_time_constraint_p = UniformIntegerHyperparameter('global_evaluation_time_constraint', 10, total_search_time, default_value=total_search_time)
cs.add_hyperparameters([use_evaluation_time_constraint_p, global_evaluation_time_constraint_p])
cs.add_condition(EqualsCondition(global_evaluation_time_constraint_p, use_evaluation_time_constraint_p, True))

use_search_memory_constraint_p = CategoricalHyperparameter('use_search_memory_constraint', [True, False], default_value=False)
global_memory_constraint_p = UniformFloatHyperparameter('global_memory_constraint', 0.00000000000001, 10, default_value=10, log=True)
cs.add_hyperparameters([global_memory_constraint_p, use_search_memory_constraint_p])
cs.add_condition(EqualsCondition(global_memory_constraint_p, use_search_memory_constraint_p, True))

use_privacy_constraint_p = CategoricalHyperparameter('use_privacy_constraint', [True, False], default_value=False)
privacy_constraint_p = UniformFloatHyperparameter('privacy_constraint', 0.0001, 10, default_value=10, log=True)
cs.add_hyperparameters([privacy_constraint_p, use_privacy_constraint_p])
cs.add_condition(EqualsCondition(privacy_constraint_p, use_privacy_constraint_p, True))

use_training_time_constraint_p = CategoricalHyperparameter('use_training_time_constraint', [True, False], default_value=False)
training_time_constraint_p = UniformFloatHyperparameter('training_time_constraint', 0.005, total_search_time, default_value=total_search_time, log=True)
cs.add_hyperparameters([training_time_constraint_p, use_training_time_constraint_p])
cs.add_condition(EqualsCondition(training_time_constraint_p, use_training_time_constraint_p, True))

use_inference_time_constraint_p = CategoricalHyperparameter('use_inference_time_constraint', [True, False], default_value=False)
inference_time_constraint_p = UniformFloatHyperparameter('inference_time_constraint', 0.0004, 60, default_value=60, log=True)
cs.add_hyperparameters([inference_time_constraint_p, use_inference_time_constraint_p])
cs.add_condition(EqualsCondition(inference_time_constraint_p, use_inference_time_constraint_p, True))

use_pipeline_size_constraint_p = CategoricalHyperparameter('use_pipeline_size_constraint', [True, False], default_value=False)
pipeline_size_constraint_p = UniformIntegerHyperparameter('pipeline_size_constraint', 2000, 350000000, default_value=350000000, log=True)
cs.add_hyperparameters([pipeline_size_constraint_p, use_pipeline_size_constraint_p])
cs.add_condition(EqualsCondition(pipeline_size_constraint_p, use_pipeline_size_constraint_p, True))

use_sampling_p = CategoricalHyperparameter('use_sampling', [True, False], default_value=False)
sample_fraction_p = UniformFloatHyperparameter('sample_fraction', 0, 1, default_value=1)
cs.add_hyperparameters([sample_fraction_p, use_sampling_p])
cs.add_condition(EqualsCondition(sample_fraction_p, use_sampling_p, True))

use_hold_out_p = CategoricalHyperparameter('use_hold_out', [True, False], default_value=True)
hold_out_fraction_p = UniformFloatHyperparameter('hold_out_fraction', 0, 1, default_value=1)

global_cv_p = UniformIntegerHyperparameter('global_cv', 2, 20, default_value=2)
use_multiple_cvs_p = CategoricalHyperparameter('use_multiple_cvs', [True, False], default_value=False)
global_number_cv_p = UniformIntegerHyperparameter('global_number_cv', 2, 10, default_value=2)

cs.add_hyperparameters([hold_out_fraction_p, use_hold_out_p, global_cv_p, use_multiple_cvs_p, global_number_cv_p])
cs.add_condition(EqualsCondition(hold_out_fraction_p, use_hold_out_p, True))
cs.add_condition(EqualsCondition(global_cv_p, use_hold_out_p, False))
cs.add_condition(EqualsCondition(use_multiple_cvs_p, use_hold_out_p, False))
cs.add_condition(EqualsCondition(global_number_cv_p, use_multiple_cvs_p, True))

gen = SpaceGenerator()
space = gen.generate_params()
space.generate_SMAC(cs)



class UncertaintySampling(AbstractAcquisitionFunction):
    def __init__(self,
                 model: AbstractEPM):

        super(UncertaintySampling, self).__init__(model)
        self.long_name = 'Uncertainty Sampling'
        self.num_data = None
        self._required_updates = ('model', 'num_data')

    def _compute(self, X: np.ndarray) -> np.ndarray:
        if self.num_data is None:
            raise ValueError('No current number of Datapoints specified. Call update('
                             'num_data=<int>) to inform the acquisition function '
                             'about the number of datapoints.')
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        #m, var_ = self.model.predict_marginalized_over_instances(X)
        m, var_ = self.model.predict(X)
        stdev = np.sqrt(var_)
        print(stdev)
        return stdev



# Scenario object
scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                     "runcount-limit": 100,  # max. number of function evaluations; for this example set to a low number
                     "cs": cs,  # configuration space
                     "deterministic": "true",
                     #"shared_model": True,
                     #"input_psmac_dirs": '/home/neutatz/phd2/smac/'
                     })


smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                tae_runner=run_AutoML,
                runhistory2epm=RunHistory2EPM4InvScaledCost,
                acquisition_function_optimizer_kwargs={'max_steps': 10},
                acquisition_function=UncertaintySampling,
                initial_design=LHDesign,
                initial_design_kwargs={'n_configs_x_params': 4, 'max_config_fracs': 0.1},
                model=RandomForestWithInstances,
                model_kwargs={'num_trees': 1000,
                              'do_bootstrapping': True,
                              'ratio_features': 1.0,
                              'min_samples_split': 2,
                              'min_samples_leaf': 1,
                              }
                )

smac.optimize()










