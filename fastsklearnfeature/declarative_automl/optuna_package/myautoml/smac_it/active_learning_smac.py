from sklearn.metrics import make_scorer
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
from sklearn.metrics import f1_score
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.smac_it.CustomRandomForest import CustomRandomForest

from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.epm.base_epm import AbstractEPM
import numpy as np
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from smac.initial_design.latin_hypercube_design import LHDesign

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition

from smac.configspace import ConfigurationSpace
from smac.intensification.simple_intensifier import SimpleIntensifier
import pickle
import openml
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.smac_it.my_objective import run_AutoML

if __name__ == '__main__':
    openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
    openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'


    def predict_range(model, X):
        y_pred = model.predict(X)
        return y_pred

    #test data

    test_holdout_dataset_id = [1134, 1495, 41147, 316, 1085, 1046, 1111, 55, 1116, 448, 1458, 162, 1101, 1561, 1061, 1506, 1235, 4135, 151, 51, 41138, 40645, 1510, 1158, 312, 38, 52, 1216, 41007, 1130]

    my_scorer = make_scorer(f1_score)

    total_search_time = 1*60#60*60

    my_openml_datasets = [3, 4, 13, 15, 24, 25, 29, 31, 37, 38, 40, 43, 44, 49, 50, 51, 52, 53, 55, 56, 59, 151, 152, 153, 161, 162, 164, 172, 179, 310, 311, 312, 316, 333, 334, 335, 336, 337, 346, 444, 446, 448, 450, 451, 459, 461, 463, 464, 465, 466, 467, 470, 472, 476, 479, 481, 682, 683, 747, 803, 981, 993, 1037, 1038, 1039, 1040, 1042, 1045, 1046, 1048, 1049, 1050, 1053, 1054, 1055, 1056, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1071, 1073, 1075, 1085, 1101, 1104, 1107, 1111, 1112, 1114, 1116, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1169, 1216, 1235, 1236, 1237, 1238, 1240, 1412, 1441, 1442, 1443, 1444, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1455, 1458, 1460, 1461, 1462, 1463, 1464, 1467, 1471, 1473, 1479, 1480, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1494, 1495, 1496, 1498, 1502, 1504, 1506, 1507, 1510, 1511, 1547, 1561, 1562, 1563, 1564, 1597, 4134, 4135, 4154, 4329, 4534, 23499, 40536, 40645, 40646, 40647, 40648, 40649, 40650, 40660, 40665, 40666, 40669, 40680, 40681, 40690, 40693, 40701, 40705, 40706, 40710, 40713, 40714, 40900, 40910, 40922, 40999, 41005, 41007, 41138, 41142, 41144, 41145, 41146, 41147, 41150, 41156, 41158, 41159, 41160, 41161, 41162, 41228, 41430, 41521, 41538, 41976, 42172, 42477]
    for t_v in test_holdout_dataset_id:
        my_openml_datasets.remove(t_v)

    #my_openml_datasets = my_openml_datasets[0:10]

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


    #random_feature_selection_p = UniformFloatHyperparameter('random_feature_selection', 0, 1, default_value=1)
    #cs.add_hyperparameters([random_feature_selection_p])

    unbalance_data_p = CategoricalHyperparameter('unbalance_data', [True, False], default_value=False)
    fraction_ids_class0_p = UniformFloatHyperparameter('fraction_ids_class0', 0, 1, default_value=1)
    fraction_ids_class1_p = UniformFloatHyperparameter('fraction_ids_class1', 0, 1, default_value=1)
    sampling_factor_train_only_p = UniformFloatHyperparameter('sampling_factor_train_only', 0, 1, default_value=1)
    cs.add_hyperparameters([unbalance_data_p, fraction_ids_class0_p, fraction_ids_class1_p, sampling_factor_train_only_p])

    cs.add_condition(EqualsCondition(fraction_ids_class0_p, unbalance_data_p, True))
    cs.add_condition(EqualsCondition(fraction_ids_class1_p, unbalance_data_p, True))
    cs.add_condition(EqualsCondition(sampling_factor_train_only_p, unbalance_data_p, False))

    gen = SpaceGenerator()
    space = gen.generate_params()
    space.generate_SMAC(cs)

    print(cs.get_hyperparameter_names())
    print(cs.get_idx_by_hyperparameter_name('dataset_id'))

    with open('/tmp/smac_space.p', "wb") as pickle_model_file:
        pickle.dump(cs, pickle_model_file)

    def compute_uncertainty(X: np.ndarray, model) -> np.ndarray:
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        results = np.empty((X.shape[0], 1))
        results[:] = 0.0
        for di in range(X.shape[0]):
            try:
                m, var_ = model.predict(X[di])
                results[di] = var_
            except:
                pass
        return results


    my_aquisition = compute_uncertainty


    class MyAquisitionFunction(AbstractAcquisitionFunction):
        def __init__(self,
                     model: AbstractEPM):

            super(MyAquisitionFunction, self).__init__(model)
            self.long_name = 'MyAquisitionFunction'
            self.num_data = None
            self._required_updates = ('model', 'num_data')
            self.count = 0

        def _compute(self, X: np.ndarray) -> np.ndarray:
            return my_aquisition(X, self.model)




    # Scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": 300,  # max. number of function evaluations; for this example set to a low number
                         "cs": cs,  # configuration space
                         "deterministic": "true"
                         })


    smac = SMAC4HPO(scenario=scenario,
                    rng=np.random.RandomState(42),
                    tae_runner=run_AutoML,
                    runhistory2epm=RunHistory2EPM4Cost,
                    acquisition_function=MyAquisitionFunction,
                    initial_design=LHDesign,
                    initial_design_kwargs={'n_configs_x_params': 2},
                    model=CustomRandomForest,
                    intensifier=SimpleIntensifier,
                    model_kwargs={'num_trees': 1000,
                                      'do_bootstrapping': True,
                                      'ratio_features': 1.0,
                                      'min_samples_split': 2,
                                      'min_samples_leaf': 1
                                      },
                    n_jobs=2
                    )

    smac.optimize()







