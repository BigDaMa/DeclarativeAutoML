from fastsklearnfeature.declarative_automl.optuna_package.myautoml.smac_it.CustomRandomForestTest import CustomRandomForest
import numpy as np
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from smac.configspace import ConfigurationSpace
from smac.epm.rf_with_instances import RandomForestWithInstances

cs = ConfigurationSpace()
for i in range(5):
    cs.add_hyperparameter(UniformIntegerHyperparameter('f' + str(i), 0, 5, default_value=1))


def opt(x):
    my_list = []
    for i in range(5):
        my_list.append(x['f'+str(i)])

    if np.sum(my_list) > 10:
        return 0.0

    return np.product(my_list) * -1

# Scenario object
scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                     "runcount-limit": 200,  # max. number of function evaluations; for this example set to a low number
                     "cs": cs,  # configuration space
                     "deterministic": "true"
                     })


smac = SMAC4HPO(scenario=scenario,
                rng=np.random.RandomState(42),
                tae_runner=opt,
                runhistory2epm=RunHistory2EPM4Cost,
                model=CustomRandomForest,
                model_kwargs={'num_trees': 1000,
                                  'do_bootstrapping': True,
                                  'ratio_features': 1.0,
                                  'min_samples_split': 2,
                                  'min_samples_leaf': 1
                                  }
                )

incumbent = smac.optimize()

inc_value = opt(incumbent)
print("Optimized Value: %.2f" % (inc_value))