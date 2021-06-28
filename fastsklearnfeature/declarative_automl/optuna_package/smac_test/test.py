from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.epm.base_epm import AbstractEPM
from smac.epm.rf_with_instances import RandomForestWithInstances
import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.runhistory.runhistory2epm import RunHistory2EPM4InvScaledCost
from smac.scenario.scenario import Scenario


def compute_uncretainty(X: np.ndarray, model) -> np.ndarray:
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    m, var_ = model.predict(X)
    print('hello unc')
    return var_

def compute_min(X: np.ndarray, model) -> np.ndarray:
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    m, var_ = model.predict(X)
    print('hello_min')
    return -m

my_aquisition = compute_uncretainty


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


if __name__ ==  '__main__':



    def toy_objective(x):
        x1 = x["x0"]
        val = (x1 - 2)** 2.
        return val

    cs = ConfigurationSpace()
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
    cs.add_hyperparameters([x0])

    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": 10,  # max. number of function evaluations; for this example set to a low number
                         "cs": cs,  # configuration space
                         "deterministic": "true",
                         })

    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=toy_objective,
                    runhistory2epm=RunHistory2EPM4InvScaledCost,
                    acquisition_function=MyAquisitionFunction,
                    model=RandomForestWithInstances,
                    model_kwargs={'num_trees': 1000,
                                  'do_bootstrapping': True,
                                  'ratio_features': 1.0,
                                  'min_samples_split': 2,
                                  'min_samples_leaf': 1
                                  }
                    )
    smac.optimize()

    print('Phase2: Maximize a different acquisition function for previously trained surrogate model')

    smac.solver.epm_chooser.acq_optimizer.n_sls_iterations = 1000
    smac.solver.scenario.acq_opt_challengers = 100000

    my_aquisition = compute_min

    intent, run_info = smac.solver.intensifier.get_next_run(
        challengers=smac.solver.initial_design_configs,
        incumbent=smac.solver.incumbent,
        chooser=smac.solver.epm_chooser,
        run_history=smac.solver.runhistory,
        repeat_configs=smac.solver.intensifier.repeat_configs,
        num_workers=smac.solver.tae_runner.num_workers(),
    )

    print(intent)
    print(run_info)

