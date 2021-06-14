if __name__ ==  '__main__':
    """
    ==========================
    scipy-style fmin interface
    ==========================
    """
    from smac.optimizer.acquisition import AbstractAcquisitionFunction
    from smac.epm.base_epm import AbstractEPM
    from smac.epm.rf_with_instances import RandomForestWithInstances
    import numpy as np
    from smac.intensification.simple_intensifier import SimpleIntensifier

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
            print(var_)
            return var_


    import logging

    import numpy as np
    from ConfigSpace.hyperparameters import UniformFloatHyperparameter

    # Import ConfigSpace and different types of parameters
    from smac.configspace import ConfigurationSpace
    from smac.facade.smac_hpo_facade import SMAC4HPO
    from smac.initial_design.latin_hypercube_design import LHDesign
    from smac.runhistory.runhistory2epm import RunHistory2EPM4InvScaledCost
    # Import SMAC-utilities
    from smac.scenario.scenario import Scenario


    def rosenbrock_2d(x):
        """ The 2 dimensional Rosenbrock function as a toy model
        The Rosenbrock function is well know in the optimization community and
        often serves as a toy problem. It can be defined for arbitrary
        dimensions. The minimium is always at x_i = 1 with a function value of
        zero. All input parameters are continuous. The search domain for
        all x's is the interval [-5, 10].
        """
        x1 = x["x0"]
        x2 = x["x1"]

        val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
        return val


    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
    cs.add_hyperparameters([x0, x1])

    # Scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": 10,  # max. number of function evaluations; for this example set to a low number
                         "cs": cs,  # configuration space
                         "deterministic": "true",
                         #"shared_model": True,
                         #"input_psmac_dirs": '/home/neutatz/phd2/smac/'
                         })

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = rosenbrock_2d(cs.get_default_configuration())
    print("Default Value: %.2f" % def_value)


    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=rosenbrock_2d,
                    runhistory2epm=RunHistory2EPM4InvScaledCost,
                    acquisition_function_optimizer_kwargs={'max_steps': 100},
                    acquisition_function=UncertaintySampling,
                    model=RandomForestWithInstances,
                    model_kwargs={'num_trees': 1000,
                                  'do_bootstrapping': True,
                                  'ratio_features': 1.0,
                                  'min_samples_split': 2,
                                  'min_samples_leaf': 1,
                                  }
                    )

    smac.optimize()

    print(smac.solver.epm_chooser.acq_optimizer.acquisition_function.model)

    #print(smac.solver.epm_chooser.acq_optimizer.acquisition_function.model.predict(np.array([[0.5, 0.5], [0.75, 0.25]])))

    print('test')